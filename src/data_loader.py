"""
Functions for loading assets and hazard maps
Assets are geodataframes of electricity stations and hazard maps are lists of file paths to .tif files.
"""
import logging
import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
import os

from config import get_model_extent, get_societal_access_config
from src.caching import get_asset_centroid_hash, load_simulation_caches
from src.dependency_knowledge_graph import build_default_knowledge_graph
from src.impacts import (
    create_voronoi_for_asset_type,
    prepare_land_use_impact_data,
    prepare_population_impact_data,
    update_voll_rates,
)
from src.societal_access import clip_population_to_service_area
from src.utils import build_voronoi_service_area_map, compile_asset_gdfs

# Data Loading Functions
def load_electricity_assets(electricity_dir, asset_types):
    """Load electricity assets from shapefile
    
    Args:
        electricity_dir (Path): Path to the directory containing electricity station shapefiles
        asset_types (list): List of asset types to load (e.g., ['ls', 'ms', 'msls'])

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of electricity assets with type column

    """
    # List station shapefiles in the electricity directory
    station_files = [
        f for f in os.listdir(electricity_dir) if f.endswith('.shp') and 'station' in f
    ]

    # Fix: Use exact matching with underscore to avoid 'ls' matching 'msls'
    # Sort asset_types by length (longest first) to prioritize longer matches
    asset_types_sorted = sorted(asset_types, key=len, reverse=True)
    
    matched_files = []
    for station_file in station_files:
        filename_prefix = station_file.split('_')[0]
        if filename_prefix in asset_types_sorted:
            matched_files.append(station_file)
    
    station_files = matched_files

    print(f"Found {len(station_files)} electricity station files matching types {asset_types}")
    
    # Add debug information to see what files were found
    print(f"All .shp files in directory: {[f for f in os.listdir(electricity_dir) if f.endswith('.shp')]}")
    print(f"Files with 'station': {[f for f in os.listdir(electricity_dir) if f.endswith('.shp') and 'station' in f]}")
    print(f"Final matched files: {station_files}")
    combined_assets = []
    
    for station_file in station_files:
        station_path = electricity_dir / station_file
        if station_path.exists():
            print(f"Loading electricity assets from {station_file}")
            gdf = gpd.read_file(station_path)
            
            # Ensure proper CRS
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Add type column based on filename, loading msls first for naming practicality
            if 'msls_' in station_file:
                gdf['type'] = 'msls'
            elif 'ms_' in station_file:
                gdf['type'] = 'ms'
            elif 'ls_' in station_file:
                gdf['type'] = 'ls'
            
            combined_assets.append(gdf)
            print(f"Loaded {len(gdf)} {gdf['type'].iloc[0]} assets")
    
    if combined_assets:
        # Combine all assets into a single GeoDataFrame
        gdf_assets = gpd.GeoDataFrame(pd.concat(combined_assets, ignore_index=True))
        print(f"Combined total: {len(gdf_assets)} electricity assets")
        print(f"Asset types: {gdf_assets['type'].value_counts().to_dict()}")
        return gdf_assets
    
    raise FileNotFoundError(f"No electricity station files found in {electricity_dir}")


def load_osm_assets(osm_asset_dir, asset_types):
    """Load real OSM-derived service assets (geometry + type) from the
    pre-extracted GeoPackage produced by
    ``book/preprocessing/extract_amenities_from_extent.ipynb``. Covers both
    healthcare facilities (hospital, primary_care, pharmacy, ems) and other
    service amenities that serve a distinct function (e.g. supermarket).

    Args:
        osm_asset_dir (Path): Directory containing ``osm_assets.gpkg``.
        asset_types (list): Asset ``type`` values to keep, e.g.
            ``['ems', 'hospital', 'primary_care', 'pharmacy', 'supermarket']``.
            The GeoPackage already stores the function category directly in
            ``type`` (classification happens at extraction time in
            ``book/preprocessing/extract_amenities_from_extent.ipynb``); pass
            ``None``/empty to keep all categories.

    Returns:
        gpd.GeoDataFrame: OSM assets with ``geometry`` and ``type`` columns,
        plus ``pupils`` when present (required for school weighted-Voronoi
        catchments; see config.py's ``weight_column``).
    """
    data_path = osm_asset_dir / "osm_assets.gpkg"
    osm_gdf = gpd.read_file(data_path)
    if asset_types:
        osm_gdf = osm_gdf[osm_gdf['type'].isin(asset_types)]
    keep_columns = ['geometry', 'type']
    if 'pupils' in osm_gdf.columns:
        keep_columns.append('pupils')
    return osm_gdf[keep_columns].reset_index(drop=True)


def load_hazard_maps(hazard_dir, max_days=None):
    """Find and load hazard map files from directory
    Args:
        hazard_dir (Path): Path to the directory containing hazard map .tif files
        max_days (int, optional): Maximum number of days to load. If None, all available days are loaded.

    Returns:
        list: List of file paths to hazard map .tif files
    """
    
    # Find all .tif files in hazard directory
    hazard_files = list(hazard_dir.glob("*.tif"))
    
    if not hazard_files:
        raise FileNotFoundError(f"No .tif hazard files found in {hazard_dir}")
    
    # Sort files to ensure consistent ordering
    # If more than 10 files, must be sorted based on integer value in filename, not alphabetically
    def extract_int_from_stem(path_obj):
        digits = ''.join(filter(str.isdigit, path_obj.stem))
        return int(digits) if digits else 0

    if len(hazard_files) > 9:
        hazard_files = sorted(hazard_files, key=extract_int_from_stem)
    else:
        hazard_files = sorted(hazard_files, key=lambda x: x.name)
    
    if max_days:
        hazard_files = hazard_files[:max_days]
    
    print(f"Found {len(hazard_files)} hazard map files")
    return [str(f) for f in hazard_files]


# Model inputs shared by the multi-map and single-map runners

LOGGER = logging.getLogger(__name__)

AMENITY_CATEGORY_LABELS = {
    "ems": "EMS",
    "hospital": "Hospitals",
    "primary_care": "Primary Care",
    "pharmacy": "Pharmacies",
    "supermarket": "Supermarkets",
}


def load_combined_assets(config):
    """Load full-extent electricity substations + OSM-derived service
    amenities and combine them into a single GeoDataFrame with a shared
    positional index.

    Returns:
        tuple: (gdf_assets_combined, substation_idx, osm_asset_idx)
    """
    gdf_electricity = load_electricity_assets(config["electricity_dir"], asset_types=["msls"])

    hc_assets = load_osm_assets(
        config["osm_asset_dir"],
        asset_types=list(AMENITY_CATEGORY_LABELS.keys()),
    ).to_crs("EPSG:4326")

    gdf_assets_combined = compile_asset_gdfs([gdf_electricity, hc_assets])

    substation_idx = list(gdf_assets_combined[gdf_assets_combined["type"] == "msls"].index)
    osm_asset_idx = list(gdf_assets_combined[gdf_assets_combined["type"].isin(AMENITY_CATEGORY_LABELS)].index)

    LOGGER.info(
        "Combined assets: %d (substations: %d, OSM service amenities: %d)",
        len(gdf_assets_combined), len(substation_idx), len(osm_asset_idx),
    )
    hc_counts = gdf_assets_combined.loc[osm_asset_idx, "type"].value_counts()
    for category, label in AMENITY_CATEGORY_LABELS.items():
        LOGGER.info("  %s: %d", label, int(hc_counts.get(category, 0)))

    return gdf_assets_combined, substation_idx, osm_asset_idx


def build_dependency_config(config, gdf_assets_combined, substation_voronoi, osm_asset_idx):
    """Attach the shared dependency knowledge graph (electricity -> healthcare)
    to a copy of *config*. The substation <-> OSM-asset Voronoi service-area
    map is also attached for traceability, even though the simulation's
    dependency engine derives its own provider assignment internally.
    """
    knowledge_graph = build_default_knowledge_graph()

    gdf_osm_assets_proj = gdf_assets_combined.loc[osm_asset_idx].to_crs(substation_voronoi.crs)
    service_area_map, _ = build_voronoi_service_area_map(
        substation_voronoi, gdf_osm_assets_proj, return_diagnostics=True,
    )

    config_combined = {**config}
    config_combined["dependency_parameters"] = {
        **config["dependency_parameters"],
        "knowledge_graph": knowledge_graph.to_config(),
        "service_area_map": service_area_map,
    }
    return config_combined, service_area_map


def prepare_population_and_land_use(config, gdf_assets_combined, substation_idx,
                                     substation_voronoi, root_dir, model_extent):
    """Build a flat {asset_id: population} exposure map for the electricity
    substations, the substation land-use/VOLL lookup used for monetary
    impact outcomes, and the population grid used by societal-access
    postprocessing.

    Only substations get a population entry: each one directly serves the
    population within its Voronoi cell. OSM service amenities (hospital,
    primary_care, pharmacy, ems, supermarket) are intentionally excluded --
    population-weighted access to them is computed separately by
    ``src.societal_access``, which
    builds its own population-to-function mapping from the population grid.
    Merging a substation-derived proxy population into this same map would
    double-count population already counted via the substation entry in the
    generic ``affected_population``/``served_population``/``total_population``
    outcomes.

    ``model_extent`` (config.get_model_extent) clips population, replacing
    stedin_area.geojson as the clip boundary.
    """
    population_buffer = gpd.GeoDataFrame(geometry=[model_extent], crs="EPSG:28992")

    pop_suffix = "_sample" if config.get("use_sample_population_data") else ""
    population_data_path = root_dir / "data" / "population" / f"cbs_vk100_2025_v1{pop_suffix}.gpkg"
    LOGGER.info("Loading population data from %s", population_data_path)
    population_data = gpd.read_file(
        population_data_path, driver="GPKG", bbox=tuple(model_extent.bounds),
    ).to_crs("EPSG:28992")

    pop_group_cols = list(config["service_node_config"]["population_groups"].values())

    # CBS encodes suppressed/unavailable group counts as negative sentinels
    # (e.g. -99997 secret, -99995 not available); clamp to 0.
    for col in pop_group_cols:
        population_data[col] = population_data[col].clip(lower=0)

    population_data = population_data[
        (population_data["aantal_inwoners"] > 0) & (~population_data.geometry.isna())
    ].dropna(subset=["geometry"])
    population_study_area = gpd.clip(population_data, population_buffer)
    # Clipping leaves zero-area line/point slivers where a cell only touches the extent edge;
    # they hold no population inside the extent and break area-weighting (division by cell area).
    population_study_area = population_study_area[population_study_area.geometry.area > 0]
    population_above_0 = population_study_area[["aantal_inwoners", "geometry"]].set_crs("EPSG:28992")

    # Population grid for societal-access postprocessing (demographic group
    # columns + a stable cell_id), independent of asset_population_map.
    population_for_societal = population_study_area[[*pop_group_cols, "geometry"]].set_crs("EPSG:28992")
    population_for_societal = population_for_societal.reset_index(drop=True)
    population_for_societal["cell_id"] = population_for_societal.index.astype(str)

    asset_population_map = prepare_population_impact_data(
        population_data=population_above_0, voronoi_gdf=substation_voronoi,
    )
    LOGGER.info(
        "asset_population_map ready for %d substations (total population: %.0f)",
        len(asset_population_map), sum(asset_population_map.values()),
    )

    # Land use / VOLL monetary-impact lookup, computed on the substation Voronoi only.
    land_use_suffix = "_sample" if config.get("use_sample_population_data") else ""
    land_use_path = root_dir / "data" / "land_use" / f"CBS_Publicatiebestand_BBG2017_v1{land_use_suffix}.gpkg"
    # Voronoi geometry hash keeps the cache from going stale when the extent
    # (e.g. model_extent_raw) changes but the land-use filename doesn't.
    voronoi_hash = get_asset_centroid_hash(substation_voronoi[["geometry"]])
    asset_to_lu_cache_path = config["interim_dir"] / f"{land_use_path.stem}_{voronoi_hash}_asset_to_lu.pkl"

    if asset_to_lu_cache_path.exists():
        LOGGER.info("Loading asset_to_lu from cache: %s", asset_to_lu_cache_path)
        with open(asset_to_lu_cache_path, "rb") as f:
            asset_to_lu = pickle.load(f)
    else:
        LOGGER.info("Computing asset_to_lu from scratch (land use x substation Voronoi intersection)...")
        land_use_data = gpd.read_file(land_use_path, driver="GPKG")
        _, voll_per_sqm = update_voll_rates(land_use_data)
        asset_land_use_map = prepare_land_use_impact_data(land_use_data, voronoi_gdf=substation_voronoi)

        asset_to_lu = {
            aid: [(lu_type, area, voll_per_sqm[lu_type])
                  for lu_type, area in lu_dict.items() if lu_type in voll_per_sqm]
            for aid, lu_dict in asset_land_use_map.items()
        }
        with open(asset_to_lu_cache_path, "wb") as f:
            pickle.dump(asset_to_lu, f)
        LOGGER.info("Saved asset_to_lu to cache: %s", asset_to_lu_cache_path)

    LOGGER.info("asset_to_lu ready with %d substations", len(asset_to_lu))
    return asset_population_map, asset_to_lu, population_for_societal


def prepare_model_inputs(config, checkpoint=lambda label: None):
    """Load assets, hazard maps, population/land use and caches shared by the multi-map and
    single-map runners. Returns a dict with ``config_combined``, ``gdf_assets``, ``hazard_maps``,
    ``asset_population_map``, ``asset_to_lu``, ``caches`` and ``societal_access_config``."""
    root_dir = Path(config["root_dir"])

    gdf_assets_combined, substation_idx, osm_asset_idx = load_combined_assets(config)
    checkpoint("combined assets loaded")

    LOGGER.info("Loading hazard maps from %s", config["hazard_dir"])
    hazard_maps = load_hazard_maps(config["hazard_dir"], max_days=None)
    LOGGER.info("Loaded %d hazard maps", len(hazard_maps))
    checkpoint("hazard maps loaded")

    # Voronoi generators padded beyond the true extent shape boundary cells
    # correctly; results are clipped to model_extent_raw before any downstream use.
    model_extent_raw = get_model_extent(config)
    model_extent_buffered = get_model_extent(config, buffer_frac=0.05)

    LOGGER.info("Building Voronoi service areas for electricity substations...")
    substation_voronoi = create_voronoi_for_asset_type(
        gdf_assets_combined, "msls", boundary=model_extent_buffered,
    )
    substation_voronoi_clipped = gpd.clip(substation_voronoi, model_extent_raw)
    checkpoint("substation Voronoi built")

    config_combined, service_area_map = build_dependency_config(
        config, gdf_assets_combined, substation_voronoi_clipped, osm_asset_idx,
    )
    checkpoint("dependency config + service-area map built")

    asset_population_map, asset_to_lu, population_for_societal = prepare_population_and_land_use(
        config_combined, gdf_assets_combined, substation_idx,
        substation_voronoi_clipped, root_dir, model_extent_raw,
    )
    population_for_societal = clip_population_to_service_area(
        population_for_societal, gdf_assets_combined,
    )
    checkpoint("population + land-use data ready")

    caches = load_simulation_caches(config["interim_dir"], config["hazard_dir"])
    LOGGER.info("Loaded caches: %s", [k for k in caches if caches[k] is not None])
    checkpoint("simulation caches loaded")

    societal_access_config = get_societal_access_config(
        config_combined, population_for_societal, caches=caches,
        asset_types=["msls", *AMENITY_CATEGORY_LABELS.keys()],
        namespace="interdependent_ci_societal_access",
    )
    LOGGER.info("Societal access functions: %s", societal_access_config["all_functions"])
    return {
        "config_combined": config_combined, "gdf_assets": gdf_assets_combined,
        "hazard_maps": hazard_maps, "asset_population_map": asset_population_map,
        "asset_to_lu": asset_to_lu, "caches": caches,
        "societal_access_config": societal_access_config,
    }
