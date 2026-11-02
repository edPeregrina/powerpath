"""
Full-extent EMA workbench run with layered adaptation measures.

Cleaned, CLI-runnable version of the exploratory ``interdependent_ci_societal_access.py``
notebook export (the script form of ``time_explicit_electricity_analysis.ipynb``).

It mirrors the combined-asset workflow demonstrated in ``Use_Case_sample.ipynb``
(electricity substations + OSM-derived service amenities, linked through the
shared dependency knowledge graph, with Voronoi-based population exposure) but
runs against the FULL extent dataset instead of the sample/development dataset:

  - Electricity substations: ``raw_data/ZH_Delfland/electricity`` (msls), as used
    in ``time_explicit_electricity_analysis.ipynb``.
  - OSM-derived service amenities (EMS / hospital / primary care / pharmacy /
    supermarket): the full-extent OpenStreetMap extraction produced by
    ``book/preprocessing/extract_amenities_from_extent.ipynb``
    (``config['osm_asset_dir']/osm_assets.gpkg``). Supermarkets serve a
    distinct function from the healthcare categories (food access rather than
    healthcare access) but share the same ``msls``-dependency/flooding rules
    as hospitals in the default knowledge graph.

Scope: this script sets up and runs ONLY the EMA workbench experiment with
adaptation measures (L1 area-based depth reduction + L2 asset-level barriers,
and their combinations). It intentionally omits:

  - exploratory plots/visualisations from the source notebooks,
  - generation of the L1/L2 adaptation GeoJSON files themselves (those are
    pre-generated artifacts already present under
    ``data/test_samples/adaptation/*``; this script only *reads* them).

Population access to each function is computed by ``src.societal_access``
(per-experiment, via the ``societal_access_config`` EMA constant built by
``config.get_societal_access_config``): electricity through substation
Voronoi service areas (no road access needed to receive electricity), and
every other function (EMS/hospital/primary care/pharmacy/supermarket)
through shared-island road-network reachability, since reaching them
requires an intact road connection. A shared SQLite-backed cache memoises
realized-state results across EMA experiments.

Repair-crew counts can be sampled independently per asset type (e.g. a wider
range for 'msls' substations than for 'hospital' facilities) via
``--repair-crews-range ASSET_TYPE MIN MAX`` (repeatable); see
``src.simulation.build_per_type_crew_uncertainties``.

Fragility-steepness (k) ranges follow the same per-asset-type pattern via
``--fragility-steepness-range ASSET_TYPE MIN MAX`` (repeatable; only
fragility-eligible types, currently 'msls'); see
``src.simulation.build_per_type_fragility_steepness_uncertainties``.

WARNING: a full run (default: 100 scenarios x ~150 policies) can take several
hours. Use ``--scenarios``/``--max-days``/``--dry-run`` for a quick smoke test
before committing to a full run.

Usage:
    python book/interdependent_ci_societal_access.py --dry-run
    python book/interdependent_ci_societal_access.py --scenarios 100
"""
import argparse
import itertools
import json
import logging
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    Policy,
    Samplers,
    SequentialEvaluator,
    TimeSeriesOutcome,
    ema_logging,
)
from ema_workbench.util.utilities import save_results

from config import (
    get_config,
    get_development_config,
    get_model_extent,
    get_societal_access_config,
    print_config_summary,
    setup_directories,
    validate_config,
)
from src.caching import (
    load_simulation_caches,
    flush_dirty_caches,
    save_hazard_extraction_cache,
    save_island_cache,
    save_overlap_cache,
    save_societal_allocation_cache,
    get_asset_centroid_hash,
)
from src.data_loader import load_electricity_assets, load_hazard_maps, load_osm_assets
from src.dependency_knowledge_graph import build_default_knowledge_graph
from src.impacts import (
    create_voronoi_for_asset_type,
    prepare_land_use_impact_data,
    prepare_population_impact_data,
    update_voll_rates,
)
from src.simulation import (
    build_per_type_crew_uncertainties,
    build_per_type_fragility_steepness_uncertainties,
    fragility_steepness_from_param,
    simulate_with_per_type_crews,
)
from src.societal_access import clip_population_to_service_area, list_societal_metric_names
from src.utils import build_voronoi_service_area_map, compile_asset_gdfs

LOGGER = logging.getLogger(__name__)

AMENITY_CATEGORY_LABELS = {
    "ems": "EMS",
    "hospital": "Hospitals",
    "primary_care": "Primary Care",
    "pharmacy": "Pharmacies",
    "supermarket": "Supermarkets",
}

# Aggregated 2D outcome variables (per-timestep, summed across assets).
KEEP_2D_VARS = ["flooded", "operational", "damage_ratio", "repair_time"]

# Default per-asset, per-timestep fields captured by --spatial-trace-policies
# (see run_spatial_trace). 'blocking_reason' is derived (see
# src.adaptation.BLOCKING_REASON_LABELS for the code -> label mapping).
DEFAULT_SPATIAL_TRACE_FIELDS = [
    "island_id", "operational", "intrinsic_operational",
    "accessible", "unreachable", "flooded", "blocking_reason",
]

MONETARY_CATEGORIES = [
    "residential",
    "commercial",
    "industrial",
    "transport",
    "public_sector",
]

# Asset types that may receive their own repair-crew uncertainty range.
# 'msls' = electricity substations; the rest are the OSM-derived amenity
# categories (healthcare + supermarket).
ASSET_TYPES_FOR_CREWS = ["msls", *AMENITY_CATEGORY_LABELS.keys()]

# Default (min, max) repair-crew ranges per asset type.
DEFAULT_CREW_RANGES = {
    "msls": (10, 20),
    "ems": (1, 3),
    "hospital": (1, 3),
    "primary_care": (1, 3),
    "pharmacy": (1, 3),
    "supermarket": (1, 1),
}

# Asset types that may receive their own fragility-steepness uncertainty
# range. Only types with an explicit hazard_blocks_operation=False flooding
# rule in the knowledge graph are fragility-eligible (see
# src.simulation._fragility_exclusions_from_knowledge_graph); currently only
# 'msls' substations.
ASSET_TYPES_FOR_FRAGILITY = ["msls"]

# Default (min, max) fragility-steepness (k) ranges per asset type, sampled
# in 0.5 increments (see src.simulation.fragility_steepness_uncertainty).
DEFAULT_FRAGILITY_STEEPNESS_RANGES = {
    "msls": (5.0, 7.5),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full-extent EMA workbench experiment (electricity + healthcare "
            "assets) with L1/L2 adaptation measures."
        ),
    )
    parser.add_argument("--scenarios", type=int, default=100,
                         help="Number of EMA uncertainty scenarios (Latin Hypercube samples). Default: 100.")
    parser.add_argument("--max-days", type=int, default=None,
                         help="Limit the number of hazard maps/days used (None = use all available days).")
    parser.add_argument("--major-timestep", type=int, default=6,
                         help="Number of sub-timesteps per day for the full-extent hazard data. Default: 6.")
    parser.add_argument(
        "--repair-crews-range", nargs=3, metavar=("ASSET_TYPE", "MIN", "MAX"), action="append",
        default=None,
        help=(
            "Override the repair-crew uncertainty range for one asset type, e.g. "
            "'--repair-crews-range msls 10 20'. Repeat for multiple asset types. "
            f"Valid asset types: {', '.join(ASSET_TYPES_FOR_CREWS)}. "
            f"Defaults: {DEFAULT_CREW_RANGES}."
        ),
    )
    parser.add_argument("--fragility-steepness-range", nargs=3, metavar=("ASSET_TYPE", "MIN", "MAX"),
                         action="append", default=None,
                         help=(
                             "Override the fragility-steepness (k) uncertainty range for one "
                             "fragility-eligible asset type, sampled in 0.5 increments, e.g. "
                             "'--fragility-steepness-range msls 5.0 7.5'. Repeat for multiple "
                             f"asset types. Valid asset types: {', '.join(ASSET_TYPES_FOR_FRAGILITY)}. "
                             f"Defaults: {DEFAULT_FRAGILITY_STEEPNESS_RANGES}."
                         ))
    parser.add_argument("--flood-threshold", type=float, default=0.2,
                         help="Flood depth threshold (m) above which an asset is considered flooded. Default: 0.2.")
    parser.add_argument("--adaptation-active-timesteps", type=int, default=198,
                         help="Number of leading timesteps during which L1/L2 adaptation measures are active. Default: 198.")
    parser.add_argument(
        "--l1-dir", type=Path, default=None,
        help="Directory with L1 (area-based) adaptation GeoJSON files. "
             "Default: <root_dir>/data/test_samples/adaptation/L1_peilgeb_policies_separate",
    )
    parser.add_argument(
        "--l2-dir", type=Path, default=None,
        help="Directory with L2 (asset-based) adaptation GeoJSON files. "
             "Default: <root_dir>/data/test_samples/adaptation/L2_asset_barriers",
    )
    parser.add_argument(
        "--max-l1-files", type=int, default=None,
        help="Limit the number of L1 adaptation files used (for quick smoke tests). Default: use all.",
    )
    parser.add_argument(
        "--max-l2-files", type=int, default=None,
        help="Limit the number of L2 adaptation files used (for quick smoke tests). Default: use all.",
    )
    parser.add_argument("--output-name", type=str, default=None,
                         help="Base filename (without extension) for the saved EMA results archive. "
                              "Default: ema_results_<execution_id>_full_extent_adaptation")
    parser.add_argument("--dry-run", action="store_true",
                         help="Build the model, caches and policy list, print a summary, and exit "
                              "without running any EMA experiments.")
    parser.add_argument("--profile", action="store_true",
                         help="Attach a SimulationTimingProfiler (shared across all experiments in "
                              "this run) and print phase/timestep timing summary tables after the "
                              "EMA experiments complete. Adds negligible overhead; intended for "
                              "quick smoke-test/optimisation runs (small --scenarios / policy counts).")
    parser.add_argument("--multiprocessing", type=int, default=None, metavar="N_PROCESSES",
                         help="Run experiments with EMA workbench's MultiprocessingEvaluator using "
                              "N_PROCESSES worker processes instead of SequentialEvaluator. CAUTION: "
                              "island/hazard/allocation caches are pickled once per worker at pool "
                              "start, so any NEW cache entries built by a worker (e.g. a not-yet-"
                              "cached road/hazard state) are NOT written back to the main process or "
                              "disk cache -- only safe to use once the on-disk caches already cover "
                              "every road/hazard state this run will touch (a prior warm sequential "
                              "run). Do not use for first-time/cold runs over new hazard maps.")
    parser.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity. Default: INFO.")
    parser.add_argument("--spatial-trace-policies", type=str, default=None, metavar="NAME1,NAME2,...",
                         help="OFF by default (skipped entirely unless set). Comma-separated exact "
                              "Policy names (see the 'Policies' log line for available names) to "
                              "additionally re-run with full per-asset, per-timestep capture "
                              "(island_id/operational/intrinsic_operational/accessible/unreachable/"
                              "flooded/blocking_reason) for GIS/spatial-trace export (e.g. to "
                              "reconstruct the animation notebook from real EMA output). Runs as a "
                              "small, separate pass after the main experiment batch -- does not "
                              "affect or slow down the main run's outcomes.")
    parser.add_argument("--spatial-trace-scenarios", type=int, default=1,
                         help="Number of independent scenario draws to capture per requested "
                              "spatial-trace policy. Default: 1 (one representative scenario).")
    parser.add_argument("--spatial-trace-fields", type=str, default=None, metavar="FIELD1,FIELD2,...",
                         help="Override the default per-asset fields captured for the spatial "
                              "trace. Default: island_id,operational,intrinsic_operational,"
                              "accessible,unreachable,flooded,blocking_reason.")
    parser.add_argument("--dev-config", action="store_true",
                         help="Use get_development_config() (data/test_samples) instead of the "
                              "full-extent get_config(). Intended for quick smoke tests/parity checks.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed numpy's global RNG before EMA sampling, for reproducible "
                              "scenario draws across runs (e.g. before/after comparisons).")
    parser.add_argument("--repair-priority", type=str, default="population impacts islands",
                         help="repair_crew_assignment_method used for the baseline and all L1/L2 "
                              "policies. Default: 'population impacts islands'.")
    return parser.parse_args()


def resolve_crew_ranges(args):
    """Merge CLI ``--repair-crews-range`` overrides into `DEFAULT_CREW_RANGES`.

    Returns:
        dict: ``{asset_type: (min_crews, max_crews)}`` for every asset type in
        `ASSET_TYPES_FOR_CREWS`.
    """
    crew_ranges = dict(DEFAULT_CREW_RANGES)
    for asset_type, min_crews, max_crews in args.repair_crews_range or []:
        if asset_type not in ASSET_TYPES_FOR_CREWS:
            raise ValueError(
                f"Unknown asset type '{asset_type}' in --repair-crews-range; "
                f"expected one of {ASSET_TYPES_FOR_CREWS}."
            )
        crew_ranges[asset_type] = (int(min_crews), int(max_crews))
    return crew_ranges


def resolve_fragility_steepness_ranges(args):
    """Merge CLI ``--fragility-steepness-range`` overrides into
    `DEFAULT_FRAGILITY_STEEPNESS_RANGES`.

    Returns:
        dict: ``{asset_type: (min_k, max_k)}`` for every asset type in
        `ASSET_TYPES_FOR_FRAGILITY`.
    """
    steepness_ranges = dict(DEFAULT_FRAGILITY_STEEPNESS_RANGES)
    for asset_type, min_k, max_k in args.fragility_steepness_range or []:
        if asset_type not in ASSET_TYPES_FOR_FRAGILITY:
            raise ValueError(
                f"Unknown asset type '{asset_type}' in --fragility-steepness-range; "
                f"expected one of {ASSET_TYPES_FOR_FRAGILITY}."
            )
        steepness_ranges[asset_type] = (float(min_k), float(max_k))
    return steepness_ranges


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
    ``src.societal_access`` (see this script's module docstring), which
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


def build_ema_model(config_combined, gdf_assets_combined, hazard_maps, caches,
                     asset_population_map, asset_to_lu, societal_access_config,
                     execution_id, args, crew_ranges, fragility_steepness_ranges,
                     profiler=None):
    """Construct the EMA workbench Model: uncertainties, constants and outcomes.

    *profiler*, when given a :class:`src.timing_profiler.SimulationTimingProfiler`,
    is attached as a shared ``Constant`` so every experiment in the run
    accumulates into the same phase/timestep timing tables (useful for
    identifying optimisation targets across a small diagnostic batch).
    """
    model = Model("ElectricitySocietalAccessSimulation",
                   function=simulate_with_per_type_crews)

    # Samples each asset type's fragility steepness (k) at 0.5 increments via
    # an IntegerParameter over k*2 (see
    # src.simulation.fragility_steepness_uncertainty).
    fragility_doubled_ranges = {
        asset_type: (round(min_k * 2), round(max_k * 2))
        for asset_type, (min_k, max_k) in fragility_steepness_ranges.items()
    }
    model.uncertainties = [
        *build_per_type_crew_uncertainties(crew_ranges),
        *build_per_type_fragility_steepness_uncertainties(fragility_doubled_ranges),
    ]
    for asset_type, (min_doubled, max_doubled) in fragility_doubled_ranges.items():
        LOGGER.info(
            "%s fragility steepness values: %s",
            asset_type,
            [fragility_steepness_from_param(v) for v in range(min_doubled, max_doubled + 1)],
        )

    model_constants = [
        Constant("flood_threshold", args.flood_threshold),
        Constant("gdf_assets", gdf_assets_combined),
        Constant("hazard_maps", hazard_maps[: args.max_days]),
        Constant("recovery_parameters", config_combined["recovery_parameters"]),
        Constant("root_dir", config_combined["root_dir"]),
        Constant("verbose", False),
        Constant("timestep_output", True),
        Constant("execution_id", execution_id),
        Constant("config", config_combined),
        Constant("major_timestep", args.major_timestep),
        # Cache parameters
        Constant("accessibility_cache", caches.get("accessibility_cache")),
        Constant("hazard_extraction_cache", caches.get("hazard_extraction_cache")),
        Constant("overlap_cache", caches.get("overlap_cache")),
        Constant("island_cache", caches.get("island_cache")),
        # Impact data
        Constant("asset_population_map", asset_population_map),
        Constant("asset_to_lu", asset_to_lu),
        # Societal access (electricity Voronoi + road-network reachability
        # for OSM service amenities)
        Constant("societal_access_config", societal_access_config),
        # Dimensionality control
        Constant("keep_3d_vars", []),
        Constant("keep_2d_vars", KEEP_2D_VARS),
    ]
    if profiler is not None:
        model_constants.append(Constant("profiler", profiler))
    model.constants = model_constants

    outcomes_list = [TimeSeriesOutcome(var) for var in KEEP_2D_VARS]
    outcomes_list.extend([
        TimeSeriesOutcome("timesteps"),
        TimeSeriesOutcome("affected_population"),
        TimeSeriesOutcome("served_population"),
        TimeSeriesOutcome("affected_population_ratio"),
        TimeSeriesOutcome("monetary_impact_total"),
    ])
    outcomes_list.extend(
        TimeSeriesOutcome(f"monetary_impact_{category}") for category in MONETARY_CATEGORIES
    )
    outcomes_list.extend(
        TimeSeriesOutcome(name) for name in list_societal_metric_names(
            all_functions=societal_access_config["all_functions"],
            pop_group_columns=societal_access_config["pop_group_columns"],
            reference_group=societal_access_config.get("reference_group", "total"),
        )
    )
    model.outcomes = outcomes_list

    return model


def _json_safe(obj):
    """Best-effort JSON-safe conversion for the run manifest.

    Config/CLI values routinely include ``Path``, ``set``/``frozenset`` and
    other objects ``json.dump`` cannot serialize directly (e.g. nested
    knowledge-graph dicts are already plain, but ``l1_dir``/``l2_dir`` etc.
    are ``Path``). Falls back to ``str()`` for anything else unrecognized
    rather than raising, since this manifest is a human-readable record, not
    a round-trippable format.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_run_manifest(out_dir, execution_id, input_parameters, output_parameters):
    """Write the run's input/output parameters as a single JSON manifest.

    Saved alongside the EMA results archive (``out_dir``) so a run's full
    configuration and outcome summary travel together when archived (e.g.
    copied to a results backup folder), instead of relying on manually
    captured console/log output.
    """
    manifest = {
        "execution_id": execution_id,
        "input_parameters": _json_safe(input_parameters),
        "output_parameters": _json_safe(output_parameters),
    }
    manifest_path = Path(out_dir) / f"run_manifest_{execution_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    return manifest_path


def run_spatial_trace(
    config_combined, gdf_assets_combined, hazard_maps, caches,
    asset_population_map, asset_to_lu, societal_access_config,
    policies, args, crew_ranges, fragility_steepness_ranges,
    execution_id, out_dir,
):
    """Re-run a small, explicitly-selected subset of policies with full
    per-asset, per-timestep capture for GIS/spatial-trace export.

    OFF by default (only called when ``--spatial-trace-policies`` is set).
    Reuses the already-loaded config/assets/hazard maps/caches from the main
    run (no reloading), but builds a *separate* Model instance so the main
    100-scenario batch's outcomes/Constants are completely unaffected --
    this never runs for the bulk of experiments, only for the handful of
    policies explicitly requested here.

    Writes one GeoParquet file per (scenario, policy) experiment under
    ``out_dir / "spatial_traces"`` (long format: one row per asset per
    timestep, plus geometry), and one static asset-to-nearest-road match
    table (see :func:`src.island_analysis.build_asset_road_match_table`).
    """
    import geopandas as gpd

    from src.island_analysis import build_asset_road_match_table

    requested_names = [n.strip() for n in args.spatial_trace_policies.split(",") if n.strip()]
    trace_policies = [p for p in policies if p.name in requested_names]
    missing = sorted(set(requested_names) - {p.name for p in trace_policies})
    if missing:
        LOGGER.warning(
            "--spatial-trace-policies requested unknown policy name(s): %s. Available: %s",
            missing, sorted(p.name for p in policies),
        )
    if not trace_policies:
        LOGGER.warning("No matching policies found for spatial trace capture; skipping.")
        return

    trace_fields = (
        [f.strip() for f in args.spatial_trace_fields.split(",") if f.strip()]
        if args.spatial_trace_fields else DEFAULT_SPATIAL_TRACE_FIELDS
    )

    LOGGER.info(
        "Running spatial trace capture for %d polic(y/ies) x %d scenario(s): fields=%s",
        len(trace_policies), args.spatial_trace_scenarios, trace_fields,
    )

    trace_model = build_ema_model(
        config_combined, gdf_assets_combined, hazard_maps, caches,
        asset_population_map, asset_to_lu, societal_access_config,
        execution_id, args, crew_ranges, fragility_steepness_ranges,
        profiler=None,
    )
    for constant in trace_model.constants:
        if constant.name == "keep_3d_vars":
            constant.value = trace_fields
        elif constant.name == "keep_2d_vars":
            # Disabled for the trace model: keep_2d_vars/keep_3d_vars share
            # result-array initialization keys in
            # simulate_asset_damage_recovery_access_breakdown_ema, so any
            # metric listed in both would have its 3D array silently
            # overwritten by a 2D (aggregated) one. The main run's 2D
            # aggregates are already captured in its own normal outcomes;
            # not needed here.
            constant.value = []
    _filtered = [oc for oc in trace_model.outcomes if oc.name not in KEEP_2D_VARS]
    _added = [TimeSeriesOutcome(field) for field in trace_fields]
    # Model.outcomes (a NamedObjectMapDescriptor) merges/extends by name on
    # assignment rather than replacing -- clear() first so trace fields that
    # share a name with a KEEP_2D_VARS outcome (e.g. 'operational',
    # 'flooded') genuinely get the new 3D TimeSeriesOutcome, not the old one.
    trace_model.outcomes.clear()
    trace_model.outcomes = _filtered + _added

    with SequentialEvaluator(trace_model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=args.spatial_trace_scenarios, policies=trace_policies,
            uncertainty_sampling=Samplers.LHS,
        )
    experiments_df = pd.DataFrame(experiments)

    trace_dir = Path(out_dir) / "spatial_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    asset_geometry = gdf_assets_combined[["geometry", "type"]].copy()
    asset_geometry["asset_id"] = asset_geometry.index
    n_assets = len(gdf_assets_combined)

    for exp_idx in range(len(experiments_df)):
        policy_name = experiments_df.loc[exp_idx, "policy"]
        scenario_name = experiments_df.loc[exp_idx, "scenario"]
        timesteps = outcomes["timesteps"][exp_idx]
        n_timesteps = len(timesteps)

        rows = {
            "timestep": np.repeat(timesteps, n_assets),
            "asset_id": np.tile(np.arange(n_assets), n_timesteps),
        }
        for field in trace_fields:
            rows[field] = np.asarray(outcomes[field][exp_idx]).reshape(n_timesteps * n_assets)

        trace_df = pd.DataFrame(rows)
        trace_gdf = trace_df.merge(asset_geometry, on="asset_id", how="left")
        trace_gdf = gpd.GeoDataFrame(trace_gdf, geometry="geometry", crs=gdf_assets_combined.crs)

        safe_policy = str(policy_name).replace(" ", "_")
        trace_path = trace_dir / f"spatial_trace_{execution_id}_{safe_policy}_scenario{scenario_name}.parquet"
        trace_gdf.to_parquet(trace_path)
        LOGGER.info("Saved spatial trace (%d rows) to %s", len(trace_gdf), trace_path)

    road_match_path = trace_dir / f"asset_road_match_{execution_id}.parquet"
    road_match_gdf = build_asset_road_match_table(
        gdf_assets_combined, config_combined["hazard_dir"],
        hazard_threshold=args.flood_threshold, hazard_column="EV0_ma",
        config=config_combined,
    )
    road_match_gdf.to_parquet(road_match_path)
    LOGGER.info("Saved static asset-to-road match table (%d rows) to %s", len(road_match_gdf), road_match_path)


def build_adaptation_policies(root_dir, args, l1_dir=None, l2_dir=None,
                               repair_priority="population impacts islands"):
    """Build the adaptation-measure policy set: base (no-adaptation) repair
    policies, plus L1-only, L2-only and all L1 x L2 combinations, read from
    pre-generated adaptation GeoJSON files.

    Args:
        l1_dir, l2_dir (Path, optional): L1/L2 adaptation GeoJSON directories.
            Default: args.l1_dir/args.l2_dir, falling back to the repo's
            data/test_samples/adaptation subdirs.
        repair_priority (str): repair_crew_assignment_method for the baseline
            and all L1/L2 policies, e.g. "population impacts islands" or
            "monetary impacts islands".
    """
    l1_dir = l1_dir or args.l1_dir or (root_dir / "data" / "test_samples" / "adaptation" / "L1_peilgeb_policies_separate")
    l2_dir = l2_dir or args.l2_dir or (root_dir / "data" / "test_samples" / "adaptation" / "L2_asset_barriers")

    l1_files = sorted(l1_dir.glob("*.geojson")) if l1_dir.exists() else []
    l2_files = sorted(l2_dir.glob("*.geojson")) if l2_dir.exists() else []
    if args.max_l1_files is not None:
        l1_files = l1_files[: args.max_l1_files]
    if args.max_l2_files is not None:
        l2_files = l2_files[: args.max_l2_files]
    LOGGER.info("Found %d L1 adaptation files in %s", len(l1_files), l1_dir)
    LOGGER.info("Found %d L2 adaptation files in %s", len(l2_files), l2_dir)

    active_timesteps = list(range(0, args.adaptation_active_timesteps))

    priority_label = repair_priority.replace(" ", "_")
    unconstrained_priority = repair_priority.replace(" islands", "")
    policies = [
        Policy(f"{priority_label}_policy", repair_crew_assignment_method=repair_priority),
        Policy(f"{unconstrained_priority.replace(' ', '_')}_unconstrained", repair_crew_assignment_method=unconstrained_priority),
    ]

    adaptation_policies = [
        Policy("baseline_no_prioritisation", repair_crew_assignment_method="islands"),
    ]

    for l1_file in l1_files:
        adaptation_policies.append(
            Policy(f"L1_only_{l1_file.stem}",
                   repair_crew_assignment_method=repair_priority,
                   l1_area_geojson=str(l1_file),
                   l1_active_timesteps=active_timesteps)
        )

    for l2_file in l2_files:
        adaptation_policies.append(
            Policy(f"L2_only_{l2_file.stem}",
                   repair_crew_assignment_method=repair_priority,
                   l2_asset_geojson=str(l2_file),
                   l2_active_timesteps=active_timesteps)
        )

    for l1_file, l2_file in itertools.product(l1_files, l2_files):
        adaptation_policies.append(
            Policy(f"L1L2_{l1_file.stem}_{l2_file.stem}",
                   repair_crew_assignment_method=repair_priority,
                   l1_area_geojson=str(l1_file),
                   l1_active_timesteps=active_timesteps,
                   l2_asset_geojson=str(l2_file),
                   l2_active_timesteps=active_timesteps)
        )

    policies.extend(adaptation_policies)

    LOGGER.info("Total policies to evaluate: %d", len(policies))
    LOGGER.info("  - Base repair policies: 2")
    LOGGER.info("  - Baseline (no adaptation): 1")
    LOGGER.info("  - L1-only policies: %d", len(l1_files))
    LOGGER.info("  - L2-only policies: %d", len(l2_files))
    LOGGER.info("  - L1+L2 combined: %d", len(l1_files) * len(l2_files))

    return policies


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    ema_logging.log_to_stderr(ema_logging.INFO)

    t_start = time.time()

    def _checkpoint(label):
        LOGGER.info("[warm-up] %s (+%.1fs)", label, time.time() - t_start)

    config = get_development_config() if args.dev_config else get_config()
    is_valid, missing_dirs, warnings = validate_config(config)
    if not is_valid:
        LOGGER.warning("Configuration issues detected: missing=%s warnings=%s", missing_dirs, warnings)
    setup_directories(config)
    print("\n" + "=" * 70)
    print("=== INPUT PARAMETERS ===")
    print("=" * 70)
    print(f"CLI arguments: {vars(args)}")
    print_config_summary(config)

    crew_ranges = resolve_crew_ranges(args)
    LOGGER.info("Per-asset-type repair-crew uncertainty ranges: %s", crew_ranges)

    fragility_steepness_ranges = resolve_fragility_steepness_ranges(args)
    LOGGER.info("Per-asset-type fragility-steepness uncertainty ranges: %s", fragility_steepness_ranges)

    root_dir = Path(config["root_dir"])

    gdf_assets_combined, substation_idx, osm_asset_idx = load_combined_assets(config)
    _checkpoint("combined assets loaded")

    LOGGER.info("Loading hazard maps from %s", config["hazard_dir"])
    hazard_maps = load_hazard_maps(config["hazard_dir"], max_days=None)
    LOGGER.info("Loaded %d hazard maps", len(hazard_maps))
    _checkpoint("hazard maps loaded")

    # Voronoi generators padded beyond the true extent shape boundary cells
    # correctly; results are clipped to model_extent_raw before any
    # downstream use (see CLAUDE.md "Dataset-extent alignment").
    model_extent_raw = get_model_extent(config)
    model_extent_buffered = get_model_extent(config, buffer_frac=0.05)

    LOGGER.info("Building Voronoi service areas for electricity substations...")
    substation_voronoi = create_voronoi_for_asset_type(
        gdf_assets_combined, "msls", boundary=model_extent_buffered,
    )
    substation_voronoi_clipped = gpd.clip(substation_voronoi, model_extent_raw)
    _checkpoint("substation Voronoi built")

    config_combined, service_area_map = build_dependency_config(
        config, gdf_assets_combined, substation_voronoi_clipped, osm_asset_idx,
    )
    _checkpoint("dependency config + service-area map built")

    asset_population_map, asset_to_lu, population_for_societal = prepare_population_and_land_use(
        config_combined, gdf_assets_combined, substation_idx,
        substation_voronoi_clipped, root_dir, model_extent_raw,
    )
    population_for_societal = clip_population_to_service_area(
        population_for_societal, gdf_assets_combined,
    )
    _checkpoint("population + land-use data ready")

    caches = load_simulation_caches(config["interim_dir"], config["hazard_dir"])
    LOGGER.info("Loaded caches: %s", [k for k in caches if caches[k] is not None])
    _checkpoint("simulation caches loaded")

    societal_access_config = get_societal_access_config(
        config_combined, population_for_societal, caches=caches,
        asset_types=["msls", *AMENITY_CATEGORY_LABELS.keys()],
        namespace="interdependent_ci_societal_access",
    )
    LOGGER.info("Societal access functions: %s", societal_access_config["all_functions"])

    execution_id = int(time.time())
    profiler = None
    if args.profile:
        from src.timing_profiler import SimulationTimingProfiler
        profiler = SimulationTimingProfiler(enabled=True)
        LOGGER.info("Profiling enabled: attaching a shared SimulationTimingProfiler.")

    model = build_ema_model(
        config_combined, gdf_assets_combined, hazard_maps, caches,
        asset_population_map, asset_to_lu, societal_access_config,
        execution_id, args, crew_ranges, fragility_steepness_ranges,
        profiler=profiler,
    )

    policies = build_adaptation_policies(
        root_dir, args, l1_dir=args.l1_dir, l2_dir=args.l2_dir,
        repair_priority=args.repair_priority,
    )

    if args.dry_run:
        LOGGER.info("Dry run requested: skipping EMA experiment execution.")
        LOGGER.info("Scenarios: %d, Policies: %d, Total runs: %d",
                    args.scenarios, len(policies), args.scenarios * len(policies))
        return

    LOGGER.info("Starting EMA experiment execution %d (%d scenarios x %d policies = %d runs)",
                execution_id, args.scenarios, len(policies), args.scenarios * len(policies))
    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)
        LOGGER.info("Seeded numpy's global RNG with %d for reproducible scenario sampling.", args.seed)

    _hazard_dir = config["hazard_dir"]

    def _flush_caches():
        flush_dirty_caches({
            ("hazard_extraction_cache", _hazard_dir): lambda: save_hazard_extraction_cache(
                caches.get("hazard_extraction_cache"), config["interim_dir"], _hazard_dir),
            ("island_cache", _hazard_dir): lambda: save_island_cache(
                caches.get("island_cache"), config["interim_dir"], _hazard_dir),
            ("overlap_cache", _hazard_dir): lambda: save_overlap_cache(
                caches.get("overlap_cache"), config["interim_dir"], _hazard_dir),
            ("societal_allocation_cache", _hazard_dir): lambda: save_societal_allocation_cache(
                caches.get("societal_allocation_cache"), config["interim_dir"], _hazard_dir),
        })

    try:
        if args.multiprocessing:
            print(f"\nWARNING: --multiprocessing {args.multiprocessing} requested. New cache "
                  "entries built by worker processes will NOT be persisted back to disk -- only "
                  "safe when caches are already warm for every road/hazard state this run touches.")
            with MultiprocessingEvaluator(model, n_processes=args.multiprocessing) as evaluator:
                experiments, outcomes = evaluator.perform_experiments(
                    scenarios=args.scenarios, policies=policies, uncertainty_sampling=Samplers.LHS,
                )
        else:
            with SequentialEvaluator(model) as evaluator:
                experiments, outcomes = evaluator.perform_experiments(
                    scenarios=args.scenarios, policies=policies, uncertainty_sampling=Samplers.LHS,
                )
    except KeyboardInterrupt:
        # Printed before the flush below (which can itself take a moment on
        # large caches), so Ctrl+C gives immediate feedback that the process
        # is shutting down deliberately rather than appearing to hang.
        print("\nInterrupted: terminating, please wait while caches are flushed...")
        raise
    finally:
        # Ensures caches accumulated so far are persisted even if the run is
        # interrupted (Ctrl+C) or raises partway through the experiments.
        _flush_caches()

    if profiler is not None:
        profiler.print_summary_tables()

    experiments_df = pd.DataFrame(experiments)
    LOGGER.info("Completed all experiments with execution id %d. Total runs: %d",
                execution_id, len(experiments_df))

    output_name = args.output_name or f"ema_results_{execution_id}_full_extent_adaptation"
    out_dir = config["output_dir"] / "ema_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name}.tar.gz"
    save_results((experiments, outcomes), out_path)
    LOGGER.info("Saved EMA results to %s", out_path)

    duration_seconds = time.time() - t_start
    output_parameters = {
        "execution_id": execution_id,
        "start_time_utc": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "end_time_utc": datetime.now(tz=timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "num_scenarios": args.scenarios,
        "num_policies": len(policies),
        "policy_names": [p.name for p in policies],
        "num_experiments": len(experiments_df),
        "societal_access_functions": societal_access_config["all_functions"],
        "results_archive_path": str(out_path),
        "seed": args.seed,
        "profiling_enabled": args.profile,
        "multiprocessing_n_processes": args.multiprocessing,
    }
    print("\n" + "=" * 70)
    print("=== OUTPUT PARAMETERS ===")
    print("=" * 70)
    for key, value in output_parameters.items():
        print(f"  {key}: {value}")

    input_parameters = {
        "cli_args": vars(args),
        "paths": {
            "root_dir": config["root_dir"],
            "electricity_dir": config["electricity_dir"],
            "osm_asset_dir": config.get("osm_asset_dir"),
            "hazard_dir": config["hazard_dir"],
            "interim_dir": config["interim_dir"],
            "output_dir": config["output_dir"],
        },
        "simulation_config": config["simulation_config"],
        "recovery_parameters": config["recovery_parameters"],
        "service_node_config": config.get("service_node_config", {}),
        "dependency_parameters": config_combined.get("dependency_parameters", {}),
        "analysis_config": config["analysis_config"],
        "crew_ranges": crew_ranges,
        "fragility_steepness_ranges": fragility_steepness_ranges,
    }
    manifest_path = write_run_manifest(out_dir, execution_id, input_parameters, output_parameters)
    LOGGER.info("Wrote run manifest (input + output parameters) to %s", manifest_path)

    if args.spatial_trace_policies:
        run_spatial_trace(
            config_combined, gdf_assets_combined, hazard_maps, caches,
            asset_population_map, asset_to_lu, societal_access_config,
            policies, args, crew_ranges, fragility_steepness_ranges,
            execution_id, out_dir,
        )


if __name__ == "__main__":
    main()
