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

WARNING: a full run (default: 100 scenarios x ~150 policies) can take several
hours. Use ``--scenarios``/``--max-days``/``--dry-run`` for a quick smoke test
before committing to a full run.

Usage:
    python book/interdependent_ci_societal_access.py --dry-run
    python book/interdependent_ci_societal_access.py --scenarios 100
"""
import argparse
import itertools
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd
from ema_workbench import (
    Constant,
    Model,
    Policy,
    RealParameter,
    Samplers,
    SequentialEvaluator,
    TimeSeriesOutcome,
    ema_logging,
)
from ema_workbench.util.utilities import save_results

from config import get_config, get_societal_access_config, print_config_summary, setup_directories, validate_config
from src.caching import load_simulation_caches
from src.data_loader import load_electricity_assets, load_hazard_maps, load_osm_assets
from src.dependency_knowledge_graph import build_default_knowledge_graph
from src.impacts import (
    create_voronoi_for_asset_type,
    prepare_land_use_impact_data,
    prepare_population_impact_data,
    update_voll_rates,
)
from src.simulation import build_per_type_crew_uncertainties, simulate_with_per_type_crews
from src.societal_access import list_societal_metric_names
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
    parser.add_argument("--fragility-k-min", type=float, default=5.0,
                         help="Lower bound for the 'fragility_param_k' uncertainty. Default: 5.0.")
    parser.add_argument("--fragility-k-max", type=float, default=7.5,
                         help="Upper bound for the 'fragility_param_k' uncertainty. Default: 7.5.")
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
    parser.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity. Default: INFO.")
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
                                     substation_voronoi, root_dir):
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
    """
    study_area_path = root_dir / "data" / "utilities" / "stedin_area.geojson"
    study_area = gpd.read_file(study_area_path, driver="GeoJSON").to_crs("EPSG:28992")
    population_buffer = study_area.buffer(1000).set_crs("EPSG:28992")

    population_data_path = root_dir / "data" / "population" / "cbs_vk100_2024.gpkg"
    LOGGER.info("Loading population data from %s", population_data_path)
    population_data = gpd.read_file(
        population_data_path, driver="GPKG", bbox=tuple(substation_voronoi.total_bounds),
    ).to_crs("EPSG:28992")

    population_data = population_data[
        (population_data["aantal_inwoners"] > 0) & (~population_data.geometry.isna())
    ].dropna(subset=["geometry"])
    population_study_area = gpd.clip(population_data, population_buffer)
    population_above_0 = population_study_area[["aantal_inwoners", "geometry"]].set_crs("EPSG:28992")

    # Population grid for societal-access postprocessing (demographic group
    # columns + a stable cell_id), independent of asset_population_map.
    pop_group_cols = list(config["service_node_config"]["population_groups"].values())
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
    land_use_path = root_dir / "data" / "land_use" / "CBS_Publicatiebestand_BBG2017_v1.gpkg"
    asset_to_lu_cache_path = config["interim_dir"] / f"{land_use_path.stem}_asset_to_lu.pkl"

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
                     execution_id, args, crew_ranges):
    """Construct the EMA workbench Model: uncertainties, constants and outcomes."""
    model = Model("ElectricitySocietalAccessSimulation",
                   function=simulate_with_per_type_crews)

    model.uncertainties = [
        *build_per_type_crew_uncertainties(crew_ranges),
        RealParameter("fragility_param_k", args.fragility_k_min, args.fragility_k_max),
    ]

    model.constants = [
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


def build_adaptation_policies(root_dir, args):
    """Build the adaptation-measure policy set: base (no-adaptation) repair
    policies, plus L1-only, L2-only and all L1 x L2 combinations, read from
    pre-generated adaptation GeoJSON files.
    """
    l1_dir = args.l1_dir or (root_dir / "data" / "test_samples" / "adaptation" / "L1_peilgeb_policies_separate")
    l2_dir = args.l2_dir or (root_dir / "data" / "test_samples" / "adaptation" / "L2_asset_barriers")

    l1_files = sorted(l1_dir.glob("*.geojson")) if l1_dir.exists() else []
    l2_files = sorted(l2_dir.glob("*.geojson")) if l2_dir.exists() else []
    if args.max_l1_files is not None:
        l1_files = l1_files[: args.max_l1_files]
    if args.max_l2_files is not None:
        l2_files = l2_files[: args.max_l2_files]
    LOGGER.info("Found %d L1 adaptation files in %s", len(l1_files), l1_dir)
    LOGGER.info("Found %d L2 adaptation files in %s", len(l2_files), l2_dir)

    active_timesteps = list(range(0, args.adaptation_active_timesteps))

    policies = [
        Policy("monetary_impacts_policy", repair_crew_assignment_method="monetary impacts islands"),
        Policy("monetary_impacts_unconstrained", repair_crew_assignment_method="monetary impacts"),
    ]

    adaptation_policies = [
        Policy("baseline_no_prioritisation", repair_crew_assignment_method="islands"),
    ]

    for l1_file in l1_files:
        adaptation_policies.append(
            Policy(f"L1_only_{l1_file.stem}",
                   repair_crew_assignment_method="monetary impacts islands",
                   l1_area_geojson=str(l1_file),
                   l1_active_timesteps=active_timesteps)
        )

    for l2_file in l2_files:
        adaptation_policies.append(
            Policy(f"L2_only_{l2_file.stem}",
                   repair_crew_assignment_method="monetary impacts islands",
                   l2_asset_geojson=str(l2_file),
                   l2_active_timesteps=active_timesteps)
        )

    for l1_file, l2_file in itertools.product(l1_files, l2_files):
        adaptation_policies.append(
            Policy(f"L1L2_{l1_file.stem}_{l2_file.stem}",
                   repair_crew_assignment_method="monetary impacts islands",
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

    config = get_config()
    is_valid, missing_dirs, warnings = validate_config(config)
    if not is_valid:
        LOGGER.warning("Configuration issues detected: missing=%s warnings=%s", missing_dirs, warnings)
    setup_directories(config)
    print_config_summary(config)

    crew_ranges = resolve_crew_ranges(args)
    LOGGER.info("Per-asset-type repair-crew uncertainty ranges: %s", crew_ranges)

    root_dir = Path(config["root_dir"])

    gdf_assets_combined, substation_idx, osm_asset_idx = load_combined_assets(config)
    _checkpoint("combined assets loaded")

    LOGGER.info("Loading hazard maps from %s", config["hazard_dir"])
    hazard_maps = load_hazard_maps(config["hazard_dir"], max_days=None)
    LOGGER.info("Loaded %d hazard maps", len(hazard_maps))
    _checkpoint("hazard maps loaded")

    LOGGER.info("Building Voronoi service areas for electricity substations...")
    substation_voronoi = create_voronoi_for_asset_type(
        gdf_assets_combined, "msls",
        boundary=gpd.read_file(root_dir / "data" / "utilities" / "stedin_area.geojson",
                                driver="GeoJSON").to_crs("EPSG:28992").geometry.unary_union,
    )
    _checkpoint("substation Voronoi built")

    config_combined, service_area_map = build_dependency_config(
        config, gdf_assets_combined, substation_voronoi, osm_asset_idx,
    )
    _checkpoint("dependency config + service-area map built")

    asset_population_map, asset_to_lu, population_for_societal = prepare_population_and_land_use(
        config_combined, gdf_assets_combined, substation_idx,
        substation_voronoi, root_dir,
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
    model = build_ema_model(
        config_combined, gdf_assets_combined, hazard_maps, caches,
        asset_population_map, asset_to_lu, societal_access_config,
        execution_id, args, crew_ranges,
    )

    policies = build_adaptation_policies(root_dir, args)

    if args.dry_run:
        LOGGER.info("Dry run requested: skipping EMA experiment execution.")
        LOGGER.info("Scenarios: %d, Policies: %d, Total runs: %d",
                    args.scenarios, len(policies), args.scenarios * len(policies))
        return

    LOGGER.info("Starting EMA experiment execution %d (%d scenarios x %d policies = %d runs)",
                execution_id, args.scenarios, len(policies), args.scenarios * len(policies))
    with SequentialEvaluator(model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=args.scenarios, policies=policies, uncertainty_sampling=Samplers.LHS,
        )

    experiments_df = pd.DataFrame(experiments)
    LOGGER.info("Completed all experiments with execution id %d. Total runs: %d",
                execution_id, len(experiments_df))

    output_name = args.output_name or f"ema_results_{execution_id}_full_extent_adaptation"
    out_dir = config["output_dir"] / "ema_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_name}.tar.gz"
    save_results((experiments, outcomes), out_path)
    LOGGER.info("Saved EMA results to %s", out_path)


if __name__ == "__main__":
    main()
