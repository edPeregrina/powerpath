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
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from ema_workbench import (
    Constant,
    Model,
    MultiprocessingEvaluator,
    Samplers,
    SequentialEvaluator,
    TimeSeriesOutcome,
    ema_logging,
)
from ema_workbench.util.utilities import save_results

from config import (
    DEFAULT_CREW_RANGES,
    DEFAULT_FRAGILITY_STEEPNESS_RANGES,
    get_config,
    get_development_config,
    print_config_summary,
    setup_directories,
    validate_config,
)
from src.adaptation import build_adaptation_policies
from src.caching import (
    flush_dirty_caches,
    save_hazard_extraction_cache,
    save_island_cache,
    save_overlap_cache,
    save_societal_allocation_cache,
)
from src.data_loader import prepare_model_inputs
from src.impacts import MONETARY_CATEGORIES
from src.island_analysis import add_road_exemption_args, apply_road_exemption_args
from src.island_report import DEFAULT_SPATIAL_TRACE_FIELDS
from src.simulation import (
    build_per_type_crew_uncertainties,
    build_per_type_fragility_steepness_uncertainties,
    fragility_steepness_from_param,
    resolve_crew_ranges,
    resolve_fragility_steepness_ranges,
    simulate_with_per_type_crews,
)
from src.societal_access import list_societal_metric_names
from src.utils import json_safe, write_run_manifest

LOGGER = logging.getLogger(__name__)

# Aggregated 2D outcome variables (per-timestep, summed across assets).
KEEP_2D_VARS = ["flooded", "operational", "damage_ratio", "repair_time"]

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
            f"Valid asset types: {', '.join(DEFAULT_CREW_RANGES)}. "
            f"Defaults: {DEFAULT_CREW_RANGES}."
        ),
    )
    parser.add_argument("--fragility-steepness-range", nargs=3, metavar=("ASSET_TYPE", "MIN", "MAX"),
                         action="append", default=None,
                         help=(
                             "Override the fragility-steepness (k) uncertainty range for one "
                             "fragility-eligible asset type, sampled in 0.5 increments, e.g. "
                             "'--fragility-steepness-range msls 5.0 7.5'. Repeat for multiple "
                             f"asset types. Valid asset types: {', '.join(DEFAULT_FRAGILITY_STEEPNESS_RANGES)}. "
                             f"Defaults: {DEFAULT_FRAGILITY_STEEPNESS_RANGES}."
                         ))
    parser.add_argument("--flood-threshold", type=float, default=0.2,
                         help="Flood depth threshold (m) above which an asset is considered flooded. Default: 0.2.")
    add_road_exemption_args(parser)
    parser.add_argument("--adaptation-active-timesteps", type=int, default=198,
                         help="Number of leading timesteps during which L1/L2 adaptation measures are active. Default: 198.")
    parser.add_argument(
        "--l1-dir", type=Path, default=None,
        help="Directory with L1 (area-based) adaptation GeoJSON files. "
             "Default: <root_dir>/data/test_samples/adaptation/L1_adaptations",
    )
    parser.add_argument(
        "--l2-dir", type=Path, default=None,
        help="Directory with L2 (asset-based) adaptation GeoJSON files. "
             "Default: <root_dir>/data/test_samples/adaptation/L2_adaptations",
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
    parser.add_argument("--spatial-trace-multiprocessing", type=int, default=None, metavar="N_PROCESSES",
                         help="OFF by default -- the spatial-trace capture pass uses "
                              "SequentialEvaluator unless this is set. Only pass this once the "
                              "on-disk caches already cover every road/hazard state the trace run "
                              "will touch (e.g. a prior warm run over the same hazard maps/extent); "
                              "same correctness caveat as --multiprocessing (new cache entries built "
                              "by a worker are not written back). A large --spatial-trace-policies x "
                              "--spatial-trace-scenarios capture is the main case this is for.")
    parser.add_argument("--dev-config", action="store_true",
                         help="Use get_development_config() (data/test_samples) instead of the "
                              "full-extent get_config(). Intended for quick smoke tests/parity checks.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed numpy's global RNG before EMA sampling, for reproducible "
                              "scenario draws across runs (e.g. before/after comparisons).")
    parser.add_argument("--repair-priority", type=str, default="population impacts islands",
                         help="repair_crew_assignment_method used for the baseline and all L1/L2 "
                              "policies. Default: 'population impacts islands'.")
    parser.add_argument("--shadow-l1-islands", action="store_true",
                         help="Diagnostic: on each L1 island-cache miss also run the incremental "
                              "L1 partition and log SHADOW_L1 timings/equality (results unchanged).")
    return parser.parse_args()


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
            capacity_groups=societal_access_config.get("capacity_groups"),
            service_area_function_provider_types=societal_access_config.get("service_area_function_provider_types"),
        )
    )
    model.outcomes = outcomes_list

    return model


def write_multi_map_road_islands(trace_dir, stem, gdf_assets, config, island_cache, policy, timestep,
                                 major_timestep, threshold, repair_crew_assignment_method):
    """``<stem>_road_islands.parquet`` for one timestep of a trace, with the island labels the
    simulation used (read from its island cache). Returns the path, or None on a cache miss.

    Islands are only built at map loads, so the column, active L1 and cache key come from the
    load timestep. Non-island crew methods use flat connectivity, so nothing is written."""
    from src.island_analysis import _road_exemption_kwargs
    from src.island_report import write_road_islands
    from src.road_islands import compute_island_geodataframe_from_graph
    from src.simulation import _get_active_adaptation, island_state_key
    from src.single_map import road_peak_depths

    if "island" not in repair_crew_assignment_method:
        return None
    loaded_timestep = int(timestep // major_timestep) * major_timestep
    column = f"EV{int(timestep // major_timestep)}_ma"
    l1, l1_active = policy.get("l1_area_geojson"), policy.get("l1_active_timesteps")
    key = island_state_key(gdf_assets, column, threshold, l1, l1_active, loaded_timestep, config)
    entry = (island_cache or {}).get(key)
    if entry is None:
        LOGGER.warning("No island cache entry for %s at t=%s; skipping road islands.", policy.name, timestep)
        return None
    graph_path = Path(config["hazard_dir"]).parent / "static" / "output_graph" / "base_graph_hazard_editted.p"
    roads = compute_island_geodataframe_from_graph(
        graph_path, hazard_threshold=threshold, hazard_column=column, buffer_distance=20,
        l1_area_geojson=_get_active_adaptation(l1, l1_active, loaded_timestep), **_road_exemption_kwargs(config),
    )
    rfids_islands = entry["rfids_islands"]
    keep = [c for c in ("rfid", "island_id", "length_m", "island_size_km", "original_geometry", "geometry")
            if c in roads]
    roads = roads.loc[roads["rfid"].isin(rfids_islands), keep].copy()
    roads["island_id"] = roads["rfid"].map(rfids_islands)
    return write_road_islands(trace_dir, stem, roads, road_peak_depths(graph_path, column))


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

    from src import island_report
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

    # always trace island_id/operational, whatever --spatial-trace-fields says
    trace_fields = list(dict.fromkeys([*trace_fields, "island_id", "operational"]))

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

    if args.spatial_trace_multiprocessing:
        LOGGER.warning(
            "Spatial trace capture running with MultiprocessingEvaluator (%d processes) -- only "
            "safe if on-disk caches already cover every road/hazard state this run touches. "
            "New cache entries built by a worker are NOT written back to disk/main process.",
            args.spatial_trace_multiprocessing,
        )
        evaluator_cm = MultiprocessingEvaluator(trace_model, n_processes=args.spatial_trace_multiprocessing)
    else:
        evaluator_cm = SequentialEvaluator(trace_model)

    with evaluator_cm as evaluator:
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

        safe_policy = str(policy_name).replace(" ", "_")
        trace_path = trace_dir / f"spatial_trace_{execution_id}_{safe_policy}_scenario{scenario_name}.parquet"
        # Persist the sampled uncertainty values for this draw alongside the
        # trace -- otherwise this scenario's parameters (repair-crew counts,
        # fragility steepness, etc.) are only known in-memory here and get
        # silently discarded, leaving downstream consumers (e.g. the island
        # asset explorer dashboard) with only an arbitrary scenario index and
        # no way to say *what* draw they're looking at.
        uncertainty_cols = [c for c in experiments_df.columns if c not in ("scenario", "policy", "model")]
        island_report.write_trace_file(
            trace_path, rows, asset_geometry, gdf_assets_combined.crs,
            json_safe(experiments_df.loc[exp_idx, uncertainty_cols].to_dict()),
        )

        policy = next(p for p in trace_policies if p.name == policy_name)
        for t in timesteps:
            if not t % 24 == 0:
                continue
            write_multi_map_road_islands(
                trace_dir, f"{trace_path.stem}_day{int(t) // 24}", gdf_assets_combined, config_combined,
                caches.get("island_cache"), policy, int(t), args.major_timestep, args.flood_threshold,
                args.repair_priority,
            )

    road_match_path = trace_dir / f"asset_road_match_{execution_id}.parquet"
    road_match_gdf = build_asset_road_match_table(
        gdf_assets_combined, config_combined["hazard_dir"],
        hazard_threshold=args.flood_threshold, hazard_column="EV0_ma",
        config=config_combined,
        graph_path=config_combined["hazard_dir"].parent / "static" / "output_graph" / "base_graph_hazard_editted.p",
    )
    road_match_gdf.to_parquet(road_match_path)
    LOGGER.info("Saved static asset-to-road match table (%d rows) to %s", len(road_match_gdf), road_match_path)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    ema_logging.log_to_stderr(ema_logging.INFO)

    t_start = time.time()

    def _checkpoint(label):
        LOGGER.info("[warm-up] %s (+%.1fs)", label, time.time() - t_start)

    config = get_development_config() if args.dev_config else get_config()
    config['simulation_config']['shadow_l1_islands'] = args.shadow_l1_islands
    apply_road_exemption_args(config, args)
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

    inputs = prepare_model_inputs(config, _checkpoint)
    config_combined = inputs["config_combined"]
    gdf_assets_combined = inputs["gdf_assets"]
    hazard_maps = inputs["hazard_maps"]
    asset_population_map = inputs["asset_population_map"]
    asset_to_lu = inputs["asset_to_lu"]
    caches = inputs["caches"]
    societal_access_config = inputs["societal_access_config"]

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
        "spatial_trace_policies": args.spatial_trace_policies,
        "spatial_trace_multiprocessing_n_processes": args.spatial_trace_multiprocessing,
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
