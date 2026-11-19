"""EMA runner for single-map mode: independent flood maps, peak depth, no time stepping.

Reuses the multi-map setup (assets, population, caches, societal config, adaptation
policies, per-type crew and fragility-steepness uncertainties); only the model function
differs (``src.single_map.run_single_map_experiment``). ``seed`` and ``flood_threshold``
are constants. Outcomes are one value per map; if the maps are return periods of one flood
(names equal except ``RP<n>``), ``annualised_*`` outcomes are added.
"""
import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from ema_workbench import (
    ArrayOutcome,
    Constant,
    Model,
    MultiprocessingEvaluator,
    Samplers,
    SequentialEvaluator,
)
from ema_workbench.util.utilities import save_results

import interdependent_ci_societal_access as multi
from config import get_config, get_development_config, setup_directories
from src.caching import save_hazard_extraction_cache, save_island_cache, save_societal_allocation_cache
from src.graph_manager import build_single_map_graph
from src.simulation import (
    build_per_type_crew_uncertainties,
    build_per_type_fragility_steepness_uncertainties,
)
from src.single_map import MONETARY_CATEGORIES, return_periods, run_single_map_experiment
from src.societal_access import list_societal_metric_names

LOGGER = logging.getLogger(__name__)
COUNT_OUTCOMES = [
    "flooded_assets", "non_operational_assets", "max_recovery_time_h",
    *(f"monetary_impact_{c}" for c in [*MONETARY_CATEGORIES, "total"]),  # EUR/h at peak
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Constant seed of the common random numbers.")
    parser.add_argument("--flood-threshold", type=float, default=0.2)
    parser.add_argument("--no-living-streets", action="store_true",
                        help="Leave living streets out of the built road graph (included by default, with a warning).")
    parser.add_argument("--skip-fraction-flooded", action="store_true",
                        help="Do not compute the flooded fraction per edge (EV<i>_fr = 0.0); faster graph build.")
    parser.add_argument("--no-exempt-motorways", action="store_true",
                        help="Let flooding cut motorways (default: exempt from the flood threshold).")
    parser.add_argument("--no-exempt-bridges", action="store_true",
                        help="Let flooding cut bridges (default: exempt from the flood threshold).")
    parser.add_argument("--no-exempt-tunnels", action="store_true",
                        help="Let flooding cut tunnels (default: exempt from the flood threshold).")
    parser.add_argument("--hazard-dir", type=Path, default=None,
                        help="Directory with the hazard maps (default: the config's hazard_dir).")
    parser.add_argument("--graph-path", type=Path, default=None,
                        help="Prebuilt graph with EV<i>_ma columns in map order; default: built per extent and maps.")
    parser.add_argument("--repair-crews-range", nargs=3, metavar=("ASSET_TYPE", "MIN", "MAX"),
                        action="append", default=None)
    parser.add_argument("--fragility-steepness-range", nargs=3, metavar=("ASSET_TYPE", "MIN", "MAX"),
                        action="append", default=None)
    parser.add_argument("--l1-dir", type=Path, default=None)
    parser.add_argument("--l2-dir", type=Path, default=None)
    parser.add_argument("--max-l1-files", type=int, default=None)
    parser.add_argument("--max-l2-files", type=int, default=None)
    parser.add_argument("--repair-priority", type=str, default="population impacts islands")
    parser.add_argument("--multiprocessing", type=int, default=None, metavar="N_PROCESSES")
    parser.add_argument("--spatial-trace-policies", type=str, default=None, metavar="NAME1,NAME2",
                        help="Also write per-asset traces and island reports for these policies.")
    parser.add_argument("--spatial-trace-scenarios", type=int, default=1)
    parser.add_argument("--spatial-trace-fields", type=str, default=None,
                        help="Comma-separated trace fields; default as in the multi-map trace.")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dev-config", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    args.adaptation_active_timesteps = 1  # policies carry it, single-map has no timesteps
    return args


def build_model(inputs, config, graph_path, args, crew_ranges, steepness_ranges):
    model = Model("SingleMapSimulation", function=run_single_map_experiment)
    model.uncertainties = [
        *build_per_type_crew_uncertainties(crew_ranges),
        *build_per_type_fragility_steepness_uncertainties({
            t: (round(lo * 2), round(hi * 2)) for t, (lo, hi) in steepness_ranges.items()
        }),
    ]
    caches = inputs["caches"]
    societal = inputs["societal_access_config"]
    model.constants = [
        Constant("gdf_assets", inputs["gdf_assets"]),
        Constant("hazard_maps", inputs["hazard_maps"]),
        Constant("config", inputs["config_combined"]),
        Constant("graph_path", graph_path),
        Constant("societal_access_config", societal),
        Constant("flood_threshold", args.flood_threshold),
        Constant("seed", args.seed),
        Constant("asset_to_lu", inputs["asset_to_lu"]),
        Constant("asset_population_map", inputs["asset_population_map"]),
        Constant("island_cache", caches.get("island_cache")),
        Constant("hazard_extraction_cache", caches.get("hazard_extraction_cache")),
    ]
    names = [*COUNT_OUTCOMES, *list_societal_metric_names(
        all_functions=societal["all_functions"], pop_group_columns=societal["pop_group_columns"],
        reference_group=societal.get("reference_group", "total"),
    )]
    if return_periods([Path(m).stem for m in inputs["hazard_maps"]]) is not None:
        names += [f"annualised_{n}" for n in names]
    model.outcomes = [ArrayOutcome(n) for n in names]
    return model


def run_spatial_trace(model, inputs, graph_path, policies, args, execution_id, out_dir):
    """Per-asset peak-state traces and per-map island reports for the requested policies.

    Same flags, fields and file format as the multi-map trace (``timestep`` = map index, plus a
    ``map`` column); one file per (scenario, policy), plus road sections coloured by island and an
    island-size distribution for every map. Runs in-process on the draws' sampled uncertainties.
    """
    from ema_workbench.em_framework import sample_uncertainties

    from src import island_report
    from src.island_analysis import build_asset_road_match_table
    from src.single_map import road_peak_depths, single_map_trace_columns

    names = [n.strip() for n in args.spatial_trace_policies.split(",") if n.strip()]
    selected = [p for p in policies if p.name in names]
    if missing := sorted(set(names) - {p.name for p in selected}):
        LOGGER.warning("Unknown spatial-trace policies %s. Available: %s", missing, sorted(p.name for p in policies))
    fields = ([f.strip() for f in args.spatial_trace_fields.split(",") if f.strip()]
              if args.spatial_trace_fields else multi.DEFAULT_SPATIAL_TRACE_FIELDS)
    constants = {c.name: c.value for c in model.constants}
    config = constants["config"]
    gdf_assets = inputs["gdf_assets"]
    map_files = {Path(m).stem: m for m in inputs["hazard_maps"]}
    population = island_report.asset_population_array(inputs["asset_population_map"], len(gdf_assets))
    asset_geometry = gdf_assets[["geometry", "type"]].assign(asset_id=range(len(gdf_assets)))
    trace_dir = Path(out_dir) / "spatial_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    scenarios = sample_uncertainties(model, args.spatial_trace_scenarios, Samplers.LHS)
    for scenario in scenarios:
        for policy in selected:
            capture = []
            run_single_map_experiment(**{**constants, **dict(scenario.items()), **dict(policy.items()), "capture": capture})
            stem = f"spatial_trace_{execution_id}_{policy.name.replace(' ', '_')}_scenario{scenario.name}"
            island_report.write_trace_file(
                trace_dir / f"{stem}.parquet", single_map_trace_columns(capture, fields), asset_geometry,
                gdf_assets.crs, multi._json_safe(dict(scenario.items())),
            )
            for c in capture:
                island_report.write_island_report(
                    trace_dir, f"{stem}_{c['map']}", c["roads"], c["outcome"]["island_id"],
                    c["outcome"]["operational"], population, road_peak_depths(graph_path, c["column"]),
                    footprint=island_report.raster_footprint(map_files[c["map"]], c["roads"].crs),
                    asset_geometry=gdf_assets.geometry,
                )
            LOGGER.info("Saved spatial trace and island reports for %s to %s", stem, trace_dir)

    if selected:
        # Static baseline road match, as in multi-map; uses the first map's road state
        road_match = build_asset_road_match_table(
            gdf_assets, config["hazard_dir"], hazard_threshold=args.flood_threshold,
            hazard_column=capture[0]["column"], config=config,
                        graph_path=graph_path,
                    )
        road_match_path = trace_dir / f"asset_road_match_{execution_id}.parquet"
        road_match.to_parquet(road_match_path)
        LOGGER.info("Saved static asset-to-road match table (%d rows) to %s", len(road_match), road_match_path)


def scope_run_dirs(config):
    """Give a non-default hazard folder its own interim/output dirs.

    Cache files and dirs are named after the hazard folder's last component only, so same-named
    folders would share them. Appends `<parent>_<sha1(full path)[:8]>`; scope.txt records the source.
    """
    hazard_dir = Path(config["hazard_dir"]).resolve()
    suffix = f"{hazard_dir.parent.name}_{hashlib.sha1(str(hazard_dir).lower().encode()).hexdigest()[:8]}"
    for key in ("interim_dir", "output_dir"):
        config[key] = Path(f"{config[key]}_{suffix}")
        config[key].mkdir(parents=True, exist_ok=True)
        (config[key] / "scope.txt").write_text(f"{hazard_dir}\n", encoding="utf-8")
    return config


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    t_start = time.time()

    config = get_development_config() if args.dev_config else get_config(hazard_dir_override=args.hazard_dir)
    if args.hazard_dir and not args.dev_config:
        scope_run_dirs(config)
    for road_type in ("motorways", "bridges", "tunnels"):
        if getattr(args, f"no_exempt_{road_type}"):
            config["simulation_config"][f"exempt_{road_type}"] = False
    setup_directories(config)
    crew_ranges = multi.resolve_crew_ranges(args)
    steepness_ranges = multi.resolve_fragility_steepness_ranges(args)

    inputs = multi.prepare_model_inputs(config)
    graph_path = args.graph_path or build_single_map_graph(
        inputs["hazard_maps"], config["hazard_dir"].parent, include_living_streets=not args.no_living_streets,
        skip_fraction_flooded=args.skip_fraction_flooded,
    )
    LOGGER.info("Road graph: %s", graph_path)
    model = build_model(inputs, config, graph_path, args, crew_ranges, steepness_ranges)
    policies = multi.build_adaptation_policies(
        Path(config["root_dir"]), args, l1_dir=args.l1_dir, l2_dir=args.l2_dir,
        repair_priority=args.repair_priority,
    )
    LOGGER.info("%d scenarios x %d policies = %d runs, %d maps each",
                args.scenarios, len(policies), args.scenarios * len(policies), len(inputs["hazard_maps"]))
    if args.dry_run:
        return

    caches = inputs["caches"]
    try:
        if args.multiprocessing:
            print("WARNING: new cache entries built by worker processes are not persisted.")
            evaluator = MultiprocessingEvaluator(model, n_processes=args.multiprocessing)
        else:
            evaluator = SequentialEvaluator(model)
        with evaluator:
            experiments, outcomes = evaluator.perform_experiments(
                scenarios=args.scenarios, policies=policies, uncertainty_sampling=Samplers.LHS,
            )
    finally:
        hazard_dir, interim = config["hazard_dir"], config["interim_dir"]
        save_hazard_extraction_cache(caches.get("hazard_extraction_cache"), interim, hazard_dir)
        save_island_cache(caches.get("island_cache"), interim, hazard_dir)
        save_societal_allocation_cache(caches.get("societal_allocation_cache"), interim, hazard_dir)

    execution_id = int(t_start)
    out_dir = config["output_dir"] / "ema_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_name or f'single_map_results_{execution_id}'}.tar.gz"
    save_results((experiments, outcomes), out_path)
    multi.write_run_manifest(out_dir, execution_id, vars(args), {
        "start_time_utc": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "duration_seconds": time.time() - t_start,
        "num_experiments": len(pd.DataFrame(experiments)),
        "map_names": [Path(m).stem for m in inputs["hazard_maps"]],
        "policy_names": [p.name for p in policies],
        "results_archive_path": str(out_path),
    })
    LOGGER.info("Saved results to %s", out_path)
    if args.spatial_trace_policies:
        run_spatial_trace(model, inputs, graph_path, policies, args, execution_id, out_dir)


if __name__ == "__main__":
    main()
