"""Baseline parity smoke test for the schools-weighted-voronoi societal-access
work. Permanent, re-runnable gate for "did this change break existing
behaviour" in future societal-access work, not just this branch.

Runs the REAL simulation engine (``src.simulation.simulate_asset_damage_
recovery_access_breakdown``) against a small, self-contained synthetic
scenario, using the same monkeypatch-the-GIS-boundary idiom as ``tests/
test_societal_access.py`` (mock raster hazard extraction and road-network
island computation; everything else -- damage/repair/dependency/societal-
access logic -- is real).

Scenario
--------
5 assets, all co-located on a single flat-connectivity island (no road
severance modelled here -- that is already covered by ``tests/test_
societal_access_weighted_voronoi.py``'s island-gating tests; this script's
job is end-to-end wiring/parity, not re-proving per-method correctness):

- 1 ``msls`` substation -> "electricity" via ``simple_voronoi`` (the
  pre-existing exception).
- 1 ``hospital`` -> base island/shared-reachability method (unchanged by
  this branch).
- 2 spatial ``school`` sites (150 and 900 pupils -- realistic Dutch primary/
  secondary school sizes, NOT a placeholder like ``1e8``) at x=0 and x=100
  (EPSG:28992 metres) plus 1 non-spatial ``school_remote`` entry (200
  pupils, ``exclude_column=True``) -> "education" via ``weighted_voronoi``,
  exercising the remote-fraction blend (Step 5) through the real engine for
  the first time (prior tests exercised it only via direct
  ``postprocess_societal_access_results``/``_build_weighted_voronoi_cell_
  assignments`` calls).

3 synthetic hazard timesteps (hazard VALUE returned by the mocked raster
extraction, uniform across all assets per timestep -- NOT realistic flood
physics, that is out of scope here):

  t0: 0.0 (below the 0.2 flood threshold) -> everything operational.
  t1: 0.5 (above threshold)               -> everything blocked.
  t2: 0.0 again                           -> immediate recovery (the
      default knowledge graph's ``TRIGGER_IMMEDIATE`` return-to-operational
      trigger means no repair-time modelling is needed for this to be
      deterministic).

Usage
-----
    python book/validation/baseline_smoke_test.py --capture   # (re)write baseline
    python book/validation/baseline_smoke_test.py --verify    # diff vs baseline (default)
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_config  # noqa: E402
from src.dependency_knowledge_graph import TRIGGER_IMMEDIATE, DependencyKnowledgeGraph  # noqa: E402
import src.island_analysis as island_analysis  # noqa: E402
import src.simulation as simulation_module  # noqa: E402
from src.simulation import simulate_asset_damage_recovery_access_breakdown  # noqa: E402

BASELINE_PATH = Path(__file__).with_name("baseline_smoke_test_results.json")
TOLERANCE = 1e-6

# Realistic Dutch primary/secondary school sizes -- NOT 1e8-style placeholders.
SCHOOL_A_PUPILS = 150
SCHOOL_B_PUPILS = 900
SCHOOL_REMOTE_PUPILS = 200
WEIGHT_SCALE_FACTOR = 1000.0  # same order of magnitude as the tuned geometry
# used throughout tests/test_societal_access_weighted_voronoi*.py.

# Uniform hazard value applied to every asset at each timestep (see module
# docstring) -- index 0/1/2 == timestep 0/1/2.
HAZARD_VALUE_BY_TIMESTEP = [0.0, 0.5, 0.0]
FLOOD_THRESHOLD = 0.2

METRICS_TO_CAPTURE = (
    "societal_access_pct__hospital__total",
    "societal_access_pct__electricity__total",
    "societal_access_pct__education__total",
)


def _build_knowledge_graph():
    """Minimal graph: each of msls/hospital/school is directly blocked by
    flooding (no dependency edges) -- the harness's job is to exercise the
    societal-access method layer, not re-prove dependency-topology wiring
    (already covered by tests/test_dependency_knowledge_graph_schema.py)."""
    rules = [
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": source_type,
            "hazard_blocks_operation": True,
            "return_to_operational": {"trigger": TRIGGER_IMMEDIATE},
        }
        for source_type in ("msls", "hospital", "school")
    ]
    return DependencyKnowledgeGraph.from_config(rules)


def _build_assets():
    return gpd.GeoDataFrame(
        {
            "type": ["msls", "hospital", "school", "school", "school"],
            "pupils": [np.nan, np.nan, SCHOOL_A_PUPILS, SCHOOL_B_PUPILS, SCHOOL_REMOTE_PUPILS],
            "is_remote": [False, False, False, False, True],
        },
        geometry=[Point(45, 0), Point(45, 0), Point(0, 0), Point(100, 0), Point(5000, 5000)],
        crs="EPSG:28992",
    )


def _build_pop_grid():
    return gpd.GeoDataFrame(
        {
            "cell_id": ["straddling_cell"],
            "aantal_inwoners": [300],
            "aantal_inwoners_65_jaar_en_ouder": [50],
            "aantal_inwoners_0_tot_15_jaar": [45],
            "aantal_inwoners_25_tot_45_jaar": [90],
        },
        geometry=[box(44, -1, 46, 1)],
        crs="EPSG:28992",
    )


WEIGHTED_VORONOI_SERVICE_AREA_CONFIG = {
    "electricity": {"provider_types": {"msls"}, "method": "simple_voronoi"},
    "education": {
        "provider_types": {"school"},
        "method": "weighted_voronoi",
        "weight_column": "pupils",
        "weight_scale_factor": WEIGHT_SCALE_FACTOR,
        "exclude_column": "is_remote",
    },
}
#: Pre-existing default behaviour (what every function except electricity
#: used before this branch, and what still applies today since schools are
#: not wired into ``weighted_voronoi`` by ``config.get_societal_access_
#: config`` outside of an explicit override like the one above): "school"
#: falls back to the base island method, same as hospital.
BASE_METHOD_SERVICE_AREA_CONFIG = {
    "electricity": {"provider_types": {"msls"}, "method": "simple_voronoi"},
}


def run_scenario(root_dir, profiler=None, service_area_function_provider_types=None):
    """Run the synthetic 3-timestep scenario through the real simulation
    engine (GIS boundary mocked, everything else real). Returns a list of 3
    dicts, one per timestep, with the metrics in ``METRICS_TO_CAPTURE``.

    ``profiler`` (optional ``SimulationTimingProfiler``) and
    ``service_area_function_provider_types`` (optional override, defaults to
    ``WEIGHTED_VORONOI_SERVICE_AREA_CONFIG``) are only used by
    ``book/validation/profiled_smoke_test.py`` to compare timing with/without
    the ``weighted_voronoi`` method engaged -- the baseline parity
    capture/verify flow below always uses the defaults."""
    hazard_dir = root_dir / "hazard_case"
    hazard_dir.mkdir(parents=True, exist_ok=True)

    config = get_config(root_dir=root_dir, hazard_dir_override=hazard_dir)
    config["simulation_config"]["verbose"] = False
    config["simulation_config"]["accessibility_model"] = None
    config["dependency_parameters"]["knowledge_graph"] = _build_knowledge_graph().to_config()

    gdf_assets = _build_assets()
    pop = _build_pop_grid()
    num_assets = len(gdf_assets)

    societal_access_config = {
        "pop_grid_gdf": pop,
        "cell_id_column": "cell_id",
        "taxonomy": {"hospital": "hospital", "msls": "electricity", "school": "education"},
        "all_functions": ["hospital", "electricity", "education"],
        "allocation_cache": {},
        "service_area_function_provider_types": (
            service_area_function_provider_types
            if service_area_function_provider_types is not None
            else WEIGHTED_VORONOI_SERVICE_AREA_CONFIG
        ),
        "fail_on_missing_allocation": True,
    }
    hazard_maps = [hazard_dir / f"hazard_{t}.tif" for t in range(len(HAZARD_VALUE_BY_TIMESTEP))]

    def _fake_hazard_extraction(hazard_map, temp_gdf, map_counter, **kwargs):
        value = HAZARD_VALUE_BY_TIMESTEP[map_counter]
        return temp_gdf.assign(**{f"EV{map_counter}_ma": value})

    def _fake_match_assets_access(*args, **kwargs):
        # Single flat-connectivity road segment (rfid 10) shared by every
        # asset -- no road severance modelled (see module docstring).
        return np.full(num_assets, 10), [], [], {10: 100.0}

    def _fake_island_geodataframe(*args, **kwargs):
        return gpd.GeoDataFrame(
            {"rfid": [10], "island_id": [1], "length_m": [100.0]},
            geometry=[LineString([(0, 0), (100, 0)])],
            crs="EPSG:28992",
        )

    with patch.object(
        simulation_module, "find_hazard_value_at_points_optimized", _fake_hazard_extraction
    ), patch.object(
        simulation_module, "match_assets_access", _fake_match_assets_access
    ), patch.object(
        island_analysis, "compute_island_geodataframe_from_graph", _fake_island_geodataframe
    ):
        all_results, _final_state, _caches = simulate_asset_damage_recovery_access_breakdown(
            gdf_assets=gdf_assets,
            hazard_maps=hazard_maps,
            number_repair_crews=1,
            repair_crew_assignment_method="island",
            flood_threshold=FLOOD_THRESHOLD,
            root_dir=root_dir,
            config=config,
            major_timestep=1,
            timestep_output=True,
            societal_access_config=societal_access_config,
            verbose=False,
            **({"profiler": profiler} if profiler is not None else {}),
        )

    summary_rows = all_results[0][1]
    return [
        {metric: row[metric] for metric in METRICS_TO_CAPTURE}
        for row in summary_rows
    ]


def _run_in_tmp_dir():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="baseline_smoke_test_") as tmp:
        return run_scenario(Path(tmp))


def capture(results):
    BASELINE_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Captured baseline with {len(results)} timestep(s) to {BASELINE_PATH}")


def verify(results):
    if not BASELINE_PATH.exists():
        print(
            f"No baseline found at {BASELINE_PATH}. Run with --capture first.",
            file=sys.stderr,
        )
        return False
    baseline = json.loads(BASELINE_PATH.read_text())
    if len(baseline) != len(results):
        print(
            f"FAIL: baseline has {len(baseline)} timestep(s), current run has "
            f"{len(results)}.",
            file=sys.stderr,
        )
        return False

    ok = True
    for t, (expected, actual) in enumerate(zip(baseline, results)):
        for metric in METRICS_TO_CAPTURE:
            exp_val = expected[metric]
            act_val = actual[metric]
            if abs(exp_val - act_val) > TOLERANCE:
                ok = False
                print(
                    f"FAIL t={t} {metric}: baseline={exp_val!r} current={act_val!r}",
                    file=sys.stderr,
                )
    if ok:
        print(f"PASS: current run matches baseline ({len(results)} timestep(s)).")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--capture", action="store_true", help="(Re)write the stored baseline."
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Diff the current run against the stored baseline (default).",
    )
    args = parser.parse_args()

    results = _run_in_tmp_dir()

    if args.capture:
        capture(results)
        return 0
    ok = verify(results)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
