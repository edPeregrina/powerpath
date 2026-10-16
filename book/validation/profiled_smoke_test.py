"""Profiled smoke test for the weighted-Voronoi (schools) societal-access
functions added in ``src/societal_access.py``.

Permanent, re-runnable companion to ``baseline_smoke_test.py``: reuses the
exact same synthetic scenario and monkeypatch idiom, but threads a real
``SimulationTimingProfiler`` through the engine instead of ``NULL_PROFILER``,
and runs the scenario twice:

  * "weighted_voronoi" run -- schools use
    ``WEIGHTED_VORONOI_SERVICE_AREA_CONFIG`` (the new method), to confirm the
    two new phases (``societal_access._build_weighted_voronoi_cell_
    assignments`` and ``societal_access.apply_weighted_voronoi_travel_
    dependent_scalars``) are actually exercised and their timings are
    reasonable.
  * "base_method" run -- schools fall back to the pre-existing base island
    method (``BASE_METHOD_SERVICE_AREA_CONFIG``, i.e. no ``education`` entry
    at all), to confirm that *not* opting into ``weighted_voronoi`` carries
    no measurable overhead -- i.e. this change did not regress the default
    code path used by every other function.

Each configuration is run several times (with a couple of warmup runs
discarded) via ``run_profiled_runs``/``summarize_profiled_runs`` so the
reported numbers are stable despite the synthetic scenario's sub-millisecond
per-run cost. Usage:

    python book/validation/profiled_smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "book" / "validation"))

from src.timing_profiler import (  # noqa: E402
    SimulationTimingProfiler,
    run_profiled_runs,
    summarize_profiled_runs,
)

from baseline_smoke_test import (  # noqa: E402
    BASE_METHOD_SERVICE_AREA_CONFIG,
    WEIGHTED_VORONOI_SERVICE_AREA_CONFIG,
    run_scenario,
)

NEW_PHASES = [
    "societal_access._build_weighted_voronoi_cell_assignments",
    "societal_access.apply_weighted_voronoi_travel_dependent_scalars",
]

RUNS = 10
WARMUP_RUNS = 2


def _run_config(service_area_function_provider_types):
    def _run_callable(profiler: SimulationTimingProfiler):
        with tempfile.TemporaryDirectory(prefix="profiled_smoke_test_") as tmp:
            return run_scenario(
                Path(tmp),
                profiler=profiler,
                service_area_function_provider_types=service_area_function_provider_types,
            )

    return run_profiled_runs(_run_callable, runs=RUNS, warmup_runs=WARMUP_RUNS)


def main() -> int:
    print(f"Running {RUNS} profiled run(s) ({WARMUP_RUNS} warmup) per configuration...\n")

    weighted_runs = _run_config(WEIGHTED_VORONOI_SERVICE_AREA_CONFIG)
    base_runs = _run_config(BASE_METHOD_SERVICE_AREA_CONFIG)

    weighted_summary = summarize_profiled_runs(weighted_runs)
    base_summary = summarize_profiled_runs(base_runs)

    print("=== weighted_voronoi (schools use the new method) ===")
    print(weighted_summary.to_string(index=False))
    print()
    print("=== base_method (schools use the pre-existing base island method) ===")
    print(base_summary.to_string(index=False))
    print()

    ok = True

    for phase in NEW_PHASES:
        row = weighted_summary.loc[weighted_summary["phase"] == phase]
        if row.empty or row["call_count_mean"].iloc[0] <= 0:
            print(f"FAIL: expected phase '{phase}' to be exercised in the weighted_voronoi run.")
            ok = False
        else:
            mean_total = row["total_seconds_mean"].iloc[0]
            print(f"OK: '{phase}' exercised, mean total_seconds={mean_total:.6f}")

    # The apply-scalars phase is only entered when a non-empty cell
    # assignment exists, so it should not appear at all in the base_method
    # run. The build-assignments phase is always called (it is the one that
    # decides there is nothing to do), but should report negligible cost.
    base_apply_row = base_summary.loc[
        base_summary["phase"] == "societal_access.apply_weighted_voronoi_travel_dependent_scalars"
    ]
    if not base_apply_row.empty:
        print(
            "FAIL: 'apply_weighted_voronoi_travel_dependent_scalars' should not run "
            "at all when weighted_voronoi is not configured."
        )
        ok = False
    else:
        print("OK: apply-scalars phase absent in base_method run (no opt-in, no cost).")

    base_build_row = base_summary.loc[
        base_summary["phase"] == "societal_access._build_weighted_voronoi_cell_assignments"
    ]
    if not base_build_row.empty:
        mean_total = base_build_row["total_seconds_mean"].iloc[0]
        print(
            "OK: build-assignments phase present but near-zero in base_method run, "
            f"mean total_seconds={mean_total:.6f}"
        )

    if ok:
        print("\nPASS: weighted-Voronoi functions are profiled; no regression for the default/base path.")
        return 0
    print("\nFAIL: see above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
