import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependency_evaluator import evaluate_dependencies_from_graph
from src.dependency_knowledge_graph import DependencyKnowledgeGraph
from src.timing_profiler import (
    SimulationTimingProfiler,
    run_profiled_runs,
    summarize_profiled_runs,
)


def test_profiler_tracks_phase_counts_and_timestep_rows():
    profiler = SimulationTimingProfiler()

    with profiler.timestep(0):
        with profiler.section("phase.a", include_in_timestep=True):
            _ = np.arange(1000).sum()
        with profiler.section("phase.a", include_in_timestep=True):
            _ = np.arange(1000).sum()
        with profiler.section("phase.b", include_in_timestep=True):
            _ = np.arange(500).sum()

    phase_df = profiler.phase_summary(sort_desc=False)
    row_a = phase_df[phase_df["phase"] == "phase.a"].iloc[0]
    assert int(row_a["call_count"]) == 2
    assert float(row_a["total_seconds"]) >= 0.0
    assert float(row_a["avg_seconds"]) >= 0.0

    ts_df = profiler.timestep_summary()
    assert len(ts_df) == 1
    assert int(ts_df.iloc[0]["timestep"]) == 0
    assert float(ts_df.iloc[0]["total_seconds"]) >= 0.0


def test_run_profiled_runs_and_summary_with_warmup():
    def _run(profiler):
        with profiler.timestep(0):
            with profiler.section("phase.run", include_in_timestep=True):
                _ = np.arange(10).sum()
        return {"ok": True}

    runs = run_profiled_runs(_run, runs=2, warmup_runs=1)
    assert len(runs) == 2
    assert all(run["output"]["ok"] for run in runs)

    summary = summarize_profiled_runs(runs)
    phase_row = summary[summary["phase"] == "phase.run"].iloc[0]
    assert int(phase_row["run_count"]) == 2
    assert float(phase_row["call_count_mean"]) >= 1.0


def test_graph_dependency_outputs_unchanged_with_profiler():
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "hazard_type": "flooding",
                "asset_type_a": "msls",
                "asset_type_b": None,
                "relationship": "direct",
                "parameters": {
                    "hazard_blocks_operation": True,
                    "return_to_operational": {"trigger": "repair_complete"},
                },
            }
        ]
    )
    operational = np.array([True], dtype=bool)
    asset_type = np.array(["msls"])
    flooded_mask = np.array([True], dtype=bool)
    repair_time = np.array([2.0], dtype=float)

    baseline = evaluate_dependencies_from_graph(
        operational,
        asset_type,
        "flooding",
        kg,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
    )
    profiler = SimulationTimingProfiler()
    instrumented = evaluate_dependencies_from_graph(
        operational,
        asset_type,
        "flooding",
        kg,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
        profiler=profiler,
    )

    assert baseline.tolist() == instrumented.tolist()
    assert "dependency.restore_pass" in profiler.phase_summary()["phase"].values
