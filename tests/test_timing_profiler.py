import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependency_evaluator import evaluate_dependencies_from_graph
from src.dependency_knowledge_graph import DependencyKnowledgeGraph
from src.timing_profiler import (
    NullProfiler,
    SimulationTimingProfiler,
    run_profiled_runs,
    summarize_profiled_runs,
)


def test_profiler_tracks_phase_counts_and_timestep_rows():
    profiler = SimulationTimingProfiler()

    with profiler.timestep(0):
        with profiler.section("phase.a"):
            _ = np.arange(1000).sum()
        with profiler.section("phase.a"):
            _ = np.arange(1000).sum()
        with profiler.section("phase.b"):
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
            with profiler.section("phase.run"):
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


def test_self_time_correct_for_nested_sections():
    """parent self + child inclusive == parent inclusive."""
    profiler = SimulationTimingProfiler()

    with profiler.timestep(0):
        with profiler.section("parent"):
            time.sleep(0.01)
            with profiler.section("child"):
                time.sleep(0.01)
            time.sleep(0.005)

    phase_df = profiler.phase_summary()
    parent_row = phase_df[phase_df["phase"] == "parent"].iloc[0]
    child_row = phase_df[phase_df["phase"] == "child"].iloc[0]

    assert float(child_row["self_seconds"]) == pytest.approx(
        float(child_row["total_seconds"]), abs=1e-6
    )
    assert float(parent_row["self_seconds"]) + float(child_row["total_seconds"]) == pytest.approx(
        float(parent_row["total_seconds"]), abs=1e-6
    )
    # Self times of all sections within a timestep must sum to <= timestep total.
    ts_df = profiler.timestep_summary()
    timestep_total = float(ts_df.iloc[0]["total_seconds"])
    self_time_sum = float(parent_row["self_seconds"]) + float(child_row["self_seconds"])
    assert self_time_sum <= timestep_total + 1e-6


def test_unattributed_equals_timestep_total_minus_section_self_times():
    profiler = SimulationTimingProfiler()

    with profiler.timestep(0, loop="simulation"):
        with profiler.section("phase.a"):
            time.sleep(0.01)
        time.sleep(0.005)  # deliberately unattributed work

    ts_df = profiler.timestep_summary()
    timestep_total = float(ts_df.iloc[0]["total_seconds"])
    unattributed = float(ts_df.iloc[0]["simulation.unattributed"])

    phase_df = profiler.phase_summary()
    section_self_sum = phase_df.loc[
        ~phase_df["phase"].isin(["simulation.total", "simulation.unattributed"]),
        "self_seconds",
    ].sum()

    assert unattributed == pytest.approx(timestep_total - section_self_sum, abs=1e-6)
    assert unattributed > 0.0

    # `unattributed` is a first-class phase row.
    assert "simulation.unattributed" in phase_df["phase"].values


def test_loop_labels_produce_separate_total_rows():
    profiler = SimulationTimingProfiler()

    with profiler.timestep(0, loop="simulation"):
        with profiler.section("simulation.step"):
            pass

    with profiler.timestep(0, loop="societal_postprocess"):
        with profiler.section("societal_access.compute_scalars"):
            pass

    phase_df = profiler.phase_summary()
    assert "simulation.total" in phase_df["phase"].values
    assert "societal_postprocess.total" in phase_df["phase"].values
    assert "simulation.unattributed" in phase_df["phase"].values
    assert "societal_postprocess.unattributed" in phase_df["phase"].values

    sim_row = phase_df[phase_df["phase"] == "simulation.total"].iloc[0]
    societal_row = phase_df[phase_df["phase"] == "societal_postprocess.total"].iloc[0]
    assert int(sim_row["call_count"]) == 1
    assert int(societal_row["call_count"]) == 1

    ts_df = profiler.timestep_summary()
    assert set(ts_df["loop"]) == {"simulation", "societal_postprocess"}


def test_disabled_profiler_records_nothing_and_does_not_alter_results():
    profiler = SimulationTimingProfiler(enabled=False)

    def _run(prof):
        with prof.timestep(0):
            with prof.section("phase.a"):
                pass
        return 42

    result = _run(profiler)
    assert result == 42
    assert profiler.phase_summary().empty
    assert profiler.timestep_summary().empty


def test_null_profiler_records_nothing_and_does_not_alter_results():
    profiler = NullProfiler()

    def _run(prof):
        with prof.timestep(0, loop="simulation"):
            with prof.section("phase.a"):
                pass
        return 42

    result = _run(profiler)
    assert result == 42
    # NullProfiler has no bookkeeping state at all.
    assert not hasattr(profiler, "_phase_timings")

    # NullProfiler returns the exact same reusable context on every call
    # (zero per-call allocation).
    assert profiler.section("a") is profiler.section("b")
    assert profiler.timestep(0) is profiler.section("a")


def test_negative_self_time_triggers_warning():
    profiler = SimulationTimingProfiler()

    # Directly exercise _record with a fabricated negative self-time, since
    # negative self-time can only arise from mismatched nesting.
    with pytest.warns(RuntimeWarning, match="Negative self time"):
        profiler._record("bogus.phase", elapsed_seconds=0.001, self_seconds=-0.001)
