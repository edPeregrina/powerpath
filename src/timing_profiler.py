"""Lightweight timing utilities for simulation profiling.

The profiler is intentionally simple and notebook-friendly:
- phase-level counters/totals/averages
- optional per-timestep timing aggregation
- optional repeated-run execution with warmup
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import pandas as pd


@dataclass
class PhaseTiming:
    """Aggregated timing for a single named phase."""

    call_count: int = 0
    total_seconds: float = 0.0

    @property
    def avg_seconds(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_seconds / self.call_count


class SimulationTimingProfiler:
    """Collect named wall-clock timings with optional per-timestep detail."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._phase_timings: dict[str, PhaseTiming] = {}
        self._timestep_records: list[dict[str, Any]] = []
        self._active_timestep: dict[str, Any] | None = None
        self._active_starts: dict[str, float] = {}

    def _record(self, phase_name: str, elapsed_seconds: float) -> None:
        timing = self._phase_timings.setdefault(phase_name, PhaseTiming())
        timing.call_count += 1
        timing.total_seconds += elapsed_seconds

    def start_timer(self, phase_name: str) -> None:
        """Start a named timer to be stopped with :meth:`stop_timer`."""
        if not self.enabled:
            return
        self._active_starts[phase_name] = perf_counter()

    def stop_timer(self, phase_name: str, *, include_in_timestep: bool = False) -> None:
        """Stop a named timer and record elapsed time."""
        if not self.enabled:
            return
        start = self._active_starts.pop(phase_name, None)
        if start is None:
            raise KeyError(f"Timer '{phase_name}' was not started.")
        elapsed = perf_counter() - start
        self._record(phase_name, elapsed)
        if include_in_timestep and self._active_timestep is not None:
            self._active_timestep["phase_seconds"][phase_name] += elapsed

    @contextmanager
    def section(self, phase_name: str, *, include_in_timestep: bool = False):
        """Context manager for a named timed section."""
        if not self.enabled:
            yield
            return

        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self._record(phase_name, elapsed)
            if include_in_timestep and self._active_timestep is not None:
                self._active_timestep["phase_seconds"][phase_name] += elapsed

    @contextmanager
    def timestep(self, timestep: int):
        """Context manager for one simulation timestep total."""
        if not self.enabled:
            yield
            return

        previous_timestep = self._active_timestep
        self._active_timestep = {
            "timestep": int(timestep),
            "phase_seconds": defaultdict(float),
        }
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self._record("timestep.total", elapsed)
            record = {
                "timestep": int(timestep),
                "total_seconds": elapsed,
            }
            record.update(dict(self._active_timestep["phase_seconds"]))
            self._timestep_records.append(record)
            self._active_timestep = previous_timestep

    def phase_summary(self, sort_desc: bool = True) -> pd.DataFrame:
        """Return phase summary with call count, total, and average time."""
        rows = [
            {
                "phase": phase_name,
                "call_count": timing.call_count,
                "total_seconds": timing.total_seconds,
                "avg_seconds": timing.avg_seconds,
            }
            for phase_name, timing in self._phase_timings.items()
        ]
        if not rows:
            return pd.DataFrame(
                columns=["phase", "call_count", "total_seconds", "avg_seconds"]
            )
        summary = pd.DataFrame(rows)
        return summary.sort_values("total_seconds", ascending=not sort_desc).reset_index(
            drop=True
        )

    def timestep_summary(self, sort_by_timestep: bool = True) -> pd.DataFrame:
        """Return per-timestep timing table."""
        if not self._timestep_records:
            return pd.DataFrame(columns=["timestep", "total_seconds"])
        df = pd.DataFrame(self._timestep_records)
        if sort_by_timestep and "timestep" in df.columns:
            df = df.sort_values("timestep")
        return df.reset_index(drop=True)

    def timestep_phase_summary(self) -> pd.DataFrame:
        """Aggregate per-timestep phase totals into count/total/average table."""
        ts = self.timestep_summary()
        if ts.empty:
            return pd.DataFrame(
                columns=["phase", "call_count", "total_seconds", "avg_seconds"]
            )

        excluded = {"timestep", "total_seconds"}
        phase_columns = [c for c in ts.columns if c not in excluded]
        rows = []
        for phase in phase_columns:
            values = ts[phase].fillna(0.0)
            non_zero = values[values > 0.0]
            call_count = int(non_zero.count())
            total = float(values.sum())
            rows.append(
                {
                    "phase": phase,
                    "call_count": call_count,
                    "total_seconds": total,
                    "avg_seconds": (total / call_count if call_count else 0.0),
                }
            )

        summary = pd.DataFrame(rows)
        if summary.empty:
            return pd.DataFrame(
                columns=["phase", "call_count", "total_seconds", "avg_seconds"]
            )
        return summary.sort_values("total_seconds", ascending=False).reset_index(drop=True)

    def print_summary_tables(self) -> None:
        """Print notebook-friendly summary tables."""
        phase_df = self.phase_summary()
        ts_df = self.timestep_summary()
        ts_phase_df = self.timestep_phase_summary()

        print("\n=== Phase timing summary ===")
        print(phase_df.to_string(index=False))

        print("\n=== Per-timestep total timing ===")
        if ts_df.empty:
            print("(no timestep data)")
        else:
            cols = [c for c in ["timestep", "total_seconds"] if c in ts_df.columns]
            print(ts_df[cols].to_string(index=False))

        print("\n=== Per-timestep phase aggregate ===")
        print(ts_phase_df.to_string(index=False))


def run_profiled_runs(
    run_callable: Callable[[SimulationTimingProfiler], Any],
    *,
    runs: int = 1,
    warmup_runs: int = 0,
) -> list[dict[str, Any]]:
    """Run profiled simulations repeatedly with optional warmup runs."""
    if runs < 1:
        raise ValueError("runs must be >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")

    for _ in range(warmup_runs):
        warmup_profiler = SimulationTimingProfiler(enabled=True)
        run_callable(warmup_profiler)

    profiled_runs: list[dict[str, Any]] = []
    for run_index in range(runs):
        profiler = SimulationTimingProfiler(enabled=True)
        output = run_callable(profiler)
        profiled_runs.append(
            {
                "run_index": run_index,
                "profiler": profiler,
                "output": output,
            }
        )
    return profiled_runs


def summarize_profiled_runs(profiled_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate phase timing statistics across multiple profiled runs."""
    if not profiled_runs:
        return pd.DataFrame(
            columns=[
                "phase",
                "run_count",
                "total_seconds_mean",
                "total_seconds_min",
                "total_seconds_max",
                "avg_seconds_mean",
                "call_count_mean",
            ]
        )

    run_phase_frames = []
    for run in profiled_runs:
        run_index = run["run_index"]
        profiler: SimulationTimingProfiler = run["profiler"]
        phase_df = profiler.phase_summary(sort_desc=False).copy()
        if phase_df.empty:
            continue
        phase_df["run_index"] = run_index
        run_phase_frames.append(phase_df)

    if not run_phase_frames:
        return pd.DataFrame(
            columns=[
                "phase",
                "run_count",
                "total_seconds_mean",
                "total_seconds_min",
                "total_seconds_max",
                "avg_seconds_mean",
                "call_count_mean",
            ]
        )

    combined = pd.concat(run_phase_frames, ignore_index=True)
    summary = (
        combined.groupby("phase", as_index=False)
        .agg(
            run_count=("run_index", "nunique"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_min=("total_seconds", "min"),
            total_seconds_max=("total_seconds", "max"),
            avg_seconds_mean=("avg_seconds", "mean"),
            call_count_mean=("call_count", "mean"),
        )
        .sort_values("total_seconds_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary
