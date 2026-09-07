"""Lightweight timing utilities for simulation profiling.

The profiler is intentionally simple and notebook-friendly:
- phase-level counters/totals/averages (inclusive *and* exclusive/self time)
- optional per-timestep timing aggregation, labelled by loop
- optional repeated-run execution with warmup

Attribution is structural rather than opt-in: any :meth:`SimulationTimingProfiler.section`
active while a :meth:`SimulationTimingProfiler.timestep` is open automatically
contributes its *self* time (its own elapsed time minus any nested sections) to
that timestep, and whatever time is left over is recorded as a first-class
``<loop>.unattributed`` phase.  This removes the previous ``include_in_timestep``
opt-in flag, which was the direct cause of unattributed "blind spots" in
profiled runs.

When no profiler is supplied, callers should default to :class:`NullProfiler`,
which is a shared, allocation-free stand-in so that unprofiled runs pay
essentially no overhead.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterator
import warnings

import pandas as pd


@dataclass
class PhaseTiming:
    """Aggregated timing for a single named phase.

    ``total_seconds`` is the *inclusive* time spent in the phase (including any
    nested sections); ``self_seconds`` is the *exclusive* time spent directly
    in the phase, i.e. ``total_seconds`` minus the inclusive time of any
    nested sections encountered while this phase was active.
    """

    call_count: int = 0
    total_seconds: float = 0.0
    self_seconds: float = 0.0

    @property
    def avg_seconds(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_seconds / self.call_count

    @property
    def avg_self_seconds(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.self_seconds / self.call_count


class SimulationTimingProfiler:
    """Collect named wall-clock timings with optional per-timestep detail.

    Sections form a stack: whichever section is innermost accrues elapsed
    time as usual, and when it completes it reports its inclusive elapsed
    time to its parent (section or timestep) so the parent's *self* time can
    be computed as ``elapsed - children_elapsed``.  Any section active while
    a timestep is open automatically contributes its self time to that
    timestep — there is no opt-in flag.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._phase_timings: dict[str, PhaseTiming] = {}
        self._timestep_records: list[dict[str, Any]] = []
        self._active_timestep: dict[str, Any] | None = None
        self._active_starts: dict[str, float] = {}
        # Stack of "children elapsed" accumulators, one per currently-open
        # section/timestep frame, used to derive self (exclusive) time.
        self._active_stack: list[float] = []

    def _record(self, phase_name: str, elapsed_seconds: float, self_seconds: float) -> None:
        timing = self._phase_timings.get(phase_name)
        if timing is None:
            timing = PhaseTiming()
            self._phase_timings[phase_name] = timing
        timing.call_count += 1
        timing.total_seconds += elapsed_seconds
        timing.self_seconds += self_seconds
        if self_seconds < 0:
            warnings.warn(
                f"Negative self time ({self_seconds:.9f}s) recorded for phase "
                f"'{phase_name}'; this indicates mismatched profiler section nesting.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _enter_frame(self) -> float:
        """Push a new frame on the active stack and return its start time."""
        self._active_stack.append(0.0)
        return perf_counter()

    def _exit_frame(self, start: float) -> tuple[float, float]:
        """Pop the active frame, returning ``(elapsed, self_elapsed)``.

        Also reports this frame's inclusive elapsed time to its parent frame
        (if any) so the parent can subtract it when computing its own self
        time.
        """
        elapsed = perf_counter() - start
        children_seconds = self._active_stack.pop()
        self_elapsed = elapsed - children_seconds
        if self._active_stack:
            self._active_stack[-1] += elapsed
        return elapsed, self_elapsed

    def start_timer(self, phase_name: str) -> None:
        """Start a named timer to be stopped with :meth:`stop_timer`."""
        if not self.enabled:
            return
        self._active_starts[phase_name] = self._enter_frame()

    def stop_timer(self, phase_name: str) -> None:
        """Stop a named timer and record elapsed/self time."""
        if not self.enabled:
            return
        start = self._active_starts.pop(phase_name, None)
        if start is None:
            raise KeyError(f"Timer '{phase_name}' was not started.")
        elapsed, self_elapsed = self._exit_frame(start)
        self._record(phase_name, elapsed, self_elapsed)
        active_timestep = self._active_timestep
        if active_timestep is not None:
            active_timestep["phase_seconds"][phase_name] += self_elapsed

    @contextmanager
    def section(self, phase_name: str) -> Iterator[None]:
        """Context manager for a named timed section.

        Any section active while a timestep is open automatically
        contributes its *self* time to that timestep's phase totals.
        """
        if not self.enabled:
            yield
            return

        start = self._enter_frame()
        try:
            yield
        finally:
            elapsed, self_elapsed = self._exit_frame(start)
            self._record(phase_name, elapsed, self_elapsed)
            active_timestep = self._active_timestep
            if active_timestep is not None:
                active_timestep["phase_seconds"][phase_name] += self_elapsed

    @contextmanager
    def timestep(self, timestep: int, *, loop: str = "simulation") -> Iterator[None]:
        """Context manager for one timestep total, labelled by *loop*.

        *loop* distinguishes concurrent per-timestep loops (e.g. the
        simulation loop vs. the societal-access postprocessing loop) so they
        do not merge into a single ``timestep.total`` row.  On exit, whatever
        elapsed time was not attributed to any nested section is recorded as
        a first-class ``<loop>.unattributed`` phase.
        """
        if not self.enabled:
            yield
            return

        previous_timestep = self._active_timestep
        phase_seconds: dict[str, float] = defaultdict(float)
        self._active_timestep = {
            "timestep": int(timestep),
            "loop": loop,
            "phase_seconds": phase_seconds,
        }
        start = self._enter_frame()
        try:
            yield
        finally:
            elapsed, self_elapsed = self._exit_frame(start)
            total_phase_name = f"{loop}.total"
            self._record(total_phase_name, elapsed, self_elapsed)

            unattributed_phase_name = f"{loop}.unattributed"
            unattributed = elapsed - sum(phase_seconds.values())
            self._record(unattributed_phase_name, unattributed, unattributed)

            record = {
                "timestep": int(timestep),
                "loop": loop,
                "total_seconds": elapsed,
            }
            record.update(dict(phase_seconds))
            record[unattributed_phase_name] = unattributed
            self._timestep_records.append(record)
            self._active_timestep = previous_timestep

    def phase_summary(self, sort_desc: bool = True) -> pd.DataFrame:
        """Return phase summary with call count, total, average, and self time."""
        columns = [
            "phase",
            "call_count",
            "total_seconds",
            "avg_seconds",
            "self_seconds",
            "self_pct",
        ]
        if not self._phase_timings:
            return pd.DataFrame(columns=columns)

        total_self_seconds = sum(t.self_seconds for t in self._phase_timings.values())
        rows = [
            {
                "phase": phase_name,
                "call_count": timing.call_count,
                "total_seconds": timing.total_seconds,
                "avg_seconds": timing.avg_seconds,
                "self_seconds": timing.self_seconds,
                "self_pct": (
                    100.0 * timing.self_seconds / total_self_seconds
                    if total_self_seconds > 0
                    else 0.0
                ),
            }
            for phase_name, timing in self._phase_timings.items()
        ]
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

        excluded = {"timestep", "total_seconds", "loop"}
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

        print("\n=== Unattributed time (blind spots) ===")
        if phase_df.empty:
            unattributed_df = phase_df
        else:
            unattributed_df = phase_df[phase_df["phase"].str.endswith(".unattributed")]
        if unattributed_df.empty:
            print("(no unattributed time recorded)")
        else:
            print(unattributed_df.to_string(index=False))

        print("\n=== Per-timestep total timing ===")
        if ts_df.empty:
            print("(no timestep data)")
        else:
            cols = [c for c in ["timestep", "loop", "total_seconds"] if c in ts_df.columns]
            print(ts_df[cols].to_string(index=False))

        print("\n=== Per-timestep phase aggregate ===")
        print(ts_phase_df.to_string(index=False))


class NullProfiler:
    """No-op stand-in for :class:`SimulationTimingProfiler`.

    Keeps unprofiled runs essentially free: ``section()`` and ``timestep()``
    return the same pre-built, allocation-free reusable context manager on
    every call (no timer reads, no dict writes, no per-call object
    construction).  This is the default ``profiler`` value throughout the
    codebase so callers no longer need an ``if profiler is not None``
    branch around every profiled section.
    """

    #: Shared, stateless, reentrant-safe context manager instance.
    _CONTEXT = nullcontext()

    def section(self, phase_name: str) -> nullcontext:
        return self._CONTEXT

    def timestep(self, timestep: int, *, loop: str = "simulation") -> nullcontext:
        return self._CONTEXT

    def start_timer(self, phase_name: str) -> None:
        return None

    def stop_timer(self, phase_name: str) -> None:
        return None


#: Shared singleton used as the default ``profiler`` argument value.
NULL_PROFILER = NullProfiler()


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
                "self_seconds_mean",
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
                "self_seconds_mean",
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
            self_seconds_mean=("self_seconds", "mean"),
        )
        .sort_values("total_seconds_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary
