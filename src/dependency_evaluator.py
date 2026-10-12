"""Policy-aware dependency evaluation over the type-level knowledge graph.

This module consumes the runtime :class:`~src.dependency_topology.DependencyEdge`
groups produced by :mod:`src.dependency_topology` and evaluates them against
current asset operational state, producing a clean separation between:

* ``intrinsic_operational`` -- physical/repair state, supplied by the caller
  (never mutated here).
* ``hazard_available`` -- whether an asset's own hazard rule currently
  permits operation (hazard exposure + return-to-operational trigger).
* ``dependency_available`` -- whether an asset's dependency edges (if any)
  are satisfied, evaluated policy-aware (``exclusive`` / ``any`` /
  ``at_least_n``) and iterated to a stable fixed point so dependency chains
  converge.
* ``restart_ready`` -- whether any *dependency* restart-delay countdown
  (``restart_wait::<edge_key>``) has finished. Independent of hazard-recovery
  waits.
* ``effective_operational`` -- ``intrinsic_operational & hazard_available &
  dependency_available & restart_ready``. This is the value the simulation
  publishes as ``state.operational`` for all downstream consumers.

Dependency evaluation is strictly hazard-agnostic: it only ever consumes
provider *operational* state (``intrinsic_operational & hazard_available``,
combined recursively with other providers' ``dependency_available``), never
provider failure cause, physical damage, repair time, or fragility state
directly. This module never mutates any of those physical arrays.

Wait-vector namespacing
------------------------
Two independent families of named wait vectors are used, both stored in the
same ``wait_vectors: dict[str, np.ndarray]`` mapping (as already used by
:mod:`src.recovery_scheduler`), but namespaced so they can never collide or be
conflated:

* ``hazard_recovery_wait::<name>`` -- crew-independent hazard recovery
  countdowns (``return_to_operational.trigger == "delayed"`` on a *hazard*
  rule). This replaces the old bare ``dependency_wait``-style naming.
* ``restart_wait::<edge_key>`` -- one independent restart-delay countdown per
  dependency edge, keyed by the edge's stable ``edge_key`` so unrelated
  dependency edges never overwrite one another's timers.

``restart_ready`` only ever inspects ``restart_wait::*`` vectors -- it never
looks at ``hazard_recovery_wait::*`` vectors, and dependency restart waiting
is never treated as physical repair.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from src.dependency_knowledge_graph import (
    POLICY_ANY,
    POLICY_AT_LEAST_N,
    POLICY_EXCLUSIVE,
    TRIGGER_DELAYED,
    TRIGGER_IMMEDIATE,
    TRIGGER_REPAIR_BELOW,
    TRIGGER_REPAIR_COMPLETE,
)
from src.dependency_topology import DependencyEdge, index_edges_by_target
from src.timing_profiler import NULL_PROFILER, NullProfiler

DependencyReport = dict[str, Any]

# ---------------------------------------------------------------------------
# Wait-vector namespaces
# ---------------------------------------------------------------------------
HAZARD_RECOVERY_WAIT_PREFIX = "hazard_recovery_wait::"
RESTART_WAIT_PREFIX = "restart_wait::"


@lru_cache(maxsize=None)
def hazard_recovery_wait_key(wait_vector_name: str) -> str:
    """Namespace a hazard rule's configured wait-vector name.

    Cached: the namespaced key is a pure function of ``wait_vector_name``,
    which is drawn from a small, fixed set of configured names for the whole
    run -- caching avoids re-formatting/re-hashing the same string on every
    timestep.
    """
    return f"{HAZARD_RECOVERY_WAIT_PREFIX}{wait_vector_name}"


@lru_cache(maxsize=None)
def restart_wait_key(edge_key: str) -> str:
    """Namespace a dependency edge's restart-delay wait vector by its stable edge key.

    Cached: ``edge_key`` is stable for the lifetime of a run (edges are
    expanded once per experiment), so caching avoids rebuilding/re-hashing
    the same namespaced string on every timestep for every edge.
    """
    return f"{RESTART_WAIT_PREFIX}{edge_key}"


# ---------------------------------------------------------------------------
# Hazard availability (per-asset-type hazard rule; hazard-only, no dependency
# concepts here at all)
# ---------------------------------------------------------------------------
def _hazard_repair_blocked_mask(
    repair_time: np.ndarray,
    asset_mask: np.ndarray,
    return_to_operational,
    wait_vectors: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Boolean mask of assets blocked because their repair/return trigger is not yet met."""
    blocked = np.zeros(len(repair_time), dtype=bool)
    trigger = return_to_operational.trigger

    if trigger == TRIGGER_IMMEDIATE:
        pass
    elif trigger == TRIGGER_REPAIR_COMPLETE:
        blocked[asset_mask] = repair_time[asset_mask] > 0.0
    elif trigger == TRIGGER_REPAIR_BELOW:
        blocked[asset_mask] = repair_time[asset_mask] > return_to_operational.threshold
    elif trigger == TRIGGER_DELAYED:
        key = hazard_recovery_wait_key(return_to_operational.wait_vector)
        if wait_vectors is None:
            blocked[asset_mask] = True
        else:
            wait_vector = np.asarray(
                wait_vectors.get(key, np.zeros(len(repair_time), dtype=np.float64)),
                dtype=np.float64,
            )
            blocked[asset_mask] = wait_vector[asset_mask] > 0.0

    return blocked


def compute_hazard_availability(
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    wait_vectors: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute ``hazard_available`` purely from current hazard rules + inputs.

    This is a pure function of ``flooded_mask``/``repair_time``/wait-vector
    contents -- it never mutates ``asset_type``, ``flooded_mask``, or
    ``repair_time``, and never touches dependency edges.
    """
    num_assets = len(asset_type)
    if flooded_mask is None:
        flooded_mask = np.zeros(num_assets, dtype=bool)
    else:
        flooded_mask = np.asarray(flooded_mask, dtype=bool)
    if repair_time is None:
        repair_time = np.zeros(num_assets, dtype=np.float64)
    else:
        repair_time = np.asarray(repair_time, dtype=np.float64)

    hazard_available = np.ones(num_assets, dtype=bool)
    for a_type in np.unique(asset_type):
        a_mask = asset_type == a_type
        rules = knowledge_graph.get_hazard_rules(hazard_type, a_type)
        if not rules:
            continue
        blocked = np.zeros(num_assets, dtype=bool)
        for rule in rules:
            if rule.hazard_blocks_operation:
                blocked |= a_mask & flooded_mask
            blocked |= _hazard_repair_blocked_mask(
                repair_time, a_mask, rule.return_to_operational, wait_vectors
            )
        hazard_available &= ~blocked
    return hazard_available


def activate_hazard_recovery_waits(
    hazard_available: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: np.ndarray,
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
) -> None:
    """Start/refresh ``hazard_recovery_wait::<name>`` countdowns for delayed hazard triggers.

    Mutates *wait_vectors* and *active_masks* in place (the same convention
    used by :mod:`src.recovery_scheduler`). Never mutates physical damage,
    repair time, or fragility state.
    """
    flooded_mask = np.asarray(flooded_mask, dtype=bool)
    num_assets = len(asset_type)

    for a_type in np.unique(asset_type):
        a_mask = asset_type == a_type
        for rule in knowledge_graph.get_hazard_rules(hazard_type, a_type):
            if rule.return_to_operational.trigger != TRIGGER_DELAYED:
                continue

            key = hazard_recovery_wait_key(rule.return_to_operational.wait_vector)
            if key not in wait_vectors:
                wait_vectors[key] = np.zeros(num_assets, dtype=np.float64)
            if key not in active_masks:
                active_masks[key] = np.zeros(num_assets, dtype=bool)
            wait_vector = wait_vectors[key]
            active = active_masks[key]

            affected = a_mask & ~hazard_available
            if rule.hazard_blocks_operation:
                affected |= a_mask & flooded_mask
            newly_affected = affected & ~active
            if np.any(newly_affected):
                wait_vector[newly_affected] = np.maximum(
                    wait_vector[newly_affected], rule.return_to_operational.delay_steps
                )
                active[newly_affected] = True


def clear_completed_hazard_recovery_waits(
    hazard_available: np.ndarray,
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
) -> None:
    """Allow a later hazard disruption to start a fresh hazard-recovery countdown."""
    hazard_available = np.asarray(hazard_available, dtype=bool)
    for key, active in active_masks.items():
        if not key.startswith(HAZARD_RECOVERY_WAIT_PREFIX):
            continue
        wait_vector = wait_vectors.get(key)
        if wait_vector is None:
            continue
        active[hazard_available & (wait_vector <= 0.0)] = False


# ---------------------------------------------------------------------------
# Policy-aware dependency-edge evaluation
# ---------------------------------------------------------------------------
def _provider_available(
    provider_index: int,
    baseline_available: np.ndarray,
    dependency_available: np.ndarray,
) -> bool:
    """A provider is available when its own baseline (intrinsic & hazard) state
    is available *and* its own dependencies (if it is itself a target
    elsewhere) are currently satisfied -- this is what allows dependency
    chains to propagate."""
    return bool(baseline_available[provider_index]) and bool(dependency_available[provider_index])


def evaluate_edge_availability(
    edge: DependencyEdge,
    baseline_available: np.ndarray,
    dependency_available: np.ndarray,
) -> bool:
    """Evaluate a single :class:`DependencyEdge` per its ``availability_policy``.

    * ``exclusive``: the topology must have produced exactly one governing
      provider. Zero providers -> unavailable. More than one provider is a
      topology-contract violation and raises :class:`ValueError` rather than
      silently picking one.
    * ``any``: available if at least one qualifying provider is available.
      Zero providers -> unavailable.
    * ``at_least_n``: available if at least ``minimum_available`` qualifying
      providers are available. Zero providers -> unavailable. Fewer
      qualifying providers than ``minimum_available`` -> unavailable even if
      every qualifying provider is available.
    """
    providers = edge.provider_indices

    if edge.availability_policy == POLICY_EXCLUSIVE:
        if len(providers) == 0:
            return False
        if len(providers) > 1:
            raise ValueError(
                f"Exclusive dependency edge '{edge.edge_key}' resolved to "
                f"{len(providers)} providers ({providers!r}); exclusive "
                "topology must assign exactly one governing provider with no "
                "fallback. This indicates a topology-contract violation "
                "upstream -- refusing to silently select one."
            )
        return _provider_available(providers[0], baseline_available, dependency_available)

    if edge.availability_policy == POLICY_ANY:
        if not providers:
            return False
        return any(
            _provider_available(p, baseline_available, dependency_available)
            for p in providers
        )

    if edge.availability_policy == POLICY_AT_LEAST_N:
        if edge.minimum_available is None or edge.minimum_available < 1:
            raise ValueError(
                f"Dependency edge '{edge.edge_key}' uses availability_policy="
                f"'at_least_n' but minimum_available is missing or invalid "
                f"({edge.minimum_available!r}); it must be a positive integer."
            )
        if not providers:
            return False
        if len(providers) < edge.minimum_available:
            # Fewer qualifying providers than required -- unavailable even if
            # every qualifying provider is operational.
            return False
        available_count = sum(
            1
            for p in providers
            if _provider_available(p, baseline_available, dependency_available)
        )
        return available_count >= edge.minimum_available

    raise ValueError(
        f"Unsupported availability_policy '{edge.availability_policy}' on "
        f"edge '{edge.edge_key}'."
    )


def evaluate_dependency_availability(
    edges: list[DependencyEdge],
    baseline_available: np.ndarray,
    *,
    num_assets: int,
    max_iterations: int | None = None,
) -> tuple[np.ndarray, DependencyReport]:
    """Evaluate ``dependency_available`` for every asset from runtime edges.

    Every asset starts assumed available (``True``) -- an asset with *no*
    dependency edges at all has nothing to evaluate and stays available.
    Assets that are the target of one or more edges are only available if
    every one of their edges is satisfied per its policy.

    Dependency chains (a provider that is itself a dependent target
    elsewhere) are supported by iterating to a fixed point. Because a
    target's ``dependency_available`` flag is only ever cleared (``True ->
    False``, never the reverse) once evaluated, and each target is only
    re-evaluated while still ``True``, the whole array is a monotonically
    non-increasing sequence with at most ``len(edges_by_target)`` possible
    ``True -> False`` flips in total -- so the loop is *provably* guaranteed
    to reach a stable fixed point within ``len(edges_by_target) + 1`` passes,
    for any graph shape, including cycles. If the loop nonetheless exits
    still ``changed`` (which should be unreachable given the above -- e.g.
    only possible if *max_iterations* is deliberately overridden below the
    natural bound, or a future bug breaks monotonicity), this function
    raises :class:`RuntimeError` rather than silently returning a stale,
    under-propagated result.

    Args:
        max_iterations: Override for the fixed-point pass bound. Defaults to
            ``len(edges_by_target) + 1`` (the proven-sufficient bound).
            Exposed mainly for testing the non-convergence guard itself.

    This function is strictly hazard-agnostic: *baseline_available* is
    expected to already combine intrinsic operational state with hazard
    availability (``intrinsic_operational & hazard_available``); this
    function never inspects hazard/flood/repair inputs directly and never
    mutates *baseline_available*.
    """
    baseline_available = np.asarray(baseline_available, dtype=bool)
    dependency_available = np.ones(num_assets, dtype=bool)
    edges_by_target = index_edges_by_target(edges)

    unavailable_targets: set[int] = set()
    if max_iterations is None:
        max_iterations = len(edges_by_target) + 1
    iterations = 0
    changed = True
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for target_index, target_edges in edges_by_target.items():
            if not dependency_available[target_index]:
                continue
            for edge in target_edges:
                if not evaluate_edge_availability(edge, baseline_available, dependency_available):
                    dependency_available[target_index] = False
                    unavailable_targets.add(target_index)
                    changed = True
                    break

    if changed:
        raise RuntimeError(
            "Dependency graph evaluation did not converge to a stable fixed "
            f"point within {max_iterations} pass(es) across "
            f"{len(edges_by_target)} target(s). Refusing to silently return "
            "a stale, under-propagated dependency_available result. This "
            "should be unreachable for the built-in monotonic evaluation -- "
            "check for a custom max_iterations override or a non-monotonic "
            "edge-evaluation change."
        )

    report: DependencyReport = {
        "evaluated_target_count": len(edges_by_target),
        "unavailable_target_count": len(unavailable_targets),
        "unavailable_targets": sorted(unavailable_targets),
        "blocked_count": len(unavailable_targets),
        "active_rule_count": len(edges),
        "active_rules": sorted(
            {edge.edge_key for edge in edges if not dependency_available[edge.target_index]}
        ),
        "iterations": iterations,
    }
    return dependency_available, report


# ---------------------------------------------------------------------------
# Dependency restart waits
# ---------------------------------------------------------------------------
# Per-``edges`` object memoization of the (target_indices, namespaced_keys)
# arrays used to vectorize `activate_dependency_restart_waits`. Keyed by
# `id(edges)` with an identity check against the stored object itself (not
# just the id) to guard against id reuse after garbage collection. `edges`
# is built once per experiment (see `expand_dependency_edges`) and passed by
# reference unchanged across every timestep, so this cache is populated once
# per experiment and then reused for the rest of that experiment's run.
_RESTART_EDGE_INDEX_CACHE: dict[int, tuple[list, np.ndarray, tuple[str, ...]]] = {}


def _restart_edge_index(edges: list[DependencyEdge]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return (target_indices, namespaced_keys) for *edges*, memoized by identity."""
    cache_key = id(edges)
    cached = _RESTART_EDGE_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is edges:
        return cached[1], cached[2]

    target_indices = np.fromiter(
        (edge.target_index for edge in edges), dtype=np.int64, count=len(edges)
    )
    keys = tuple(restart_wait_key(edge.edge_key) for edge in edges)
    _RESTART_EDGE_INDEX_CACHE[cache_key] = (edges, target_indices, keys)
    return target_indices, keys


def activate_dependency_restart_waits(
    dependency_available: np.ndarray,
    previous_dependency_available: np.ndarray | None,
    edges: list[DependencyEdge],
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
    *,
    restart_delay_steps: float = 0.0,
    num_assets: int,
) -> None:
    """(Re)start or reset each edge's namespaced ``restart_wait::<edge_key>`` timer.

    * A required dependency transitioning from unavailable to available
      (re)starts its configured restart delay -- namespaced per edge key so
      independent edges never overwrite one another's timers.
    * A required dependency that is (still or newly) unavailable resets its
      restart timer, so the next recovery starts a fresh countdown.
    * If a dependency was already available on the previous evaluation (no
      transition), its existing timer (if any) is left untouched so it can
      keep counting down via the generic wait-vector decrement mechanism.

    When *previous_dependency_available* is ``None`` (first-ever call), no
    transition is assumed for targets already available, so no spurious
    restart delay is applied at start-up.

    Performance note: the per-edge ``now_available``/``was_available``
    comparison is vectorized across all edges at once (a single gather per
    array), and the actual per-edge wait-vector/active-mask writes -- the
    only part that cannot be vectorized, since each edge owns an
    independently namespaced array -- are restricted to edges whose
    availability actually transitioned this timestep (typically a small
    subset, since a flooded-state transition only occurs at the start/end
    of a flood event, not on every timestep within one), rather than
    looping over every edge unconditionally.
    """
    if not edges:
        return

    target_indices, keys = _restart_edge_index(edges)

    # Every edge's namespaced entries must exist in `wait_vectors`/
    # `active_masks` after this call, even if nothing transitions this
    # timestep (downstream/test callers rely on the zero-initialized entry
    # being present from the first evaluation onward). This membership
    # check is O(num_edges) but cheap (a dict lookup, not an allocation) --
    # the actual `np.zeros(...)` allocation only ever happens once per edge,
    # the first time its key is seen.
    for key in keys:
        if key not in wait_vectors:
            wait_vectors[key] = np.zeros(num_assets, dtype=np.float64)
        if key not in active_masks:
            active_masks[key] = np.zeros(num_assets, dtype=bool)

    now_available = dependency_available[target_indices]
    if previous_dependency_available is None:
        was_available = now_available
    else:
        was_available = previous_dependency_available[target_indices]

    became_available = now_available & ~was_available
    still_unavailable_or_newly_unavailable = ~now_available
    # An edge needs a write this timestep iff it just recovered (start its
    # restart delay) or it is (still or newly) unavailable (keep resetting
    # its timer, matching the original per-edge `elif not now_available`
    # branch exactly). Edges that are unchanged-and-available need no write
    # at all -- their existing timer, if any, keeps counting down untouched.
    needs_write = became_available | still_unavailable_or_newly_unavailable
    if not np.any(needs_write):
        return

    for i in np.flatnonzero(needs_write):
        key = keys[i]
        target = target_indices[i]

        if became_available[i]:
            wait_vectors[key][target] = restart_delay_steps
            active_masks[key][target] = True
        else:
            wait_vectors[key][target] = 0.0
            active_masks[key][target] = False


def compute_restart_ready(
    num_assets: int,
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
) -> np.ndarray:
    """Compute ``restart_ready`` from ``restart_wait::*`` vectors only.

    Deliberately ignores ``hazard_recovery_wait::*`` (and any other
    non-``restart_wait::``-prefixed) vectors -- dependency restart waiting
    must never be conflated with hazard-recovery waiting or physical repair.
    """
    ready = np.ones(num_assets, dtype=bool)
    for key, active in active_masks.items():
        if not key.startswith(RESTART_WAIT_PREFIX):
            continue
        vector = wait_vectors.get(key)
        if vector is None:
            continue
        blocked = np.asarray(active, dtype=bool) & (np.asarray(vector, dtype=np.float64) > 0.0)
        ready &= ~blocked
    return ready


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def evaluate_operational_state(
    *,
    intrinsic_operational: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    dependency_edges: list[DependencyEdge] | None = None,
    wait_vectors: dict[str, np.ndarray],
    hazard_active_masks: dict[str, np.ndarray],
    restart_active_masks: dict[str, np.ndarray],
    previous_dependency_available: np.ndarray | None = None,
    restart_delay_steps: float = 0.0,
    profiler=NULL_PROFILER,
) -> tuple[np.ndarray, DependencyReport]:
    """Compute the fully separated operational-state layers for one timestep.

    Returns ``(effective_operational, report)``. *report* always contains
    ``intrinsic_operational``, ``hazard_available``, ``dependency_available``,
    ``restart_ready``, and ``effective_operational`` (each a fresh copy, safe
    to store/compare), plus diagnostic dependency-evaluation fields.

    Never mutates *intrinsic_operational*, *flooded_mask*, or *repair_time*.
    Mutates *wait_vectors*, *hazard_active_masks*, and *restart_active_masks*
    in place (the existing :mod:`src.recovery_scheduler` convention) to
    record wait-vector countdown state across timesteps.
    """
    if profiler is None:
        profiler = NullProfiler()

    num_assets = len(asset_type)
    intrinsic_operational = np.asarray(intrinsic_operational, dtype=bool)
    dependency_edges = dependency_edges or []

    with profiler.section("dependency.hazard_availability"):
        hazard_available = compute_hazard_availability(
            asset_type,
            hazard_type,
            knowledge_graph,
            flooded_mask=flooded_mask,
            repair_time=repair_time,
            wait_vectors=wait_vectors,
        )
        activate_hazard_recovery_waits(
            hazard_available,
            asset_type,
            hazard_type,
            knowledge_graph,
            flooded_mask=(
                np.zeros(num_assets, dtype=bool) if flooded_mask is None
                else np.asarray(flooded_mask, dtype=bool)
            ),
            wait_vectors=wait_vectors,
            active_masks=hazard_active_masks,
        )
        clear_completed_hazard_recovery_waits(hazard_available, wait_vectors, hazard_active_masks)

    baseline_available = intrinsic_operational & hazard_available

    with profiler.section("dependency.availability"):
        dependency_available, dependency_report = evaluate_dependency_availability(
            dependency_edges, baseline_available, num_assets=num_assets
        )

    with profiler.section("dependency.restart_waits"):
        if restart_delay_steps <= 0:
            # A non-positive global restart delay means every dependency
            # restart is immediate by definition -- `compute_restart_ready`
            # would always evaluate to all-True regardless of activation
            # history (see `activate_dependency_restart_waits`: a started
            # wait is recorded with `vector[target] = restart_delay_steps`,
            # and `compute_restart_ready` only blocks while `vector > 0.0`).
            # Skip the whole activate/compute pass rather than doing
            # provably-inert bookkeeping on every timestep.
            restart_ready = np.ones(num_assets, dtype=bool)
        else:
            activate_dependency_restart_waits(
                dependency_available,
                previous_dependency_available,
                dependency_edges,
                wait_vectors,
                restart_active_masks,
                restart_delay_steps=restart_delay_steps,
                num_assets=num_assets,
            )
            restart_ready = compute_restart_ready(num_assets, wait_vectors, restart_active_masks)

    effective_operational = baseline_available & dependency_available & restart_ready

    report: DependencyReport = {
        **dependency_report,
        "intrinsic_operational": intrinsic_operational.copy(),
        "hazard_available": hazard_available.copy(),
        "dependency_available": dependency_available.copy(),
        "restart_ready": restart_ready.copy(),
        "effective_operational": effective_operational.copy(),
        "dependency_blocked_mask": ~dependency_available,
        "warning": None,
    }
    return effective_operational, report
