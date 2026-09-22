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


def hazard_recovery_wait_key(wait_vector_name: str) -> str:
    """Namespace a hazard rule's configured wait-vector name."""
    return f"{HAZARD_RECOVERY_WAIT_PREFIX}{wait_vector_name}"


def restart_wait_key(edge_key: str) -> str:
    """Namespace a dependency edge's restart-delay wait vector by its stable edge key."""
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
            wait_vector = wait_vectors.setdefault(key, np.zeros(num_assets, dtype=np.float64))
            active = active_masks.setdefault(key, np.zeros(num_assets, dtype=bool))

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
    """
    for edge in edges:
        key = restart_wait_key(edge.edge_key)
        vector = wait_vectors.setdefault(key, np.zeros(num_assets, dtype=np.float64))
        active = active_masks.setdefault(key, np.zeros(num_assets, dtype=bool))
        target = edge.target_index

        now_available = bool(dependency_available[target])
        if previous_dependency_available is None:
            was_available = now_available
        else:
            was_available = bool(previous_dependency_available[target])

        if now_available and not was_available:
            vector[target] = restart_delay_steps
            active[target] = True
        elif not now_available:
            vector[target] = 0.0
            active[target] = False
        # else: still available, no new transition -- leave the existing
        # timer alone so it can continue counting down.


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
