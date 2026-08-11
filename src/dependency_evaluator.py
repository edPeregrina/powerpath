"""Dependency evaluation layer for operational-state blocking.

This module provides two evaluation paths:

1. **Legacy path** – :func:`evaluate_dependencies` uses flat flags
   (``enable_default_rules``, ``require_repair_for_operational``) to preserve
   backwards compatibility with existing call sites.

2. **Graph-aware path** – :func:`evaluate_dependencies_from_graph` consults a
   :class:`~src.dependency_knowledge_graph.DependencyKnowledgeGraph` to apply
   per-pair rules between hazard types, primary asset types (A), and optional
   downstream asset types (B) within A's service area.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DependencyContext = Dict[str, Any]
DependencyReport = Dict[str, Any]


def build_dependency_context(
    asset_type: np.ndarray,
    hazard_values: Optional[np.ndarray] = None,
    flooded_mask: Optional[np.ndarray] = None,
    repair_time: Optional[np.ndarray] = None,
    repair_threshold: float = 0.0,
    dependency_map: Optional[Dict[Any, Iterable[Any]]] = None,
    area_dependencies: Optional[Iterable[Dict[str, Any]]] = None,
    pairwise_dependencies: Optional[Iterable[Tuple[int, int]]] = None,
) -> DependencyContext:
    """Build a normalized dependency context dictionary.

    The context is intentionally lightweight for first-pass integration and can
    be extended in-place without changing call sites.
    """
    num_assets = len(asset_type)
    if hazard_values is None:
        hazard_values = np.zeros(num_assets, dtype=np.float64)
    if flooded_mask is None:
        flooded_mask = np.zeros(num_assets, dtype=bool)
    if repair_time is None:
        repair_time = np.zeros(num_assets, dtype=np.float64)

    return {
        "asset_type": np.asarray(asset_type),
        "hazard_values": np.asarray(hazard_values),
        "flooded_mask": np.asarray(flooded_mask, dtype=bool),
        "repair_time": np.asarray(repair_time, dtype=np.float64),
        "repair_threshold": float(repair_threshold),
        "dependency_map": dependency_map or {},
        "area_dependencies": list(area_dependencies) if area_dependencies is not None else [],
        "pairwise_dependencies": list(pairwise_dependencies) if pairwise_dependencies is not None else [],
    }


def evaluate_area_dependencies(context: DependencyContext) -> np.ndarray:
    """Evaluate area-based dependency constraints.

    Placeholder implementation returns no additional blocking for now.
    """
    return np.zeros(len(context["asset_type"]), dtype=bool)


def evaluate_pairwise_dependencies(context: DependencyContext) -> np.ndarray:
    """Evaluate pairwise dependency constraints.

    Placeholder implementation returns no additional blocking for now.
    """
    return np.zeros(len(context["asset_type"]), dtype=bool)


def evaluate_dependency_rules(
    context: DependencyContext,
    *,
    area_blocked_mask: Optional[np.ndarray] = None,
    pairwise_blocked_mask: Optional[np.ndarray] = None,
    enable_default_rules: bool = False,
    require_repair_for_operational: bool = False,
) -> np.ndarray:
    """Combine dependency rules into one blocking mask.

    Default rules are disabled by default to preserve existing behavior.
    This function boundary is where rules such as msls/road behavior can be
    activated in future passes.
    """
    num_assets = len(context["asset_type"])
    blocked_mask = np.zeros(num_assets, dtype=bool)

    if area_blocked_mask is not None:
        blocked_mask |= np.asarray(area_blocked_mask, dtype=bool)
    if pairwise_blocked_mask is not None:
        blocked_mask |= np.asarray(pairwise_blocked_mask, dtype=bool)

    if enable_default_rules:
        asset_type = context["asset_type"]
        flooded_mask = context["flooded_mask"]

        # Future-compatible baseline rule: roads are non-operational while flooded.
        road_mask = asset_type == "road"
        blocked_mask |= road_mask & flooded_mask

    if require_repair_for_operational:
        repair_time = context["repair_time"]
        repair_threshold = context["repair_threshold"]
        blocked_mask |= repair_time > repair_threshold

    return blocked_mask


def apply_dependency_blocking(operational: np.ndarray, blocked_mask: np.ndarray) -> np.ndarray:
    """Apply dependency blocking to the operational state."""
    return np.asarray(operational, dtype=bool) & ~np.asarray(blocked_mask, dtype=bool)


def build_dependency_report(
    *,
    blocked_mask: np.ndarray,
    area_blocked_mask: np.ndarray,
    pairwise_blocked_mask: np.ndarray,
) -> DependencyReport:
    """Build a concise report suitable for logging/debugging."""
    return {
        "blocked_count": int(np.sum(blocked_mask)),
        "area_blocked_count": int(np.sum(area_blocked_mask)),
        "pairwise_blocked_count": int(np.sum(pairwise_blocked_mask)),
    }


def evaluate_dependencies(
    operational: np.ndarray,
    asset_type: np.ndarray,
    *,
    hazard_values: Optional[np.ndarray] = None,
    flooded_mask: Optional[np.ndarray] = None,
    repair_time: Optional[np.ndarray] = None,
    repair_threshold: float = 0.0,
    dependency_map: Optional[Dict[Any, Iterable[Any]]] = None,
    area_dependencies: Optional[Iterable[Dict[str, Any]]] = None,
    pairwise_dependencies: Optional[Iterable[Tuple[int, int]]] = None,
    enable_default_rules: bool = False,
    require_repair_for_operational: bool = False,
    return_report: bool = False,
):
    """High-level dependency evaluation entry point (legacy path).

    Returns updated operational state, plus report when requested.
    """
    context = build_dependency_context(
        asset_type,
        hazard_values=hazard_values,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
        repair_threshold=repair_threshold,
        dependency_map=dependency_map,
        area_dependencies=area_dependencies,
        pairwise_dependencies=pairwise_dependencies,
    )
    area_blocked_mask = evaluate_area_dependencies(context)
    pairwise_blocked_mask = evaluate_pairwise_dependencies(context)
    blocked_mask = evaluate_dependency_rules(
        context,
        area_blocked_mask=area_blocked_mask,
        pairwise_blocked_mask=pairwise_blocked_mask,
        enable_default_rules=enable_default_rules,
        require_repair_for_operational=require_repair_for_operational,
    )
    updated_operational = apply_dependency_blocking(operational, blocked_mask)

    if not return_report:
        return updated_operational

    report = build_dependency_report(
        blocked_mask=blocked_mask,
        area_blocked_mask=area_blocked_mask,
        pairwise_blocked_mask=pairwise_blocked_mask,
    )
    return updated_operational, report


# ---------------------------------------------------------------------------
# Graph-aware evaluation path
# ---------------------------------------------------------------------------

def _compute_repair_blocked_mask(
    repair_time: np.ndarray,
    asset_mask: np.ndarray,
    return_to_operational,
) -> np.ndarray:
    """Return a boolean mask of assets blocked due to repair status.

    Args:
        repair_time: Per-asset remaining repair time array.
        asset_mask: Boolean mask selecting which assets this rule applies to.
        return_to_operational: A :class:`ReturnToOperational` instance from the
            knowledge graph rule.

    Returns:
        Boolean array; ``True`` where an asset is blocked because repair
        requirements are not yet satisfied.
    """
    from src.dependency_knowledge_graph import (
        TRIGGER_IMMEDIATE,
        TRIGGER_REPAIR_COMPLETE,
        TRIGGER_REPAIR_BELOW,
    )

    blocked = np.zeros(len(repair_time), dtype=bool)
    trigger = return_to_operational.trigger

    if trigger == TRIGGER_IMMEDIATE:
        # No repair needed – asset is operational as soon as hazard clears.
        pass
    elif trigger == TRIGGER_REPAIR_COMPLETE:
        # Asset must have zero remaining repair time.
        blocked[asset_mask] = repair_time[asset_mask] > 0.0
    elif trigger == TRIGGER_REPAIR_BELOW:
        threshold = return_to_operational.threshold
        blocked[asset_mask] = repair_time[asset_mask] >= threshold

    return blocked


def _compute_restore_eligible_mask(
    repair_time: np.ndarray,
    flooded_mask: np.ndarray,
    asset_mask: np.ndarray,
    return_to_operational,
) -> np.ndarray:
    """Return a boolean mask of assets eligible to be restored to operational.

    An asset is eligible when it is no longer flooded AND its
    ``return_to_operational`` trigger condition is met.

    Args:
        repair_time: Per-asset remaining repair time array.
        flooded_mask: Boolean array; ``True`` where the asset is currently flooded.
        asset_mask: Boolean mask selecting which assets this rule applies to.
        return_to_operational: A :class:`ReturnToOperational` instance from the
            knowledge graph rule.

    Returns:
        Boolean array; ``True`` where an asset may be restored to operational.
    """
    from src.dependency_knowledge_graph import (
        TRIGGER_IMMEDIATE,
        TRIGGER_REPAIR_COMPLETE,
        TRIGGER_REPAIR_BELOW,
    )

    eligible = np.zeros(len(repair_time), dtype=bool)
    trigger = return_to_operational.trigger
    not_flooded = asset_mask & ~flooded_mask

    if trigger == TRIGGER_IMMEDIATE:
        # Asset can return as soon as hazard clears, regardless of repair state.
        eligible[not_flooded] = True
    elif trigger == TRIGGER_REPAIR_COMPLETE:
        # Repair must be fully complete (repair_time == 0).
        eligible[not_flooded] = repair_time[not_flooded] == 0.0
    elif trigger == TRIGGER_REPAIR_BELOW:
        threshold = return_to_operational.threshold
        eligible[not_flooded] = repair_time[not_flooded] < threshold

    return eligible


def restore_operational_from_graph(
    operational: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: Optional[np.ndarray] = None,
    repair_time: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Restore assets to operational based on knowledge graph return-to-operational triggers.

    This is the positive-restore counterpart to :func:`evaluate_dependencies_from_graph`.
    It re-enables assets that were previously marked non-operational and now satisfy
    both conditions:

    1. The asset is no longer flooded.
    2. The ``return_to_operational`` trigger for its asset type is satisfied
       (e.g. ``repair_complete``: repair_time == 0; ``repair_below``: repair_time
       < threshold; ``immediate``: always satisfied once hazard clears).

    This function only enables assets — it never disables them.  Call it *before*
    :func:`evaluate_dependencies_from_graph` so that the block pass can immediately
    re-suppress any asset that still does not meet conditions (e.g. still flooded).

    Args:
        operational: Boolean array of current operational states (will not be mutated).
        asset_type: String array of asset types corresponding to each index.
        hazard_type: The active hazard type (e.g. ``"flooding"``).
        knowledge_graph: A :class:`~src.dependency_knowledge_graph.DependencyKnowledgeGraph`
            instance.
        flooded_mask: Boolean array; ``True`` where hazard exposure exceeds the flood threshold.
        repair_time: Per-asset remaining repair time array.

    Returns:
        Updated operational array with eligible assets restored to ``True``.
    """
    num_assets = len(asset_type)
    operational = np.asarray(operational, dtype=bool).copy()

    if flooded_mask is None:
        flooded_mask = np.zeros(num_assets, dtype=bool)
    else:
        flooded_mask = np.asarray(flooded_mask, dtype=bool)

    if repair_time is None:
        repair_time = np.zeros(num_assets, dtype=np.float64)
    else:
        repair_time = np.asarray(repair_time, dtype=np.float64)

    unique_asset_types = np.unique(asset_type)
    for a_type in unique_asset_types:
        a_mask = asset_type == a_type
        direct_rules = knowledge_graph.get_rules_or_default(hazard_type, a_type, asset_type_b=None)
        for rule in direct_rules:
            if rule.relationship != "direct":
                continue
            eligible = _compute_restore_eligible_mask(
                repair_time, flooded_mask, a_mask, rule.return_to_operational
            )
            # Only restore assets that were previously non-operational.
            operational |= eligible & ~operational

    return operational


def evaluate_dependencies_from_graph(
    operational: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: Optional[np.ndarray] = None,
    repair_time: Optional[np.ndarray] = None,
    service_area_map: Optional[Dict[int, List[int]]] = None,
    return_report: bool = False,
):
    """Graph-aware dependency evaluation using a :class:`DependencyKnowledgeGraph`.

    Performs two passes for each asset type:

    1. **Restore pass** (:func:`restore_operational_from_graph`): re-enables any
       asset that is no longer flooded and whose ``return_to_operational`` trigger
       is now satisfied.  This makes the "return to service" logic explicit and
       driven solely by the knowledge graph — not by ad-hoc side-effects elsewhere
       in the simulation loop.

    2. **Block pass**: applies hazard blocking and repair-state blocking according
       to the same rules, so assets that still do not meet conditions are kept or
       returned to non-operational.

    For each unique ``asset_type_a`` present in *asset_type*, the matching rules
    (or the default rule) are fetched from *knowledge_graph* and applied:

    * **Direct rules** (``asset_type_b`` is ``None``):

      1. Restore pass: assets not flooded and meeting their trigger are restored to True.
      2. If ``hazard_blocks_operation`` is ``True``, assets of type A that are
         currently flooded are marked as blocked.
      3. The ``return_to_operational`` trigger is applied to determine whether
         outstanding repair work also blocks the asset.

    * **Service-area rules** (``asset_type_b`` is set):

      Uses *service_area_map* (``{asset_idx_a: [asset_idx_b, ...]}``) to block
      B-type assets when their governing A-type asset is not operational.

    Args:
        operational: Boolean array of current operational states (will not be
            mutated).
        asset_type: String array of asset types corresponding to each index.
        hazard_type: The active hazard type (e.g. ``"flooding"``).
        knowledge_graph: A :class:`~src.dependency_knowledge_graph.DependencyKnowledgeGraph`
            instance.
        flooded_mask: Boolean array; ``True`` where hazard exposure exceeds the
            flood threshold.
        repair_time: Per-asset remaining repair time array.
        service_area_map: Mapping ``{asset_idx_a: [asset_idx_b, ...]}``.  Required
            for service-area rules; ignored otherwise.
        return_report: If ``True``, also return a report dict.

    Returns:
        Updated operational array (and report dict if *return_report* is
        ``True``).
    """
    num_assets = len(asset_type)
    # Restore pass first: re-enable assets whose return-to-operational condition is met.
    operational = restore_operational_from_graph(
        operational,
        asset_type,
        hazard_type,
        knowledge_graph,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
    )

    if flooded_mask is None:
        flooded_mask = np.zeros(num_assets, dtype=bool)
    else:
        flooded_mask = np.asarray(flooded_mask, dtype=bool)

    if repair_time is None:
        repair_time = np.zeros(num_assets, dtype=np.float64)
    else:
        repair_time = np.asarray(repair_time, dtype=np.float64)

    blocked_mask = np.zeros(num_assets, dtype=bool)
    hazard_blocked_count = 0
    repair_blocked_count = 0
    service_area_blocked_count = 0

    unique_asset_types = np.unique(asset_type)

    for a_type in unique_asset_types:
        a_mask = asset_type == a_type

        # --- Direct rules (rule on A itself) --------------------------------
        direct_rules = knowledge_graph.get_rules_or_default(
            hazard_type, a_type, asset_type_b=None
        )
        for rule in direct_rules:
            if rule.relationship != "direct":
                continue

            # 1. Hazard blocking
            if rule.hazard_blocks_operation:
                hazard_blocked = a_mask & flooded_mask
                newly_hazard_blocked = hazard_blocked & ~blocked_mask
                blocked_mask |= hazard_blocked
                hazard_blocked_count += int(np.sum(newly_hazard_blocked))

            # 2. Repair-state blocking
            repair_blocked = _compute_repair_blocked_mask(
                repair_time, a_mask, rule.return_to_operational
            )
            newly_repair_blocked = repair_blocked & ~blocked_mask
            blocked_mask |= repair_blocked
            repair_blocked_count += int(np.sum(newly_repair_blocked))

        # --- Service-area rules (A → B) -------------------------------------
        if service_area_map is None:
            continue

        for b_type in unique_asset_types:
            if b_type == a_type:
                continue
            sa_rules = knowledge_graph.get_rules(hazard_type, a_type, asset_type_b=b_type)
            if not sa_rules:
                continue

            b_mask = asset_type == b_type
            for rule in sa_rules:
                if rule.relationship != "service_area":
                    continue

                # Find A-type assets that are currently non-operational (after
                # direct blocking above has been folded in).
                a_non_operational = a_mask & (blocked_mask | ~operational)

                # Block B-type assets whose governing A asset is non-operational.
                for a_idx in np.where(a_non_operational)[0]:
                    b_indices = service_area_map.get(int(a_idx), [])
                    for b_idx in b_indices:
                        if b_idx < num_assets and b_mask[b_idx]:
                            if not blocked_mask[b_idx]:
                                blocked_mask[b_idx] = True
                                service_area_blocked_count += 1

    updated_operational = apply_dependency_blocking(operational, blocked_mask)

    if not return_report:
        return updated_operational

    report = {
        "blocked_count": int(np.sum(blocked_mask)),
        "hazard_blocked_count": hazard_blocked_count,
        "repair_blocked_count": repair_blocked_count,
        "service_area_blocked_count": service_area_blocked_count,
    }
    return updated_operational, report
