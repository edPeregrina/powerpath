"""Dependency evaluation layer for operational-state blocking.

This module keeps the original flat-rule evaluation as the baseline path and
adds an explicit graph-aware path for opt-in dependency relationships.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

DependencyContext = dict[str, Any]
DependencyReport = dict[str, Any]


def build_dependency_context(
    asset_type: np.ndarray,
    operational: np.ndarray | None = None,
    hazard_values: np.ndarray | None = None,
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    repair_threshold: float = 0.0,
    dependency_map: dict[Any, Iterable[Any]] | None = None,
    area_dependencies: Iterable[dict[str, Any]] | None = None,
    pairwise_dependencies: Iterable[tuple[int, int]] | None = None,
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
    if operational is None:
        operational = np.ones(num_assets, dtype=bool)

    if isinstance(area_dependencies, Mapping):
        normalized_area_dependencies = [
            {
                "supplier_index": supplier_index,
                "dependent_indices": dependent_indices,
            }
            for supplier_index, dependent_indices in area_dependencies.items()
        ]
    else:
        normalized_area_dependencies = (
            list(area_dependencies) if area_dependencies is not None else []
        )

    return {
        "asset_type": np.asarray(asset_type),
        "operational": np.asarray(operational, dtype=bool),
        "hazard_values": np.asarray(hazard_values),
        "flooded_mask": np.asarray(flooded_mask, dtype=bool),
        "repair_time": np.asarray(repair_time, dtype=np.float64),
        "repair_threshold": float(repair_threshold),
        "dependency_map": dependency_map or {},
        "area_dependencies": normalized_area_dependencies,
        "pairwise_dependencies": list(pairwise_dependencies) if pairwise_dependencies is not None else [],
    }


def _dependency_operational_state(context: DependencyContext) -> np.ndarray:
    operational = np.asarray(context["operational"], dtype=bool).copy()
    base_blocked_mask = context.get("base_blocked_mask")
    if base_blocked_mask is not None:
        operational &= ~np.asarray(base_blocked_mask, dtype=bool)
    return operational


def _validate_dependency_index(index: Any, num_assets: int, field_name: str) -> int:
    try:
        normalized = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain integer asset indices") from exc
    if normalized < 0 or normalized >= num_assets:
        raise IndexError(
            f"{field_name} index {normalized} is outside the asset range 0..{num_assets - 1}"
        )
    return normalized


def _area_dependency_pairs(context: DependencyContext) -> list[tuple[int, int]]:
    num_assets = len(context["asset_type"])
    dependencies = list(context.get("area_dependencies", []))
    dependencies.extend(
        {
            "supplier_index": supplier_index,
            "dependent_indices": dependent_indices,
        }
        for supplier_index, dependent_indices in context.get("dependency_map", {}).items()
    )

    pairs: list[tuple[int, int]] = []
    supplier_fields = (
        "supplier_index",
        "source_index",
        "asset_index_a",
        "supplier",
        "source",
    )
    dependent_fields = (
        "dependent_indices",
        "target_indices",
        "asset_indices_b",
        "dependents",
        "targets",
    )
    for dependency in dependencies:
        if isinstance(dependency, (tuple, list)) and len(dependency) == 2:
            supplier, dependents = dependency
        elif isinstance(dependency, Mapping):
            supplier = next(
                (dependency[name] for name in supplier_fields if name in dependency),
                None,
            )
            dependents = next(
                (dependency[name] for name in dependent_fields if name in dependency),
                None,
            )
            if supplier is None or dependents is None:
                raise ValueError(
                    "Area dependencies require a supplier/source index and dependent/target indices"
                )
        else:
            raise TypeError("Area dependencies must be mappings or (supplier, dependents) pairs")

        supplier_index = _validate_dependency_index(
            supplier, num_assets, "area dependency supplier"
        )
        if np.isscalar(dependents):
            dependents = [dependents]
        for dependent in dependents:
            dependent_index = _validate_dependency_index(
                dependent, num_assets, "area dependency dependent"
            )
            if (
                context["asset_type"][supplier_index] != "road"
                and context["asset_type"][dependent_index] != "road"
            ):
                pairs.append((supplier_index, dependent_index))
    return pairs


def _pairwise_dependency_pairs(context: DependencyContext) -> list[tuple[int, int]]:
    num_assets = len(context["asset_type"])
    pairs: list[tuple[int, int]] = []
    for dependency in context.get("pairwise_dependencies", []):
        if isinstance(dependency, Mapping):
            supplier = dependency.get(
                "supplier_index",
                dependency.get("source_index", dependency.get("source")),
            )
            dependent = dependency.get(
                "dependent_index",
                dependency.get("target_index", dependency.get("target")),
            )
        elif isinstance(dependency, (tuple, list)) and len(dependency) == 2:
            supplier, dependent = dependency
        else:
            raise TypeError(
                "Pairwise dependencies must be mappings or (supplier, dependent) pairs"
            )
        if supplier is None or dependent is None:
            raise ValueError(
                "Pairwise dependencies require supplier/source and dependent/target indices"
            )
        supplier_index = _validate_dependency_index(
            supplier, num_assets, "pairwise dependency supplier"
        )
        dependent_index = _validate_dependency_index(
            dependent, num_assets, "pairwise dependency dependent"
        )
        if (
            context["asset_type"][supplier_index] != "road"
            and context["asset_type"][dependent_index] != "road"
        ):
            pairs.append((supplier_index, dependent_index))
    return pairs


def _evaluate_dependency_pairs(
    operational: np.ndarray,
    pairs: Iterable[tuple[int, int]],
) -> np.ndarray:
    blocked = np.zeros(len(operational), dtype=bool)
    pairs = list(pairs)
    changed = True
    while changed:
        changed = False
        for supplier_index, dependent_index in pairs:
            if (
                (not operational[supplier_index] or blocked[supplier_index])
                and not blocked[dependent_index]
            ):
                blocked[dependent_index] = True
                changed = True
    return blocked


def _evaluate_combined_dependency_pairs(
    operational: np.ndarray,
    area_pairs: Iterable[tuple[int, int]],
    pairwise_pairs: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    area_blocked = np.zeros(len(operational), dtype=bool)
    pairwise_blocked = np.zeros(len(operational), dtype=bool)
    tagged_pairs = [
        *((supplier, dependent, area_blocked) for supplier, dependent in area_pairs),
        *(
            (supplier, dependent, pairwise_blocked)
            for supplier, dependent in pairwise_pairs
        ),
    ]
    combined_blocked = np.zeros(len(operational), dtype=bool)
    changed = True
    while changed:
        changed = False
        for supplier_index, dependent_index, relationship_mask in tagged_pairs:
            if (
                (
                    not operational[supplier_index]
                    or combined_blocked[supplier_index]
                )
                and not combined_blocked[dependent_index]
            ):
                combined_blocked[dependent_index] = True
                relationship_mask[dependent_index] = True
                changed = True
    return area_blocked, pairwise_blocked


def evaluate_area_dependencies(context: DependencyContext) -> np.ndarray:
    """Block assets supplied by a non-operational service-area supplier."""
    return _evaluate_dependency_pairs(
        _dependency_operational_state(context),
        _area_dependency_pairs(context),
    )


def evaluate_pairwise_dependencies(context: DependencyContext) -> np.ndarray:
    """Block pairwise dependents when their supplier is non-operational."""
    return _evaluate_dependency_pairs(
        _dependency_operational_state(context),
        _pairwise_dependency_pairs(context),
    )


def evaluate_dependency_rules(
    context: DependencyContext,
    *,
    area_blocked_mask: np.ndarray | None = None,
    pairwise_blocked_mask: np.ndarray | None = None,
    enable_default_rules: bool = True,
    require_repair_for_operational: bool = False,
) -> np.ndarray:
    """Combine dependency rules into one blocking mask.

    Road availability is intentionally handled by the simulation's distinct
    exposure-only path and is never changed here.
    """
    num_assets = len(context["asset_type"])
    blocked_mask = np.zeros(num_assets, dtype=bool)

    if area_blocked_mask is not None:
        blocked_mask |= np.asarray(area_blocked_mask, dtype=bool)
    if pairwise_blocked_mask is not None:
        blocked_mask |= np.asarray(pairwise_blocked_mask, dtype=bool)

    if require_repair_for_operational:
        repair_time = context["repair_time"]
        repair_threshold = context["repair_threshold"]
        blocked_mask |= (
            (repair_time > repair_threshold)
            & (context["asset_type"] != "road")
        )

    return blocked_mask


def apply_dependency_blocking(operational: np.ndarray, blocked_mask: np.ndarray) -> np.ndarray:
    """Apply dependency blocking to the operational state."""
    return np.asarray(operational, dtype=bool) & ~np.asarray(blocked_mask, dtype=bool)


def build_dependency_report(
    *,
    blocked_mask: np.ndarray,
    area_blocked_mask: np.ndarray,
    pairwise_blocked_mask: np.ndarray,
    rules_enabled: bool,
    require_repair_for_operational: bool,
    dependency_blocked_mask: np.ndarray | None = None,
    warning: str | None = None,
) -> DependencyReport:
    """Build a concise report suitable for logging/debugging."""
    report = {
        "blocked_count": int(np.sum(blocked_mask)),
        "area_blocked_count": int(np.sum(area_blocked_mask)),
        "pairwise_blocked_count": int(np.sum(pairwise_blocked_mask)),
        "dependency_blocked_mask": (
            np.asarray(dependency_blocked_mask, dtype=bool).copy()
            if dependency_blocked_mask is not None
            else np.zeros(len(blocked_mask), dtype=bool)
        ),
        "rules_enabled": bool(rules_enabled),
        "require_repair_for_operational": bool(require_repair_for_operational),
        "active_rules": [
            "repair_time:threshold" if require_repair_for_operational else None,
        ],
    }
    report["active_rules"] = [rule for rule in report["active_rules"] if rule is not None]
    if warning is not None:
        report["warning"] = warning
    return report


def evaluate_dependencies(
    operational: np.ndarray,
    asset_type: np.ndarray,
    *,
    hazard_values: np.ndarray | None = None,
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    repair_threshold: float = 0.0,
    dependency_map: dict[Any, Iterable[Any]] | None = None,
    area_dependencies: Iterable[dict[str, Any]] | None = None,
    pairwise_dependencies: Iterable[tuple[int, int]] | None = None,
    previous_dependency_blocked_mask: np.ndarray | None = None,
    enable_default_rules: bool = True,
    require_repair_for_operational: bool = False,
    return_report: bool = False,
):
    """High-level dependency evaluation entry point (legacy path).

    Returns updated operational state, plus report when requested.
    """
    context = build_dependency_context(
        asset_type,
        operational=operational,
        hazard_values=hazard_values,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
        repair_threshold=repair_threshold,
        dependency_map=dependency_map,
        area_dependencies=area_dependencies,
        pairwise_dependencies=pairwise_dependencies,
    )
    preserved_dependency_blocked_mask = np.zeros(
        len(context["operational"]), dtype=bool
    )
    if previous_dependency_blocked_mask is not None:
        previous_dependency_blocked_mask = np.asarray(
            previous_dependency_blocked_mask, dtype=bool
        ).copy()
        if previous_dependency_blocked_mask.shape != context["operational"].shape:
            raise ValueError(
                "previous_dependency_blocked_mask must match the operational array"
            )
        previous_dependency_blocked_mask &= context["asset_type"] != "road"
        restorable_dependency_outages = (
            previous_dependency_blocked_mask
            & ~context["flooded_mask"]
            & (context["repair_time"] <= context["repair_threshold"])
        )
        context["operational"][restorable_dependency_outages] = True
        preserved_dependency_blocked_mask = (
            previous_dependency_blocked_mask
            & ~restorable_dependency_outages
        )

    base_blocked_mask = evaluate_dependency_rules(
        context,
        enable_default_rules=enable_default_rules,
        require_repair_for_operational=require_repair_for_operational,
    )
    context["base_blocked_mask"] = base_blocked_mask
    area_blocked_mask, pairwise_blocked_mask = (
        _evaluate_combined_dependency_pairs(
            _dependency_operational_state(context),
            _area_dependency_pairs(context),
            _pairwise_dependency_pairs(context),
        )
    )
    blocked_mask = evaluate_dependency_rules(
        context,
        area_blocked_mask=area_blocked_mask,
        pairwise_blocked_mask=pairwise_blocked_mask,
        enable_default_rules=enable_default_rules,
        require_repair_for_operational=require_repair_for_operational,
    )
    updated_operational = apply_dependency_blocking(
        context["operational"], blocked_mask
    )
    dependency_blocked_mask = (
        preserved_dependency_blocked_mask
        | (
            (area_blocked_mask | pairwise_blocked_mask)
            & context["operational"]
        )
    )

    if not return_report:
        return updated_operational

    warning = None
    if not enable_default_rules:
        warning = (
            "Default dependency rules are disabled; only explicitly configured "
            "area and pairwise dependencies are being applied."
        )

    report = build_dependency_report(
        blocked_mask=blocked_mask,
        area_blocked_mask=area_blocked_mask,
        pairwise_blocked_mask=pairwise_blocked_mask,
        rules_enabled=enable_default_rules,
        require_repair_for_operational=require_repair_for_operational,
        dependency_blocked_mask=dependency_blocked_mask,
        warning=warning,
    )
    return updated_operational, report


# ---------------------------------------------------------------------------
# Graph-aware evaluation path
# ---------------------------------------------------------------------------
def activate_delayed_trigger_waits(
    operational: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: np.ndarray,
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
) -> None:
    """Start direct delayed-trigger countdowns when an asset becomes affected."""
    from src.dependency_knowledge_graph import TRIGGER_DELAYED

    operational = np.asarray(operational, dtype=bool)
    flooded_mask = np.asarray(flooded_mask, dtype=bool)
    num_assets = len(asset_type)

    for a_type in np.unique(asset_type):
        if a_type == "road":
            continue
        asset_mask = asset_type == a_type
        for rule in knowledge_graph.get_rules(
            hazard_type, a_type, asset_type_b=None
        ):
            if (
                rule.relationship != "direct"
                or rule.return_to_operational.trigger != TRIGGER_DELAYED
            ):
                continue

            wait_vector_name = rule.return_to_operational.wait_vector
            wait_vector = wait_vectors.setdefault(
                wait_vector_name, np.zeros(num_assets, dtype=np.float64)
            )
            active = active_masks.setdefault(
                wait_vector_name, np.zeros(num_assets, dtype=bool)
            )
            affected = asset_mask & ~operational
            if rule.hazard_blocks_operation:
                affected |= asset_mask & flooded_mask
            newly_affected = affected & ~active
            if np.any(newly_affected):
                wait_vector[newly_affected] = np.maximum(
                    wait_vector[newly_affected],
                    rule.return_to_operational.delay_steps,
                )
                active[newly_affected] = True


def clear_completed_delayed_triggers(
    operational: np.ndarray,
    wait_vectors: dict[str, np.ndarray],
    active_masks: dict[str, np.ndarray],
) -> None:
    """Allow a later disruption to start a fresh delayed-trigger countdown."""
    operational = np.asarray(operational, dtype=bool)
    for wait_vector_name, active in active_masks.items():
        wait_vector = wait_vectors.get(wait_vector_name)
        if wait_vector is None:
            continue
        active[operational & (wait_vector <= 0.0)] = False


def _compute_repair_blocked_mask(
    repair_time: np.ndarray,
    asset_mask: np.ndarray,
    return_to_operational,
    wait_vectors: dict[str, np.ndarray] | None = None,
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
        TRIGGER_DELAYED,
        TRIGGER_IMMEDIATE,
        TRIGGER_REPAIR_BELOW,
        TRIGGER_REPAIR_COMPLETE,
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
        blocked[asset_mask] = repair_time[asset_mask] > threshold
    elif trigger == TRIGGER_DELAYED:
        if wait_vectors is None:
            blocked[asset_mask] = True
        else:
            wait_vector = np.asarray(
                wait_vectors.get(
                    return_to_operational.wait_vector,
                    np.zeros(len(repair_time), dtype=np.float64),
                ),
                dtype=np.float64,
            )
            blocked[asset_mask] = wait_vector[asset_mask] > 0.0

    return blocked


def _compute_restore_eligible_mask(
    repair_time: np.ndarray,
    flooded_mask: np.ndarray,
    asset_mask: np.ndarray,
    return_to_operational,
    wait_vectors: dict[str, np.ndarray] | None = None,
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
        TRIGGER_DELAYED,
        TRIGGER_IMMEDIATE,
        TRIGGER_REPAIR_BELOW,
        TRIGGER_REPAIR_COMPLETE,
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
        eligible[not_flooded] = repair_time[not_flooded] <= threshold
    elif trigger == TRIGGER_DELAYED and wait_vectors is not None:
        wait_vector = np.asarray(
            wait_vectors.get(
                return_to_operational.wait_vector,
                np.zeros(len(repair_time), dtype=np.float64),
            ),
            dtype=np.float64,
        )
        eligible[not_flooded] = wait_vector[not_flooded] <= 0.0

    return eligible


def restore_operational_from_graph(
    operational: np.ndarray,
    asset_type: np.ndarray,
    hazard_type: str,
    knowledge_graph,
    *,
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    wait_vectors: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Restore assets to operational based on knowledge graph return-to-operational triggers.

    This is the positive-restore counterpart to :func:`evaluate_dependencies_from_graph`.
    It re-enables assets that were previously marked non-operational and now satisfy
    both conditions:

    1. The asset is no longer flooded.
    2. The ``return_to_operational`` trigger for its asset type is satisfied
       (e.g. ``repair_complete``: repair_time == 0; ``repair_below``: repair_time
       < threshold; ``immediate``: always satisfied once hazard clears).

    This function only enables assets — it never disables them.  It is therefore
    always paired with the block pass in :func:`evaluate_dependencies_from_graph`:
    assets move up only when the trigger is satisfied, then the block pass can
    immediately re-suppress any asset whose rule conditions are still not met.

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
        if a_type == "road":
            continue
        a_mask = asset_type == a_type
        direct_rules = knowledge_graph.get_rules(
            hazard_type, a_type, asset_type_b=None
        )
        for rule in direct_rules:
            if rule.relationship != "direct":
                continue
            eligible = _compute_restore_eligible_mask(
                repair_time,
                flooded_mask,
                a_mask,
                rule.return_to_operational,
                wait_vectors=wait_vectors,
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
    flooded_mask: np.ndarray | None = None,
    repair_time: np.ndarray | None = None,
    wait_vectors: dict[str, np.ndarray] | None = None,
    service_area_map: dict[int, list[int]] | None = None,
    previous_dependency_blocked_mask: np.ndarray | None = None,
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
    if flooded_mask is None:
        flooded_mask = np.zeros(num_assets, dtype=bool)
    else:
        flooded_mask = np.asarray(flooded_mask, dtype=bool)

    if repair_time is None:
        repair_time = np.zeros(num_assets, dtype=np.float64)
    else:
        repair_time = np.asarray(repair_time, dtype=np.float64)

    operational = np.asarray(operational, dtype=bool).copy()
    preserved_dependency_blocked_mask = np.zeros(num_assets, dtype=bool)
    if previous_dependency_blocked_mask is not None:
        previous_dependency_blocked_mask = np.asarray(
            previous_dependency_blocked_mask, dtype=bool
        ).copy()
        if previous_dependency_blocked_mask.shape != operational.shape:
            raise ValueError(
                "previous_dependency_blocked_mask must match the operational array"
            )
        previous_dependency_blocked_mask &= asset_type != "road"
        restorable_dependency_outages = (
            previous_dependency_blocked_mask
            & ~flooded_mask
            & (repair_time <= 0.0)
        )
        operational[restorable_dependency_outages] = True
        preserved_dependency_blocked_mask = (
            previous_dependency_blocked_mask
            & ~restorable_dependency_outages
        )

    # Restore pass first: re-enable assets whose return-to-operational condition is met.
    operational = restore_operational_from_graph(
        operational,
        asset_type,
        hazard_type,
        knowledge_graph,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
        wait_vectors=wait_vectors,
    )
    dependency_candidate_operational = operational.copy()

    blocked_mask = np.zeros(num_assets, dtype=bool)
    hazard_blocked_count = 0
    repair_blocked_count = 0
    service_area_blocked_count = 0
    active_rules: list[dict[str, Any]] = []

    unique_asset_types = np.unique(asset_type)

    for a_type in unique_asset_types:
        if a_type == "road":
            continue
        a_mask = asset_type == a_type

        # --- Direct rules (rule on A itself) --------------------------------
        direct_rules = knowledge_graph.get_rules(
            hazard_type, a_type, asset_type_b=None
        )
        for rule in direct_rules:
            if rule.relationship != "direct":
                continue
            active_rules.append(
                {
                    "hazard_type": hazard_type,
                    "asset_type_a": a_type,
                    "asset_type_b": None,
                    "relationship": "direct",
                    "hazard_blocks_operation": bool(rule.hazard_blocks_operation),
                    "return_trigger": rule.return_to_operational.trigger,
                }
            )

            # 1. Hazard blocking
            if rule.hazard_blocks_operation:
                hazard_blocked = a_mask & flooded_mask
                newly_hazard_blocked = hazard_blocked & ~blocked_mask
                blocked_mask |= hazard_blocked
                hazard_blocked_count += int(np.sum(newly_hazard_blocked))

            # 2. Repair-state blocking
            repair_blocked = _compute_repair_blocked_mask(
                repair_time,
                a_mask,
                rule.return_to_operational,
                wait_vectors=wait_vectors,
            )
            newly_repair_blocked = repair_blocked & ~blocked_mask
            blocked_mask |= repair_blocked
            repair_blocked_count += int(np.sum(newly_repair_blocked))

    # --- Service-area rules (A → B), propagated to a fixed point ------------
    service_area_blocked_mask = np.zeros(num_assets, dtype=bool)
    service_edges: list[tuple[int, int]] = []
    if service_area_map is not None:
        for rule in knowledge_graph.rules:
            if (
                rule.hazard_type != hazard_type
                or rule.relationship != "service_area"
                or rule.asset_type_b is None
                or rule.asset_type_a == "road"
                or rule.asset_type_b == "road"
            ):
                continue
            active_rules.append(
                {
                    "hazard_type": hazard_type,
                    "asset_type_a": rule.asset_type_a,
                    "asset_type_b": rule.asset_type_b,
                    "relationship": "service_area",
                    "hazard_blocks_operation": bool(
                        rule.hazard_blocks_operation
                    ),
                    "return_trigger": rule.return_to_operational.trigger,
                }
            )
            source_indices = np.where(asset_type == rule.asset_type_a)[0]
            for source_index in source_indices:
                for dependent_index in service_area_map.get(
                    int(source_index), []
                ):
                    if (
                        0 <= dependent_index < num_assets
                        and asset_type[dependent_index] == rule.asset_type_b
                    ):
                        service_edges.append(
                            (int(source_index), int(dependent_index))
                        )

    changed = True
    while changed:
        changed = False
        for source_index, dependent_index in service_edges:
            source_unavailable = (
                not operational[source_index]
                or blocked_mask[source_index]
                or service_area_blocked_mask[source_index]
            )
            if (
                source_unavailable
                and not service_area_blocked_mask[dependent_index]
            ):
                service_area_blocked_mask[dependent_index] = True
                changed = True

    service_area_blocked_count = int(
        np.sum(service_area_blocked_mask & ~blocked_mask)
    )
    blocked_mask |= service_area_blocked_mask

    updated_operational = apply_dependency_blocking(operational, blocked_mask)
    dependency_blocked_mask = (
        preserved_dependency_blocked_mask
        | (service_area_blocked_mask & dependency_candidate_operational)
    )

    if not return_report:
        return updated_operational

    report = {
        "blocked_count": int(np.sum(blocked_mask)),
        "hazard_blocked_count": hazard_blocked_count,
        "repair_blocked_count": repair_blocked_count,
        "service_area_blocked_count": service_area_blocked_count,
        "dependency_blocked_mask": dependency_blocked_mask,
        "active_rule_count": len(active_rules),
        "active_rules": active_rules,
    }
    return updated_operational, report
