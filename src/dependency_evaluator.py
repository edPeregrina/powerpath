"""Dependency evaluation layer for operational-state blocking.

This first-pass module keeps behavior non-breaking while establishing explicit
function boundaries for future dependency rules.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

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
    """High-level dependency evaluation entry point.

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
