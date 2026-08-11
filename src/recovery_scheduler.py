"""Recovery timing/scheduling helpers.

First-pass scheduler that supports multiple vectorized wait dimensions per
asset (repair/dependency/reset), while remaining backward compatible with the
existing scalar repair-time workflow.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


DEFAULT_WAIT_VECTORS = ("repair_time", "dependency_wait", "reset_wait")


def initialize_recovery_wait_vectors(
    num_assets: int,
    *,
    repair_time: Optional[np.ndarray] = None,
    dependency_wait: Optional[np.ndarray] = None,
    reset_wait: Optional[np.ndarray] = None,
    extra_wait_vectors: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Initialize vectorized wait-state storage for each asset."""
    wait_vectors = {
        "repair_time": np.zeros(num_assets, dtype=np.float64),
        "dependency_wait": np.zeros(num_assets, dtype=np.float64),
        "reset_wait": np.zeros(num_assets, dtype=np.float64),
    }

    if repair_time is not None:
        wait_vectors["repair_time"] = np.asarray(repair_time, dtype=np.float64).copy()
    if dependency_wait is not None:
        wait_vectors["dependency_wait"] = np.asarray(dependency_wait, dtype=np.float64).copy()
    if reset_wait is not None:
        wait_vectors["reset_wait"] = np.asarray(reset_wait, dtype=np.float64).copy()

    if extra_wait_vectors:
        for name, values in extra_wait_vectors.items():
            wait_vectors[name] = np.asarray(values, dtype=np.float64).copy()

    return wait_vectors


def set_wait_vector(wait_vectors: Dict[str, np.ndarray], name: str, values: np.ndarray) -> Dict[str, np.ndarray]:
    """Set or replace one wait vector."""
    wait_vectors[name] = np.asarray(values, dtype=np.float64).copy()
    return wait_vectors


def sync_repair_time_vector(wait_vectors: Dict[str, np.ndarray], repair_time: np.ndarray) -> Dict[str, np.ndarray]:
    """Synchronize the canonical repair-time vector from legacy state arrays."""
    wait_vectors["repair_time"] = np.asarray(repair_time, dtype=np.float64).copy()
    return wait_vectors


def decrement_recovery_wait_vectors(
    wait_vectors: Dict[str, np.ndarray],
    *,
    elapsed_time: float = 1.0,
    active_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Subtract elapsed time from all active wait vectors.

    If an active mask is provided for a vector, only masked assets are decremented.
    Otherwise, all assets with positive waits in that vector are decremented.
    """
    for name, vector in wait_vectors.items():
        arr = np.asarray(vector, dtype=np.float64)
        if active_masks and name in active_masks:
            active = np.asarray(active_masks[name], dtype=bool)
        else:
            active = arr > 0.0

        if np.any(active):
            arr[active] = np.maximum(arr[active] - elapsed_time, 0.0)
        wait_vectors[name] = arr

    return wait_vectors


def get_wait_vector(wait_vectors: Dict[str, np.ndarray], name: str) -> np.ndarray:
    """Get one wait vector by name, creating a zero vector if absent."""
    if name not in wait_vectors:
        if not wait_vectors:
            raise ValueError("wait_vectors must contain at least one vector")
        template = next(iter(wait_vectors.values()))
        wait_vectors[name] = np.zeros_like(template, dtype=np.float64)
    return wait_vectors[name]


def all_required_waits_cleared(
    wait_vectors: Dict[str, np.ndarray],
    *,
    required_vectors: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Return mask where all required wait vectors are cleared (<= 0)."""
    if required_vectors is None:
        required_vectors = DEFAULT_WAIT_VECTORS

    required_vectors = list(required_vectors)
    if not required_vectors:
        if not wait_vectors:
            return np.array([], dtype=bool)
        template = next(iter(wait_vectors.values()))
        return np.ones_like(template, dtype=bool)

    masks = []
    for name in required_vectors:
        masks.append(get_wait_vector(wait_vectors, name) <= 0.0)

    return np.logical_and.reduce(masks)


def build_recovery_report(wait_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Build basic recovery scheduler metrics for reporting/debugging."""
    return {f"{name}_active_count": int(np.sum(vector > 0.0)) for name, vector in wait_vectors.items()}
