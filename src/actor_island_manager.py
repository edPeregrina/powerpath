"""Actor-by-island redistribution primitives.

The public repair-crew entry point lives in :mod:`src.island_analysis`.
This module retains the reusable probabilistic allocation primitives and a
compatibility wrapper for any code that still imports
``update_actor_islands`` from here.
"""

from __future__ import annotations

import numpy as np

ActorCountsByIsland = dict[int, int]
ActorDistribution = dict[str, ActorCountsByIsland]


def _is_nested_actor_distribution(value) -> bool:
    return isinstance(value, dict) and all(isinstance(v, dict) for v in value.values())


def _extract_actor_counts(
    available_actors,
    actor_type: str,
) -> tuple[ActorCountsByIsland | int, str, ActorDistribution]:
    """Extract a single actor type from flat/nested compatibility inputs."""
    if isinstance(available_actors, int):
        return available_actors, "int", {actor_type: {}}

    if not isinstance(available_actors, dict):
        raise TypeError("available_actors must be int, dict[island->count], or dict[type->dict]")

    if _is_nested_actor_distribution(available_actors):
        nested_distribution = {k: dict(v) for k, v in available_actors.items()}
        actor_counts = dict(nested_distribution.get(actor_type, {}))
        return actor_counts, "nested", nested_distribution

    flat_distribution = {int(k): int(v) for k, v in available_actors.items()}
    return flat_distribution, "flat", {actor_type: flat_distribution.copy()}


def _pack_actor_counts(
    *,
    updated_actor_counts: ActorCountsByIsland,
    input_shape: str,
    nested_distribution: ActorDistribution,
    actor_type: str,
):
    """Pack updated actor counts back to the requested compatibility shape."""
    if input_shape == "nested":
        nested_distribution[actor_type] = updated_actor_counts
        return nested_distribution

    # For int or flat legacy inputs, keep legacy output shape.
    return updated_actor_counts


def _compute_initial_distribution(
    total_actor_count: int,
    current_rfids_islands: dict[int, int],
    rfids_lengths: dict[int, float],
    *,
    skip_unassigned_islands: bool = False,
):
    curr_island_lengths = {}
    for rfid, island_id in current_rfids_islands.items():
        if skip_unassigned_islands and island_id == -1:
            continue
        curr_island_lengths.setdefault(island_id, 0.0)
        curr_island_lengths[island_id] += rfids_lengths.get(rfid, 0.0)

    unique_islands = list(curr_island_lengths.keys())
    if not unique_islands:
        unique_islands = [0]
        probabilities = np.array([1.0], dtype=np.float64)
    else:
        lengths = np.array([curr_island_lengths[i] for i in unique_islands], dtype=np.float64)
        total_length = lengths.sum()
        probabilities = lengths / total_length if total_length > 0 else np.ones_like(lengths) / len(lengths)

    assigned = np.random.choice(unique_islands, size=total_actor_count, p=probabilities, replace=True)
    actor_counts = dict(zip(*np.unique(assigned, return_counts=True))) if total_actor_count > 0 else {}
    return {island: int(actor_counts.get(island, 0)) for island in unique_islands}, dict(zip(unique_islands, probabilities))


def _compute_transition_probabilities(
    previous_rfids_islands: dict[int, int],
    current_rfids_islands: dict[int, int],
    rfids_lengths: dict[int, float],
):
    transition_probabilities = {}

    for prev_island in set(previous_rfids_islands.values()):
        if prev_island == -1:
            continue

        rfids_in_prev = [rfid for rfid, island in previous_rfids_islands.items() if island == prev_island]
        curr_lengths = {}
        for rfid in rfids_in_prev:
            curr_island = current_rfids_islands.get(rfid, None)
            if curr_island is None or curr_island == -1:
                continue
            curr_lengths.setdefault(curr_island, 0.0)
            curr_lengths[curr_island] += rfids_lengths.get(rfid, 0.0)

        total_length = sum(curr_lengths.values())
        if total_length > 0:
            transition_probabilities[prev_island] = {
                curr_island: length / total_length
                for curr_island, length in curr_lengths.items()
            }

    return transition_probabilities


def _redistribute_by_transition_probabilities(
    actor_counts: ActorCountsByIsland,
    transition_probabilities,
    *,
    verbose: bool = False,
):
    redistributed = {}

    for prev_island, count in actor_counts.items():
        if prev_island == -1:
            if verbose:
                print(f"WARNING: Skipping actor redistribution from island_id = -1 ({count} actors lost)")
            continue

        if transition_probabilities.get(prev_island):
            island_transitions = transition_probabilities[prev_island]
            curr_islands = list(island_transitions.keys())
            probabilities = [island_transitions[i] for i in curr_islands]

            if verbose:
                print(f"Probability distribution from/to island {prev_island}: {dict(zip(curr_islands, probabilities))}")

            assigned = np.random.choice(curr_islands, size=count, p=probabilities, replace=True)
            assigned_counts = dict(zip(*np.unique(assigned, return_counts=True))) if count > 0 else {}
            for island in curr_islands:
                redistributed[island] = redistributed.get(island, 0) + int(assigned_counts.get(island, 0))
        else:
            redistributed[prev_island] = redistributed.get(prev_island, 0) + int(count)

    return redistributed


def update_actor_islands(*args, **kwargs):
    """Compatibility wrapper that defers to :mod:`src.island_analysis`."""
    from src.island_analysis import update_actor_islands as _update_actor_islands

    return _update_actor_islands(*args, **kwargs)
