"""Generic actor-by-island redistribution helpers.

This module generalizes island redistribution so repair crews are only one actor
category among many.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.caching import create_overlap_cache_key, save_overlap_cache


ActorCountsByIsland = Dict[int, int]
ActorDistribution = Dict[str, ActorCountsByIsland]


def _is_nested_actor_distribution(value) -> bool:
    return isinstance(value, dict) and all(isinstance(v, dict) for v in value.values())


def _extract_actor_counts(
    available_actors,
    actor_type: str,
) -> Tuple[ActorCountsByIsland | int, str, ActorDistribution]:
    """Extract a single actor type from flat/nested compatibility inputs."""
    if isinstance(available_actors, int):
        return available_actors, "int", {actor_type: {}}

    if not isinstance(available_actors, dict):
        raise ValueError("available_actors must be int, dict[island->count], or dict[type->dict]")

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
    current_rfids_islands: Dict[int, int],
    rfids_lengths: Dict[int, float],
):
    curr_island_lengths = {}
    for rfid, island_id in current_rfids_islands.items():
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
    previous_rfids_islands: Dict[int, int],
    current_rfids_islands: Dict[int, int],
    rfids_lengths: Dict[int, float],
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

        if prev_island in transition_probabilities and transition_probabilities[prev_island]:
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


def update_actor_islands(
    available_actors,
    previous_rfids_islands,
    current_rfids_islands,
    rfids_lengths,
    *,
    actor_type: str = "repair_crews",
    verbose: bool = False,
    overlap_cache=None,
    current_map=None,
    previous_map=None,
    hazard_threshold=None,
    hazard_dir=None,
    _config=None,
    cache_updated=None,
    l1_area_geojson=None,
    l1_active_timesteps=None,
):
    """Generic island redistribution entry point.

    Supports compatibility inputs:
    - int (legacy first-round total actor count)
    - dict[island_id -> count] (legacy actor counts)
    - dict[actor_type -> dict[island_id -> count]] (new nested structure)
    """
    actor_counts, input_shape, nested_distribution = _extract_actor_counts(available_actors, actor_type)

    # First round: actor count as scalar total.
    if isinstance(actor_counts, int):
        initial_probabilities = None
        overlap_cache_key = None

        if overlap_cache is not None and current_map is not None and hazard_threshold is not None:
            overlap_cache_key = create_overlap_cache_key(
                "initial",
                current_map,
                hazard_threshold,
                hazard_dir,
                l1_area_geojson=l1_area_geojson,
            )
            if overlap_cache_key in overlap_cache:
                if verbose:
                    print(f"Using cached initial distribution for {overlap_cache_key}")
                initial_probabilities = overlap_cache[overlap_cache_key]

        if initial_probabilities is not None:
            unique_islands = list(initial_probabilities.keys())
            probabilities = np.array([initial_probabilities[i] for i in unique_islands], dtype=np.float64)
            assigned = np.random.choice(unique_islands, size=actor_counts, p=probabilities, replace=True)
            sampled_counts = dict(zip(*np.unique(assigned, return_counts=True))) if actor_counts > 0 else {}
            redistributed_counts = {i: int(sampled_counts.get(i, 0)) for i in unique_islands}
        else:
            redistributed_counts, computed_probabilities = _compute_initial_distribution(
                actor_counts,
                current_rfids_islands,
                rfids_lengths,
            )
            if overlap_cache is not None and overlap_cache_key is not None:
                overlap_cache[overlap_cache_key] = computed_probabilities
                if _config is not None:
                    cache_dir = _config["interim_dir"]
                    save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                    if verbose:
                        print(f"Cached initial distribution for {overlap_cache_key}")
                if cache_updated is not None:
                    cache_updated["overlap_cache"] = overlap_cache

        if verbose:
            print(f"Initial {actor_type} distribution: {redistributed_counts}")

        return _pack_actor_counts(
            updated_actor_counts=redistributed_counts,
            input_shape=input_shape,
            nested_distribution=nested_distribution,
            actor_type=actor_type,
        )

    if previous_rfids_islands is None:
        if verbose:
            print(f"First timestep with dict {actor_type}: redistributing from aggregate count")
        total_count = int(sum(actor_counts.values()))
        redistributed_counts, _ = _compute_initial_distribution(total_count, current_rfids_islands, rfids_lengths)
        if verbose:
            print(f"Initial {actor_type} distribution (from dict): {redistributed_counts}")
        return _pack_actor_counts(
            updated_actor_counts=redistributed_counts,
            input_shape=input_shape,
            nested_distribution=nested_distribution,
            actor_type=actor_type,
        )

    transition_probabilities = None
    overlap_cache_key = None

    if (
        overlap_cache is not None
        and current_map is not None
        and previous_map is not None
        and hazard_threshold is not None
    ):
        overlap_cache_key = create_overlap_cache_key(
            previous_map,
            current_map,
            hazard_threshold,
            hazard_dir,
            l1_area_geojson=l1_area_geojson,
            l1_active_timesteps=l1_active_timesteps,
        )

        if overlap_cache_key in overlap_cache:
            if verbose:
                print(f"Using cached transition probabilities for {overlap_cache_key}")
            transition_probabilities = overlap_cache[overlap_cache_key]

    if transition_probabilities is None:
        if verbose:
            print("Computing transition probabilities (cache miss)")
        transition_probabilities = _compute_transition_probabilities(
            previous_rfids_islands,
            current_rfids_islands,
            rfids_lengths,
        )

        if overlap_cache is not None and overlap_cache_key is not None:
            overlap_cache[overlap_cache_key] = transition_probabilities
            if _config is not None:
                cache_dir = _config["interim_dir"]
                save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                if verbose:
                    print(f"Cached transition probabilities for {overlap_cache_key}")
            if cache_updated is not None:
                cache_updated["overlap_cache"] = overlap_cache

    redistributed_counts = _redistribute_by_transition_probabilities(
        actor_counts,
        transition_probabilities,
        verbose=verbose,
    )

    if verbose:
        print(f"Redistributed {actor_type} distribution: {redistributed_counts}")

    return _pack_actor_counts(
        updated_actor_counts=redistributed_counts,
        input_shape=input_shape,
        nested_distribution=nested_distribution,
        actor_type=actor_type,
    )
