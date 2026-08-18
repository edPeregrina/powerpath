"""Actor-by-island redistribution primitives.

All implementation lives in :mod:`src.island_analysis`.  This module exists
solely for backwards compatibility with any code that still imports symbols
directly from ``actor_island_manager``.
"""

from src.island_analysis import (  # noqa: F401  (re-export)
    ActorCountsByIsland,
    ActorDistribution,
    _compute_initial_distribution,
    _compute_transition_probabilities,
    _extract_actor_counts,
    _is_nested_actor_distribution,
    _pack_actor_counts,
    _redistribute_by_transition_probabilities,
    update_actor_islands,
)
