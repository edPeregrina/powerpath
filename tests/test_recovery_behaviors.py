import networkx as nx
import numpy as np

from src.simulation import (
    SimulationState,
    _assign_repair_crews,
    _handle_completed_repairs,
    _normalize_repair_crews_by_asset_type_config,
)
from src.utils import filter_hazard_graph


def test_grouped_crew_pool_cannot_cross_islands():
    pool_state = _normalize_repair_crews_by_asset_type_config({"hospital": 1})
    pool_state["pools"][0]["available"] = {0: 1}
    assigned = np.zeros(2, dtype=bool)

    _, assigned = _assign_repair_crews(
        timestep=0,
        available_repair_crews={0: 0, 1: 0},
        repair_crews_assigned=assigned,
        accessible=np.ones(2, dtype=bool),
        flooded_mask=np.zeros(2, dtype=bool),
        repair_time=np.array([0.0, 3.0]),
        island_ids=np.array([0, 1]),
        method="islands",
        verbose=False,
        asset_type=np.array(["hospital", "hospital"]),
        repair_crews_by_asset_type=pool_state,
    )

    assert assigned.tolist() == [False, False]
    assert pool_state["pools"][0]["available"] == {0: 1}


def test_grouped_crew_returns_to_repaired_assets_current_island():
    pool_state = _normalize_repair_crews_by_asset_type_config({"hospital": 1})
    pool_state["pools"][0]["available"] = {0: 0}
    state = SimulationState(None, 1)
    state.island_ids[0] = 3
    state.repair_crews_assigned[0] = True

    _handle_completed_repairs(
        state,
        available_repair_crews={0: 0},
        verbose=False,
        timestep=1,
        asset_type=np.array(["hospital"]),
        repair_crews_by_asset_type=pool_state,
    )

    assert pool_state["pools"][0]["available"] == {0: 0, 3: 1}
    assert not state.repair_crews_assigned[0]


def test_completed_repair_releases_crew_while_dependency_wait_continues():
    state = SimulationState(None, 1)
    state.repair_crews_assigned[0] = True
    state.recovery_wait_vectors["dependency_wait"][0] = 3.0

    available = _handle_completed_repairs(
        state,
        available_repair_crews=0,
        verbose=False,
        timestep=1,
    )

    assert available == 1
    assert not state.repair_crews_assigned[0]
    assert state.recovery_wait_vectors["dependency_wait"][0] == 3.0


def test_hazard_graph_filtering_always_starts_from_baseline_graph():
    graph = nx.Graph()
    graph.add_edge(0, 1, EV0_ma=1.0, EV1_ma=0.0)
    graph.add_edge(1, 2, EV0_ma=0.0, EV1_ma=0.0)

    flooded_graph = filter_hazard_graph(graph, 0.2, "EV0_ma")
    clear_graph = filter_hazard_graph(graph, 0.2, "EV1_ma")

    assert not flooded_graph.has_edge(0, 1)
    assert clear_graph.has_edge(0, 1)
    assert graph.has_edge(0, 1)
