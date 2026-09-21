"""Focused tests for policy-aware dependency evaluation and evaluator integration.

Covers the new type-level knowledge graph (:mod:`src.dependency_knowledge_graph`),
runtime topology expansion (:mod:`src.dependency_topology`), and policy-aware
evaluation (:mod:`src.dependency_evaluator`) -- replacing the legacy flat
dependency schema this module previously tested.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependency_evaluator import (
    activate_dependency_restart_waits,
    compute_restart_ready,
    evaluate_dependency_availability,
    evaluate_edge_availability,
    evaluate_operational_state,
    hazard_recovery_wait_key,
    restart_wait_key,
)
from src.dependency_knowledge_graph import (
    DependencyKnowledgeGraph,
    POLICY_ANY,
    POLICY_AT_LEAST_N,
    POLICY_EXCLUSIVE,
)
from src.dependency_topology import DependencyEdge
from src.simulation import SimulationState, _update_operational_state


def _edge(
    edge_key,
    target_index,
    provider_indices,
    availability_policy=POLICY_ANY,
    minimum_available=None,
    target_type="hospital",
    source_type="msls",
):
    return DependencyEdge(
        edge_key=edge_key,
        target_index=target_index,
        target_type=target_type,
        source_type=source_type,
        relation="dependency",
        topology="direct",
        availability_policy=availability_policy,
        minimum_available=minimum_available,
        provider_indices=tuple(sorted(provider_indices)),
    )


# ---------------------------------------------------------------------------
# exclusive policy
# ---------------------------------------------------------------------------
def test_exclusive_provider_loss_blocks_target():
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    baseline = np.array([False, True])  # provider 0 lost
    dependency_available = np.ones(2, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_exclusive_available_when_sole_provider_available():
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    baseline = np.array([True, True])
    dependency_available = np.ones(2, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is True


def test_exclusive_zero_providers_is_unavailable():
    edge = _edge("e1", target_index=0, provider_indices=(), availability_policy=POLICY_EXCLUSIVE)
    baseline = np.array([True])
    dependency_available = np.ones(1, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_exclusive_does_not_fall_back_to_second_provider():
    """An exclusive edge unexpectedly carrying >1 providers must fail clearly,
    never silently pick a fallback provider."""
    edge = _edge("e1", target_index=2, provider_indices=(0, 1), availability_policy=POLICY_EXCLUSIVE)
    baseline = np.array([False, True, True])  # provider 0 down, provider 1 up
    dependency_available = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="[Ee]xclusive"):
        evaluate_edge_availability(edge, baseline, dependency_available)


# ---------------------------------------------------------------------------
# any policy
# ---------------------------------------------------------------------------
def test_any_survives_one_provider_loss():
    edge = _edge("e1", target_index=2, provider_indices=(0, 1), availability_policy=POLICY_ANY)
    baseline = np.array([False, True, True])
    dependency_available = np.ones(3, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is True


def test_any_fails_when_all_providers_unavailable():
    edge = _edge("e1", target_index=2, provider_indices=(0, 1), availability_policy=POLICY_ANY)
    baseline = np.array([False, False, True])
    dependency_available = np.ones(3, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_any_zero_providers_is_unavailable():
    edge = _edge("e1", target_index=0, provider_indices=(), availability_policy=POLICY_ANY)
    baseline = np.array([True])
    dependency_available = np.ones(1, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


# ---------------------------------------------------------------------------
# at_least_n policy
# ---------------------------------------------------------------------------
def test_at_least_n_available_when_quorum_met():
    edge = _edge(
        "e1", target_index=3, provider_indices=(0, 1, 2),
        availability_policy=POLICY_AT_LEAST_N, minimum_available=2,
    )
    baseline = np.array([True, True, False, True])
    dependency_available = np.ones(4, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is True


def test_at_least_n_fails_when_quorum_lost():
    edge = _edge(
        "e1", target_index=3, provider_indices=(0, 1, 2),
        availability_policy=POLICY_AT_LEAST_N, minimum_available=2,
    )
    baseline = np.array([True, False, False, True])
    dependency_available = np.ones(4, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_at_least_n_fails_when_fewer_qualifying_providers_than_minimum():
    """Even if every qualifying provider is operational, too few qualifying
    providers relative to minimum_available must fail."""
    edge = _edge(
        "e1", target_index=1, provider_indices=(0,),
        availability_policy=POLICY_AT_LEAST_N, minimum_available=2,
    )
    baseline = np.array([True, True])
    dependency_available = np.ones(2, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_at_least_n_zero_providers_is_unavailable():
    edge = _edge(
        "e1", target_index=0, provider_indices=(),
        availability_policy=POLICY_AT_LEAST_N, minimum_available=1,
    )
    baseline = np.array([True])
    dependency_available = np.ones(1, dtype=bool)
    assert evaluate_edge_availability(edge, baseline, dependency_available) is False


def test_at_least_n_invalid_minimum_available_raises():
    edge = _edge(
        "e1", target_index=1, provider_indices=(0,),
        availability_policy=POLICY_AT_LEAST_N, minimum_available=0,
    )
    baseline = np.array([True, True])
    dependency_available = np.ones(2, dtype=bool)
    with pytest.raises(ValueError, match="minimum_available"):
        evaluate_edge_availability(edge, baseline, dependency_available)


# ---------------------------------------------------------------------------
# Graph-level evaluation: empty/missing cases and chain convergence
# ---------------------------------------------------------------------------
def test_no_target_assets_produces_no_evaluation():
    dependency_available, report = evaluate_dependency_availability([], np.ones(3, dtype=bool), num_assets=3)
    assert dependency_available.tolist() == [True, True, True]
    assert report["evaluated_target_count"] == 0
    assert report["unavailable_targets"] == []


def test_zero_providers_fails_for_existing_target():
    edge = _edge("e1", target_index=0, provider_indices=(), availability_policy=POLICY_ANY)
    dependency_available, report = evaluate_dependency_availability(
        [edge], np.ones(1, dtype=bool), num_assets=1
    )
    assert dependency_available.tolist() == [False]
    assert report["unavailable_targets"] == [0]


def test_dependency_chains_converge():
    """target(2) <- provider B(1) <- provider C(0). If C fails, B becomes
    dependency-unavailable, which must cascade to make target 2 unavailable,
    even though B's own baseline_available is True."""
    edge_b_on_c = _edge("b_on_c", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    edge_target_on_b = _edge("target_on_b", target_index=2, provider_indices=(1,), availability_policy=POLICY_EXCLUSIVE)

    baseline = np.array([False, True, True])  # C (index 0) has failed
    dependency_available, report = evaluate_dependency_availability(
        [edge_b_on_c, edge_target_on_b], baseline, num_assets=3
    )
    assert dependency_available.tolist() == [True, False, False]
    assert report["iterations"] >= 2


def test_long_dependency_chain_converges_within_default_bound():
    """A 6-node linear chain (target <- p4 <- p3 <- p2 <- p1 <- p0, all
    exclusive) with the root provider failing must fully propagate the
    failure to every downstream node within the default (proven-sufficient)
    max_iterations bound, without raising."""
    num_assets = 6
    edges = [
        _edge(f"edge_{i}", target_index=i, provider_indices=(i - 1,), availability_policy=POLICY_EXCLUSIVE)
        for i in range(1, num_assets)
    ]
    baseline = np.ones(num_assets, dtype=bool)
    baseline[0] = False  # root provider fails

    dependency_available, report = evaluate_dependency_availability(
        edges, baseline, num_assets=num_assets
    )
    assert dependency_available.tolist() == [True, False, False, False, False, False]
    assert report["iterations"] <= num_assets - 1 + 1


def test_dependency_graph_raises_clear_error_instead_of_stale_result_when_bound_too_small():
    """If max_iterations is deliberately set below the proven-sufficient
    bound, evaluation must raise a clear RuntimeError rather than silently
    returning a stale, under-propagated dependency_available result."""
    num_assets = 6
    edges = [
        _edge(f"edge_{i}", target_index=i, provider_indices=(i - 1,), availability_policy=POLICY_EXCLUSIVE)
        for i in range(1, num_assets)
    ]
    baseline = np.ones(num_assets, dtype=bool)
    baseline[0] = False  # root provider fails; propagation needs 5 passes

    with pytest.raises(RuntimeError, match="did not converge"):
        evaluate_dependency_availability(
            edges, baseline, num_assets=num_assets, max_iterations=1
        )


# ---------------------------------------------------------------------------
# Restart-wait behavior
# ---------------------------------------------------------------------------
def test_restoration_starts_a_namespaced_restart_wait():
    edge = _edge("e1", target_index=0, provider_indices=(1,), availability_policy=POLICY_ANY)
    wait_vectors: dict = {}
    active_masks: dict = {}

    # First call: dependency already available, previous=None -> no spurious start.
    activate_dependency_restart_waits(
        np.array([True]), None, [edge], wait_vectors, active_masks,
        restart_delay_steps=5.0, num_assets=1,
    )
    key = restart_wait_key("e1")
    assert wait_vectors[key][0] == 0.0

    # Now simulate a genuine unavailable -> available transition.
    previous = np.array([False])
    now = np.array([True])
    activate_dependency_restart_waits(
        now, previous, [edge], wait_vectors, active_masks,
        restart_delay_steps=5.0, num_assets=1,
    )
    assert wait_vectors[key][0] == 5.0
    assert active_masks[key][0] is True or active_masks[key][0] == True  # noqa: E712


def test_restart_wait_resets_when_dependency_lost_again():
    edge = _edge("e1", target_index=0, provider_indices=(1,), availability_policy=POLICY_ANY)
    wait_vectors = {restart_wait_key("e1"): np.array([3.0])}
    active_masks = {restart_wait_key("e1"): np.array([True])}

    activate_dependency_restart_waits(
        np.array([False]), np.array([True]), [edge], wait_vectors, active_masks,
        restart_delay_steps=5.0, num_assets=1,
    )
    key = restart_wait_key("e1")
    assert wait_vectors[key][0] == 0.0
    assert not active_masks[key][0]


def test_independent_edge_restart_vectors_do_not_overwrite_one_another():
    edge_a = _edge("edge_a", target_index=0, provider_indices=(2,), availability_policy=POLICY_ANY)
    edge_b = _edge("edge_b", target_index=1, provider_indices=(2,), availability_policy=POLICY_ANY)
    wait_vectors: dict = {}
    active_masks: dict = {}

    previous = np.array([False, False])
    now = np.array([True, False])  # only target 0 recovered
    activate_dependency_restart_waits(
        now, previous, [edge_a, edge_b], wait_vectors, active_masks,
        restart_delay_steps=4.0, num_assets=2,
    )

    assert wait_vectors[restart_wait_key("edge_a")][0] == 4.0
    assert wait_vectors[restart_wait_key("edge_b")][1] == 0.0


def test_hazard_recovery_wait_and_restart_wait_do_not_collide_for_same_name():
    """hazard_recovery_wait::<name> and restart_wait::<name> must be stored as
    distinct entries in the same shared wait_vectors dict even when the
    unprefixed name/edge_key is identical -- the namespace prefix alone must
    disambiguate them."""
    shared_name = "msls_to_hospital"
    hazard_key = hazard_recovery_wait_key(shared_name)
    restart_key = restart_wait_key(shared_name)

    assert hazard_key != restart_key
    assert hazard_key == "hazard_recovery_wait::msls_to_hospital"
    assert restart_key == "restart_wait::msls_to_hospital"

    wait_vectors: dict = {}
    hazard_active_masks: dict = {}
    restart_active_masks: dict = {}

    # Populate the hazard-recovery wait first.
    wait_vectors.setdefault(hazard_key, np.zeros(1))[0] = 7.0
    hazard_active_masks[hazard_key] = np.array([True])

    # Then independently populate the restart wait under the *same* shared
    # name via the real activation helper.
    edge = _edge(shared_name, target_index=0, provider_indices=(1,), availability_policy=POLICY_ANY)
    activate_dependency_restart_waits(
        np.array([True]), np.array([False]), [edge], wait_vectors, restart_active_masks,
        restart_delay_steps=3.0, num_assets=1,
    )

    # Both entries coexist untouched in the same dict, keyed independently.
    assert wait_vectors[hazard_key][0] == 7.0
    assert wait_vectors[restart_key][0] == 3.0
    assert len(wait_vectors) == 2


def test_hazard_recovery_wait_does_not_affect_restart_ready():
    hazard_key = hazard_recovery_wait_key("dependency_wait")
    restart_key = restart_wait_key("some_edge")

    wait_vectors = {hazard_key: np.array([10.0]), restart_key: np.array([0.0])}
    active_masks = {hazard_key: np.array([True]), restart_key: np.array([False])}

    ready = compute_restart_ready(1, wait_vectors, active_masks)
    # hazard_recovery_wait:: is still counting down (10.0, active) but must be
    # ignored entirely by restart_ready.
    assert ready.tolist() == [True]


def test_restart_ready_blocks_while_restart_wait_active():
    restart_key = restart_wait_key("edge_x")
    wait_vectors = {restart_key: np.array([2.0])}
    active_masks = {restart_key: np.array([True])}

    ready = compute_restart_ready(1, wait_vectors, active_masks)
    assert ready.tolist() == [False]


# ---------------------------------------------------------------------------
# Full orchestration: evaluate_operational_state
# ---------------------------------------------------------------------------
def _empty_graph():
    return DependencyKnowledgeGraph()


def test_state_operational_equals_final_effective_state():
    asset_type = np.array(["msls", "hospital"])
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    wait_vectors = {"repair_time": np.zeros(2)}
    hazard_active_masks: dict = {}
    restart_active_masks: dict = {}

    effective_operational, report = evaluate_operational_state(
        intrinsic_operational=np.array([True, True]),
        asset_type=asset_type,
        hazard_type="flooding",
        knowledge_graph=_empty_graph(),
        flooded_mask=np.array([True, False]),  # msls flooded, but no hazard rule -> no effect
        dependency_edges=[edge],
        wait_vectors=wait_vectors,
        hazard_active_masks=hazard_active_masks,
        restart_active_masks=restart_active_masks,
    )

    assert effective_operational.tolist() == report["effective_operational"].tolist()
    combined = (
        report["intrinsic_operational"]
        & report["hazard_available"]
        & report["dependency_available"]
        & report["restart_ready"]
    )
    assert effective_operational.tolist() == combined.tolist()


def test_dependency_blocking_does_not_change_damage_or_repair_time():
    asset_type = np.array(["msls", "hospital"])
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    repair_time = np.array([5.0, 0.0])
    repair_time_before = repair_time.copy()
    wait_vectors = {"repair_time": repair_time}

    evaluate_operational_state(
        intrinsic_operational=np.array([False, True]),  # provider 0 physically down
        asset_type=asset_type,
        hazard_type="flooding",
        knowledge_graph=_empty_graph(),
        repair_time=repair_time,
        dependency_edges=[edge],
        wait_vectors=wait_vectors,
        hazard_active_masks={},
        restart_active_masks={},
    )

    assert repair_time.tolist() == repair_time_before.tolist()


def test_dependency_blocking_does_not_trigger_physical_delayed_failure():
    """Dependency unavailability must never populate hazard_recovery_wait::*
    vectors -- those are reserved for hazard return-to-operational triggers."""
    asset_type = np.array(["msls", "hospital"])
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)
    wait_vectors = {"repair_time": np.zeros(2)}
    hazard_active_masks: dict = {}

    _, report = evaluate_operational_state(
        intrinsic_operational=np.array([False, True]),
        asset_type=asset_type,
        hazard_type="flooding",
        knowledge_graph=_empty_graph(),
        dependency_edges=[edge],
        wait_vectors=wait_vectors,
        hazard_active_masks=hazard_active_masks,
        restart_active_masks={},
    )
    assert report["dependency_available"].tolist() == [True, False]
    assert not any(k.startswith("hazard_recovery_wait::") for k in wait_vectors)
    assert hazard_active_masks == {}


def test_evaluate_operational_state_via_simulation_wiring():
    """Integration check that _update_operational_state publishes the same
    5-way separated state and that state.operational == effective_operational."""
    asset_type = np.array(["msls", "hospital"])
    edge = _edge("e1", target_index=1, provider_indices=(0,), availability_policy=POLICY_EXCLUSIVE)

    state = SimulationState(None, 2)
    state.operational = np.array([False, True])  # provider physically down

    config = {
        "dependency_parameters": {
            "knowledge_graph": [],
            "hazard_type": "flooding",
            "dependency_restart_delay_steps": 3.0,
        }
    }

    _update_operational_state(
        state, asset_type, np.zeros(2, dtype=bool), config, repair_threshold=0.0,
        knowledge_graph=_empty_graph(), dependency_edges=[edge],
    )

    assert state.operational.tolist() == [False, False]
    assert state.operational.tolist() == state.effective_operational.tolist()
    assert state.dependency_available.tolist() == [True, False]

    # Provider recovers and the target itself is intrinsically fine -> the
    # target's dependency becomes available and its namespaced restart wait
    # starts counting down, holding the target non-operational until the
    # restart delay elapses (restart-ready gating, independent of the
    # dependency itself already being satisfied).
    state.operational = np.array([True, True])
    _update_operational_state(
        state, asset_type, np.zeros(2, dtype=bool), config, repair_threshold=0.0,
        knowledge_graph=_empty_graph(), dependency_edges=[edge],
    )
    key = restart_wait_key("e1")
    assert state.recovery_wait_vectors[key][1] == 3.0
    assert state.dependency_available.tolist() == [True, True]
    assert state.restart_ready.tolist() == [True, False]
    assert state.operational.tolist() == [True, False]
