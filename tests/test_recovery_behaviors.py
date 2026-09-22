import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.damage_recovery import (
    default_damage_ratio_function,
    default_fragility_function,
)
from src.simulation import (
    SimulationState,
    _assign_repair_crews,
    _build_crew_pools,
    _handle_completed_repairs,
    _normalize_number_repair_crews_config,
    _update_unreachable_assets,
)


def _typed_pool_state(typed_config):
    """Build a crew_pools structure directly from a type->count dict, mirroring
    what run_simulation derives from a type-specific number_repair_crews config."""
    normalized = _normalize_number_repair_crews_config(typed_config)
    return _build_crew_pools(normalized["typed"])


def test_grouped_crew_pool_cannot_cross_islands():
    pool_state = _typed_pool_state({"hospital": 1})
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
        crew_pools=pool_state,
    )

    assert assigned.tolist() == [False, False]
    assert pool_state["pools"][0]["available"] == {0: 1}


def test_grouped_pool_assigns_for_singular_island_method():
    pool_state = _typed_pool_state({"hospital": 1})
    pool_state["pools"][0]["available"] = {0: 1}

    _, assigned = _assign_repair_crews(
        timestep=0,
        available_repair_crews={0: 0},
        repair_crews_assigned=np.zeros(2, dtype=bool),
        accessible=np.ones(2, dtype=bool),
        flooded_mask=np.zeros(2, dtype=bool),
        repair_time=np.ones(2),
        island_ids=np.zeros(2, dtype=int),
        method="island",
        verbose=False,
        asset_type=np.array(["hospital", "hospital"]),
        crew_pools=pool_state,
    )

    assert assigned.sum() == 1
    assert pool_state["pools"][0]["available"] == {0: 0}


def test_grouped_scalar_fallback_assigns_for_singular_island_method():
    pool_state = _typed_pool_state({"hospital": 1})

    _, assigned = _assign_repair_crews(
        timestep=0,
        available_repair_crews=0,
        repair_crews_assigned=np.zeros(2, dtype=bool),
        accessible=np.ones(2, dtype=bool),
        flooded_mask=np.zeros(2, dtype=bool),
        repair_time=np.ones(2),
        island_ids=np.zeros(2, dtype=int),
        method="island",
        verbose=False,
        asset_type=np.array(["hospital", "hospital"]),
        crew_pools=pool_state,
    )

    assert assigned.sum() == 1
    assert pool_state["pools"][0]["available"] == 0


def test_grouped_crew_returns_to_repaired_assets_current_island():
    pool_state = _typed_pool_state({"hospital": 1})
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
        crew_pools=pool_state,
    )

    assert pool_state["pools"][0]["available"] == {0: 0, 3: 1}
    assert not state.repair_crews_assigned[0]


def test_busy_grouped_crew_keeps_its_island_reachable():
    pool_state = _typed_pool_state({"hospital": 1})
    pool_state["pools"][0]["available"] = {2: 0}
    state = SimulationState(None, 2)
    state.island_ids[:] = 2
    state.damage_ratio[:] = 1.0
    state.repair_crews_assigned[0] = True

    _update_unreachable_assets(
        state,
        available_repair_crews={2: 0},
        flooded_mask=np.zeros(2, dtype=bool),
        damage_threshold=0.1,
        asset_type=np.array(["hospital", "hospital"]),
        crew_pools=pool_state,
    )

    assert state.unreachable.tolist() == [False, False]


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


def test_completed_repair_writes_intrinsic_operational_not_effective_operational():
    """`_handle_completed_repairs` must restore `state.intrinsic_operational`
    (the persisted physical/intrinsic state), not `state.operational`
    directly -- the effective state is recomputed from the intrinsic state
    every timestep by `_update_operational_state`. Writing to
    `state.operational` here would be silently overwritten (or would mask a
    still-active dependency block) on the very next call."""
    state = SimulationState(None, 1)
    state.intrinsic_operational[0] = False
    state.operational[0] = False
    state.repair_crews_assigned[0] = True
    # repair_time already at 0 (default) -> repair is complete this call.

    _handle_completed_repairs(
        state,
        available_repair_crews=0,
        verbose=False,
        timestep=1,
    )

    assert state.intrinsic_operational[0]
    # `state.operational` is intentionally left untouched here; it is only
    # ever updated by `_update_operational_state`, which reads
    # `state.intrinsic_operational` as input on the next call.
    assert not state.operational[0]


# ---------------------------------------------------------------------------
# Fragility-driven (not damage-driven) MSLS failure
# ---------------------------------------------------------------------------
_MSLS_MEDIAN_FAILURE_DEPTH = 0.6
_DAMAGE_RATIO_COEFFICIENTS = (0.0468, 0.0077)


def test_positive_damage_does_not_imply_fragility_failure(monkeypatch):
    """A flooded MSLS asset can have positive damage_ratio while remaining
    fragility-operational: damage_ratio is a deterministic function of
    hazard depth alone, while fragility failure is a separate, stochastic
    depth-logistic draw. Forcing the random draw high (via monkeypatch)
    makes the stochastic sampler deterministic for this assertion."""
    monkeypatch.setattr(np.random, "random", lambda size: np.full(size, 0.99))

    hazard = np.array([1.0])  # above the msls median failure depth (0.6)
    asset_type = np.array(["msls"])

    damage_ratio = default_damage_ratio_function(hazard, _DAMAGE_RATIO_COEFFICIENTS)
    assert damage_ratio[0] > 0.0  # positive damage from hazard depth alone

    status = default_fragility_function(hazard, asset_type, k=6.0)
    # failure_probability at depth=1.0, k=6.0, median=0.6 is ~0.917 < 0.99,
    # so the (monkeypatched) random draw of 0.99 means the asset survives.
    assert status.tolist() == [1]


def test_fragility_failure_does_imply_intrinsic_failure(monkeypatch):
    """When the (stochastic) fragility draw indicates failure, the asset
    must be reported as failed -- independent of the deterministic damage
    ratio computation."""
    monkeypatch.setattr(np.random, "random", lambda size: np.zeros(size))

    hazard = np.array([1.0])  # above the msls median failure depth (0.6)
    asset_type = np.array(["msls"])

    status = default_fragility_function(hazard, asset_type, k=6.0)
    # failure_probability at depth=1.0, k=6.0, median=0.6 is ~0.917 > 0.0,
    # so the (monkeypatched) random draw of 0.0 means the asset fails.
    assert status.tolist() == [0]


def test_hazard_map_states_fragility_uses_intrinsic_operational_not_effective(monkeypatch):
    """`_update_hazard_map_states`'s fragility gate/write must use
    `state.intrinsic_operational`, not `state.operational` -- a dependency-
    blocked (effective=False) but intrinsically-fine MSLS asset must still
    be subject to fragility evaluation, and a fragility failure must lower
    `state.intrinsic_operational` (not merely `state.operational`, which is
    recomputed from the intrinsic state every timestep).

    This mirrors the exact gating/write logic inside
    `_update_hazard_map_states` without requiring real hazard-map/
    GeoDataFrame plumbing.
    """
    monkeypatch.setattr(np.random, "random", lambda size: np.zeros(size))  # force failure

    state = SimulationState(None, 1)
    state.intrinsic_operational[0] = True
    state.operational[0] = False  # dependency-blocked, but intrinsically fine
    state.current_hazard_values = np.array([1.0])
    asset_type = np.array(["msls"])

    flooded_mask = state.current_hazard_values > 0.5
    assets_to_evaluate = flooded_mask & ~state.repair_crews_assigned & state.intrinsic_operational
    assert assets_to_evaluate.tolist() == [True]  # evaluated despite operational=False

    fragility_operational = np.ones_like(state.intrinsic_operational, dtype=bool)
    fragility_result = default_fragility_function(
        state.current_hazard_values[assets_to_evaluate],
        asset_type[assets_to_evaluate],
        k=6.0,
    )
    fragility_operational[assets_to_evaluate] = fragility_result.astype(bool)
    state.intrinsic_operational = np.minimum(state.intrinsic_operational, fragility_operational)

    assert not state.intrinsic_operational[0]
