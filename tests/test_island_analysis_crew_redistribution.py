import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.island_analysis import (
    _is_nested_actor_distribution,
    _extract_actor_counts,
    _pack_actor_counts,
    update_actor_islands,
    update_repair_crew_islands,
)
from src.simulation import (
    _assemble_nested_crew_state,
    _assign_repair_crews,
    _build_crew_pools,
    _disassemble_nested_crew_state,
    _normalize_number_repair_crews_config,
)
from src.visualisations import _sum_number_repair_crews


# ---------------------------------------------------------------------------
# _is_nested_actor_distribution / extract / pack: island-outer/type-inner
# ---------------------------------------------------------------------------

def test_is_nested_actor_distribution_empty_dict_is_explicitly_not_nested():
    # An empty dict must not be misclassified as ambiguously nested or
    # non-nested: it is explicitly treated as "not nested".
    assert _is_nested_actor_distribution({}) is False


def test_is_nested_actor_distribution_detects_island_outer_type_inner():
    assert _is_nested_actor_distribution({0: {"msls": 3}, 1: {"hospital": 1}}) is True
    assert _is_nested_actor_distribution({0: 3, 1: 5}) is False
    assert _is_nested_actor_distribution(5) is False


def test_extract_and_pack_actor_counts_round_trip_island_outer_type_inner():
    nested = {0: {"msls": 3, "hospital": 1}, 1: {"msls": 2}}

    msls_counts, shape, nested_distribution = _extract_actor_counts(nested, "msls")
    assert shape == "nested"
    assert msls_counts == {0: 3, 1: 2}

    # Redistribute msls crews entirely to island 1; hospital's island-0 entry
    # must survive untouched (independent per-type packing).
    updated = _pack_actor_counts(
        updated_actor_counts={1: 5},
        input_shape=shape,
        nested_distribution=nested_distribution,
        actor_type="msls",
    )
    assert updated == {0: {"hospital": 1}, 1: {"msls": 5}}


# ---------------------------------------------------------------------------
# update_actor_islands / update_repair_crew_islands
# ---------------------------------------------------------------------------

# A single road entirely within island 0 makes redistribution deterministic
# (probability 1.0 on island 0), avoiding flaky random-sampling assertions.
SINGLE_ISLAND_RFIDS = {1: 0}
SINGLE_ISLAND_LENGTHS = {1: 100.0}


def test_update_repair_crew_islands_legacy_flat_dict_unchanged():
    result = update_repair_crew_islands(
        {0: 3},
        previous_rfids_islands=SINGLE_ISLAND_RFIDS,
        current_rfids_islands=SINGLE_ISLAND_RFIDS,
        rfids_lengths=SINGLE_ISLAND_LENGTHS,
    )
    assert result == {0: 3}


def test_update_repair_crew_islands_scalar_int_unchanged_behavior():
    result = update_repair_crew_islands(
        3,
        previous_rfids_islands=None,
        current_rfids_islands=SINGLE_ISLAND_RFIDS,
        rfids_lengths=SINGLE_ISLAND_LENGTHS,
    )
    assert result == {0: 3}


def test_update_repair_crew_islands_nested_multiple_types_isolated():
    nested = {0: {"msls": 4, "hospital": 2}}
    result = update_repair_crew_islands(
        nested,
        previous_rfids_islands=SINGLE_ISLAND_RFIDS,
        current_rfids_islands=SINGLE_ISLAND_RFIDS,
        rfids_lengths=SINGLE_ISLAND_LENGTHS,
    )
    assert result == {0: {"msls": 4, "hospital": 2}}


def test_update_repair_crew_islands_reuses_cached_transition_probabilities():
    overlap_cache = {}
    nested = {0: {"msls": 4, "hospital": 2}}
    update_repair_crew_islands(
        nested,
        previous_rfids_islands=SINGLE_ISLAND_RFIDS,
        current_rfids_islands=SINGLE_ISLAND_RFIDS,
        rfids_lengths=SINGLE_ISLAND_LENGTHS,
        overlap_cache=overlap_cache,
        current_map="map_b",
        previous_map="map_a",
        hazard_threshold=0.2,
    )
    # Both "msls" and "hospital" redistribution passes share exactly one
    # overlap-cache entry for the current_map/previous_map pair.
    assert len(overlap_cache) == 1


# ---------------------------------------------------------------------------
# _normalize_number_repair_crews_config
# ---------------------------------------------------------------------------

def test_normalize_scalar_crews():
    assert _normalize_number_repair_crews_config(10) == {"default": 10, "typed": {}}


def test_normalize_island_keyed_crews_unchanged_shape():
    result = _normalize_number_repair_crews_config({1: 3, 2: 5})
    assert result == {"default": {1: 3, 2: 5}, "typed": {}}


def test_normalize_global_type_specific_crews():
    result = _normalize_number_repair_crews_config({"msls": 4, "hospital": 2})
    assert result == {"default": 0, "typed": {"msls": 4, "hospital": 2}}


def test_normalize_global_type_specific_crews_with_default_sentinel():
    result = _normalize_number_repair_crews_config({"msls": 4, "*": 1})
    assert result == {"default": 1, "typed": {"msls": 4}}


def test_normalize_island_and_type_nested_crews():
    result = _normalize_number_repair_crews_config(
        {1: {"msls": 3, "hospital": 1}, 2: {"msls": 2, "hospital": 4}}
    )
    assert result == {
        "default": {},
        "typed": {"msls": {1: 3, 2: 2}, "hospital": {1: 1, 2: 4}},
    }


def test_normalize_rejects_mixed_key_types():
    with pytest.raises(ValueError):
        _normalize_number_repair_crews_config({1: 3, "hospital": 2})


def test_normalize_rejects_mixed_island_value_types():
    with pytest.raises(ValueError):
        _normalize_number_repair_crews_config({1: 3, 2: {"msls": 1}})


def test_normalize_rejects_negative_counts():
    with pytest.raises(ValueError):
        _normalize_number_repair_crews_config({"hospital": -1})


# ---------------------------------------------------------------------------
# Assemble/disassemble bridge used by _update_hazard_map_states -- this is
# the previously-unreachable, NameError-bugged path (current_map_str /
# previous_map_str) now fixed and exercised together with island redistribution.
# ---------------------------------------------------------------------------

def test_assemble_and_disassemble_typed_and_island_redistribution_no_nameerror():
    normalized = _normalize_number_repair_crews_config(
        {1: {"msls": 3, "hospital": 1}, 2: {"msls": 2, "hospital": 4}}
    )
    available_repair_crews = normalized["default"] or {1: 0, 2: 0}
    crew_pools = _build_crew_pools(normalized["typed"])

    combined = _assemble_nested_crew_state(available_repair_crews, crew_pools)
    assert combined == {
        1: {"*": 0, "msls": 3, "hospital": 1},
        2: {"*": 0, "msls": 2, "hospital": 4},
    }

    current_rfids_islands = {10: 1, 20: 2}
    rfids_lengths = {10: 100.0, 20: 100.0}
    redistributed = update_repair_crew_islands(
        combined,
        previous_rfids_islands=current_rfids_islands,
        current_rfids_islands=current_rfids_islands,
        rfids_lengths=rfids_lengths,
    )

    default_pool = _disassemble_nested_crew_state(redistributed, crew_pools)
    assert default_pool == {1: 0, 2: 0}
    msls_pool = next(p for p in crew_pools["pools"] if "msls" in p["asset_types"])
    hospital_pool = next(p for p in crew_pools["pools"] if "hospital" in p["asset_types"])
    assert msls_pool["available"] == {1: 3, 2: 2}
    assert hospital_pool["available"] == {1: 1, 2: 4}


# ---------------------------------------------------------------------------
# msls-type pool never assigns to hospital assets and vice versa
# ---------------------------------------------------------------------------

def test_typed_pools_never_cross_assign_between_asset_types():
    normalized = _normalize_number_repair_crews_config({"msls": 1, "hospital": 1})
    crew_pools = _build_crew_pools(normalized["typed"])
    for pool in crew_pools["pools"]:
        pool["available"] = {0: 1}

    asset_type = np.array(["msls", "hospital"])
    accessible = np.ones(2, dtype=bool)
    flooded_mask = np.zeros(2, dtype=bool)
    repair_time = np.array([5.0, 5.0])
    island_ids = np.zeros(2, dtype=int)
    assigned = np.zeros(2, dtype=bool)

    _, assigned = _assign_repair_crews(
        timestep=0,
        available_repair_crews={0: 0},
        repair_crews_assigned=assigned,
        accessible=accessible,
        flooded_mask=flooded_mask,
        repair_time=repair_time,
        island_ids=island_ids,
        method="island",
        verbose=False,
        asset_type=asset_type,
        crew_pools=crew_pools,
    )

    # Both the msls and hospital asset each got their own dedicated crew.
    assert assigned.tolist() == [True, True]
    msls_pool = crew_pools["pools"][crew_pools["asset_type_to_pool"]["msls"]]
    hospital_pool = crew_pools["pools"][crew_pools["asset_type_to_pool"]["hospital"]]
    assert msls_pool["available"] == {0: 0}
    assert hospital_pool["available"] == {0: 0}


# ---------------------------------------------------------------------------
# repair_crews_by_asset_type fully removed
# ---------------------------------------------------------------------------

def test_repair_crews_by_asset_type_removed_from_simulation_module():
    import src.simulation as simulation_module

    assert not hasattr(simulation_module, "_normalize_repair_crews_by_asset_type_config")
    assert not hasattr(simulation_module, "_normalize_asset_type_group_key")


# ---------------------------------------------------------------------------
# plot_detailed_analysis_panels total-crews derivation (lightweight)
# ---------------------------------------------------------------------------

def test_sum_number_repair_crews_scalar():
    assert _sum_number_repair_crews(10) == 10


def test_sum_number_repair_crews_island_keyed():
    assert _sum_number_repair_crews({1: 3, 2: 5}) == 8


def test_sum_number_repair_crews_type_keyed():
    assert _sum_number_repair_crews({"msls": 4, "hospital": 2}) == 6


def test_sum_number_repair_crews_island_and_type_nested():
    assert _sum_number_repair_crews(
        {1: {"msls": 3, "hospital": 1}, 2: {"msls": 2, "hospital": 4}}
    ) == 10
