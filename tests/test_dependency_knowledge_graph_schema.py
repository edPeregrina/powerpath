"""Focused tests for the type-level knowledge graph schema (slice A).

Covers:
* type-level-only declarations (no asset IDs/indices anywhere in config)
* multiple assets of the same type sharing a single type-level hazard rule
* validation of the new relation/topology/availability_policy schema
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependency_knowledge_graph import (
    POLICY_ANY,
    POLICY_AT_LEAST_N,
    POLICY_EXCLUSIVE,
    RELATION_DEPENDENCY,
    RELATION_HAZARD,
    TOPOLOGY_DIRECT,
    TOPOLOGY_RADIUS,
    TOPOLOGY_VORONOI,
    TRIGGER_IMMEDIATE,
    TRIGGER_REPAIR_BELOW,
    TRIGGER_REPAIR_COMPLETE,
    DependencyKnowledgeGraph,
    KnowledgeGraphRule,
    ReturnToOperational,
    build_default_knowledge_graph,
)


# ---------------------------------------------------------------------------
# Type-level-only declarations
# ---------------------------------------------------------------------------

def test_hazard_rule_accepts_type_only_fields():
    rule = KnowledgeGraphRule(
        relation=RELATION_HAZARD,
        source_type="msls",
        hazard_type="flooding",
        hazard_blocks_operation=True,
        return_to_operational=ReturnToOperational(trigger=TRIGGER_REPAIR_COMPLETE),
    )
    assert rule.is_hazard_rule
    assert rule.source_type == "msls"
    assert rule.target_type is None


def test_dependency_rule_accepts_type_only_fields():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="msls",
        target_type="hospital",
        topology=TOPOLOGY_DIRECT,
        availability_policy=POLICY_EXCLUSIVE,
    )
    assert rule.is_dependency_rule
    assert rule.source_type == "msls"
    assert rule.target_type == "hospital"


def test_knowledge_graph_rule_dataclass_has_no_asset_id_fields():
    """Static schema must not expose per-asset identifiers of any kind."""
    field_names = set(KnowledgeGraphRule.__dataclass_fields__.keys())
    forbidden = {
        "asset_id",
        "asset_index",
        "supplier_index",
        "dependent_index",
        "source_index",
        "target_index",
        "asset_ids",
        "indices",
    }
    assert field_names.isdisjoint(forbidden)


def test_from_config_round_trips_type_level_dicts():
    config = [
        {
            "relation": "hazard",
            "hazard_type": "flooding",
            "source_type": "msls",
            "hazard_blocks_operation": False,
            "return_to_operational": {"trigger": "repair_complete"},
        },
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "voronoi",
            "availability_policy": "any",
        },
    ]
    kg = DependencyKnowledgeGraph.from_config(config)
    assert len(kg) == 2
    round_tripped = kg.to_config()
    assert round_tripped[0]["source_type"] == "msls"
    assert round_tripped[1]["target_type"] == "hospital"
    assert round_tripped[1]["topology"] == "voronoi"
    assert round_tripped[1]["availability_policy"] == "any"


def test_json_round_trip_preserves_schema():
    kg = build_default_knowledge_graph()
    json_str = kg.to_json()
    reloaded = DependencyKnowledgeGraph.from_json(json_str)
    assert len(reloaded) == len(kg)
    assert reloaded.to_config() == kg.to_config()


# ---------------------------------------------------------------------------
# Multiple assets sharing the same type-level hazard rule
# ---------------------------------------------------------------------------

def test_single_hazard_rule_governs_every_asset_of_that_type():
    """A single type-level rule must apply uniformly no matter how many
    concrete assets of that type exist -- the graph never grows with the
    asset population."""
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "relation": "hazard",
                "hazard_type": "flooding",
                "source_type": "msls",
                "hazard_blocks_operation": True,
                "return_to_operational": {"trigger": "repair_complete"},
            }
        ]
    )
    # Simulate 50 msls assets among a larger population of mixed types.
    asset_type = np.array(["msls"] * 50 + ["hospital"] * 10 + ["road"] * 5)

    matched_rule_ids = set()
    for a_type in np.unique(asset_type):
        rules = kg.get_hazard_rules("flooding", a_type)
        if a_type == "msls":
            assert len(rules) == 1
            matched_rule_ids.add(id(rules[0]))
        else:
            assert rules == []

    # Exactly one rule object services every msls asset regardless of count.
    assert len(matched_rule_ids) == 1
    assert len(kg) == 1


def test_get_hazard_rules_or_default_returns_default_for_unlisted_type():
    kg = build_default_knowledge_graph()
    rules = kg.get_hazard_rules_or_default("flooding", "unlisted_type")
    assert len(rules) == 1
    assert rules[0].hazard_blocks_operation is False
    assert rules[0].return_to_operational.trigger == TRIGGER_IMMEDIATE


def test_get_dependency_rules_filters_by_source_and_target():
    kg = build_default_knowledge_graph()
    all_deps = kg.get_dependency_rules()
    assert len(all_deps) == 1
    assert kg.get_dependency_rules(source_type="msls")[0].target_type == "hospital"
    assert kg.get_dependency_rules(target_type="hospital")[0].source_type == "msls"
    assert kg.get_dependency_rules(source_type="nonexistent") == []


def test_default_knowledge_graph_has_no_road_rules():
    kg = build_default_knowledge_graph()
    assert "road" not in kg.all_source_types()
    assert "road" not in kg.all_target_types()


# ---------------------------------------------------------------------------
# Rule validation
# ---------------------------------------------------------------------------

def test_invalid_relation_rejected():
    with pytest.raises(ValueError, match="Invalid relation"):
        KnowledgeGraphRule(relation="bogus", source_type="msls")


def test_hazard_rule_requires_hazard_type():
    with pytest.raises(ValueError, match="hazard_type is required"):
        KnowledgeGraphRule(relation=RELATION_HAZARD, source_type="msls")


def test_hazard_rule_rejects_target_type():
    with pytest.raises(ValueError, match="target_type must not be set"):
        KnowledgeGraphRule(
            relation=RELATION_HAZARD,
            source_type="msls",
            hazard_type="flooding",
            target_type="hospital",
        )


def test_hazard_rule_rejects_topology_and_policy_fields():
    with pytest.raises(ValueError, match="topology is not applicable"):
        KnowledgeGraphRule(
            relation=RELATION_HAZARD,
            source_type="msls",
            hazard_type="flooding",
            topology=TOPOLOGY_DIRECT,
        )
    with pytest.raises(ValueError, match="availability_policy is not applicable"):
        KnowledgeGraphRule(
            relation=RELATION_HAZARD,
            source_type="msls",
            hazard_type="flooding",
            availability_policy=POLICY_ANY,
        )


def test_dependency_rule_requires_target_type():
    with pytest.raises(ValueError, match="target_type is required"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="msls",
            topology=TOPOLOGY_DIRECT,
            availability_policy=POLICY_EXCLUSIVE,
        )


def test_dependency_rule_rejects_hazard_only_fields():
    with pytest.raises(ValueError, match="hazard_type is not applicable"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="msls",
            target_type="hospital",
            hazard_type="flooding",
            topology=TOPOLOGY_DIRECT,
            availability_policy=POLICY_EXCLUSIVE,
        )
    with pytest.raises(ValueError, match="return_to_operational is not applicable"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="msls",
            target_type="hospital",
            return_to_operational=ReturnToOperational(),
            topology=TOPOLOGY_DIRECT,
            availability_policy=POLICY_EXCLUSIVE,
        )


def test_dependency_rule_requires_valid_topology():
    with pytest.raises(ValueError, match="Invalid topology"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="msls",
            target_type="hospital",
            topology="nearest",
            availability_policy=POLICY_EXCLUSIVE,
        )


def test_dependency_rule_requires_valid_availability_policy():
    with pytest.raises(ValueError, match="Invalid availability_policy"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="msls",
            target_type="hospital",
            topology=TOPOLOGY_DIRECT,
            availability_policy="majority",
        )


def test_radius_topology_requires_positive_radius_m():
    with pytest.raises(ValueError, match="radius_m must be a positive number"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_RADIUS,
            availability_policy=POLICY_ANY,
        )
    with pytest.raises(ValueError, match="radius_m must be a positive number"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_RADIUS,
            availability_policy=POLICY_ANY,
            radius_m=-5.0,
        )


def test_radius_m_only_applicable_when_topology_is_radius():
    with pytest.raises(ValueError, match="radius_m is only applicable"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_VORONOI,
            availability_policy=POLICY_ANY,
            radius_m=500.0,
        )


def test_at_least_n_policy_requires_positive_minimum_available():
    with pytest.raises(ValueError, match="minimum_available must be a positive integer"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_RADIUS,
            radius_m=500.0,
            availability_policy=POLICY_AT_LEAST_N,
        )
    with pytest.raises(ValueError, match="minimum_available must be a positive integer"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_RADIUS,
            radius_m=500.0,
            availability_policy=POLICY_AT_LEAST_N,
            minimum_available=0,
        )


def test_minimum_available_only_applicable_when_policy_is_at_least_n():
    with pytest.raises(ValueError, match="minimum_available is only applicable"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_DIRECT,
            availability_policy=POLICY_EXCLUSIVE,
            minimum_available=2,
        )


def test_valid_radius_with_quorum_policy_constructs_successfully():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="ms",
        target_type="hospital",
        topology=TOPOLOGY_RADIUS,
        radius_m=500.0,
        availability_policy=POLICY_AT_LEAST_N,
        minimum_available=2,
    )
    assert rule.radius_m == 500.0
    assert rule.minimum_available == 2


def test_source_type_is_required():
    with pytest.raises(ValueError, match="source_type is required"):
        KnowledgeGraphRule(relation=RELATION_HAZARD, source_type="", hazard_type="flooding")


# ---------------------------------------------------------------------------
# exclusive availability_policy requires a single-governing-provider topology
# ---------------------------------------------------------------------------
def test_exclusive_with_radius_topology_raises_at_construction_time():
    with pytest.raises(ValueError, match="exclusive.*requires a topology"):
        KnowledgeGraphRule(
            relation=RELATION_DEPENDENCY,
            source_type="ms",
            target_type="hospital",
            topology=TOPOLOGY_RADIUS,
            radius_m=500.0,
            availability_policy=POLICY_EXCLUSIVE,
        )


def test_radius_topology_with_any_policy_is_still_valid():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="ms",
        target_type="hospital",
        topology=TOPOLOGY_RADIUS,
        radius_m=500.0,
        availability_policy=POLICY_ANY,
    )
    assert rule.topology == TOPOLOGY_RADIUS
    assert rule.availability_policy == POLICY_ANY


def test_radius_topology_with_at_least_n_policy_is_still_valid():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="ms",
        target_type="hospital",
        topology=TOPOLOGY_RADIUS,
        radius_m=500.0,
        availability_policy=POLICY_AT_LEAST_N,
        minimum_available=2,
    )
    assert rule.topology == TOPOLOGY_RADIUS
    assert rule.availability_policy == POLICY_AT_LEAST_N


def test_exclusive_with_direct_topology_remains_valid():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="ms",
        target_type="hospital",
        topology=TOPOLOGY_DIRECT,
        availability_policy=POLICY_EXCLUSIVE,
    )
    assert rule.topology == TOPOLOGY_DIRECT
    assert rule.availability_policy == POLICY_EXCLUSIVE


def test_exclusive_with_voronoi_topology_remains_valid():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="ms",
        target_type="hospital",
        topology=TOPOLOGY_VORONOI,
        availability_policy=POLICY_EXCLUSIVE,
    )
    assert rule.topology == TOPOLOGY_VORONOI
    assert rule.availability_policy == POLICY_EXCLUSIVE


def test_repair_below_threshold_validation_still_enforced_on_return_to_operational():
    with pytest.raises(ValueError, match="threshold must be > 0"):
        ReturnToOperational(trigger=TRIGGER_REPAIR_BELOW, threshold=0.0)
