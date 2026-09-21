"""Focused tests for runtime dependency-topology expansion (slice B).

Covers direct/voronoi/radius topology expansion from type-level rules into
concrete asset-level :class:`~src.dependency_topology.DependencyEdge` runtime
edges, including CRS handling for radius topology and the three-state
no-provider/no-target/has-provider representation.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dependency_knowledge_graph import (
    DependencyKnowledgeGraph,
    KnowledgeGraphRule,
    POLICY_ANY,
    POLICY_AT_LEAST_N,
    POLICY_EXCLUSIVE,
    RELATION_DEPENDENCY,
    TOPOLOGY_DIRECT,
    TOPOLOGY_RADIUS,
    TOPOLOGY_VORONOI,
)
from src.dependency_topology import (
    DependencyEdge,
    expand_dependency_edges,
    index_edges_by_key,
    index_edges_by_target,
    targets_without_providers,
)

PROJECTED_CRS = "EPSG:28992"  # Dutch RD New, projected + metric


def _make_assets(records, crs=PROJECTED_CRS):
    """Build a positionally-indexed GeoDataFrame like ``compile_asset_gdfs`` would.

    ``records`` is a list of ``(type, x, y)`` tuples; the resulting index is
    a plain RangeIndex (0..N-1), matching the convention used throughout the
    simulation for asset positional indices.
    """
    types = [r[0] for r in records]
    geoms = [Point(r[1], r[2]) for r in records]
    return gpd.GeoDataFrame({"type": types, "geometry": geoms}, crs=crs)


def _single_dependency_graph(rule_dict):
    return DependencyKnowledgeGraph.from_config([rule_dict])


# ---------------------------------------------------------------------------
# Direct topology
# ---------------------------------------------------------------------------

def test_direct_topology_single_source_governs_all_targets():
    gdf = _make_assets(
        [
            ("msls", 0, 0),
            ("hospital", 10, 0),
            ("hospital", -10, 0),
            ("hospital", 0, 20),
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 3
    assert all(e.provider_indices == (0,) for e in edges)
    assert [e.target_index for e in edges] == [1, 2, 3]  # deterministic order


def test_direct_topology_multiple_sources_raises_value_error():
    gdf = _make_assets(
        [
            ("msls", 0, 0),
            ("msls", 100, 0),
            ("hospital", 10, 0),
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    with pytest.raises(ValueError, match="topology='direct' cannot deterministically map"):
        expand_dependency_edges(gdf, kg)


def test_direct_topology_no_source_assets_produces_unavailable_edges():
    gdf = _make_assets([("hospital", 0, 0), ("hospital", 10, 0)])
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 2
    assert all(e.provider_indices == () for e in edges)
    assert all(not e.has_provider for e in edges)


def test_direct_topology_no_target_assets_returns_no_edges():
    gdf = _make_assets([("msls", 0, 0)])
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert edges == []
    # Absence of target assets must not be reported as "unavailable".
    assert targets_without_providers(edges) == []


# ---------------------------------------------------------------------------
# Voronoi topology
# ---------------------------------------------------------------------------

def test_voronoi_topology_assigns_exactly_one_provider_per_target():
    # create_voronoi_for_asset_type() (reused from src/impacts.py) only keeps
    # *bounded* Voronoi cells, so at least two providers need to be flanked by
    # additional "bounding" providers far outside the area under test for
    # their cells to close up. Only the west (index 0) / east (index 1)
    # providers are relevant to this test's assertions.
    gdf = _make_assets(
        [
            ("msls", 0, 0),        # index 0 - west provider
            ("msls", 1000, 0),     # index 1 - east provider
            ("msls", -100_000, -100_000),  # index 2 - far bounding provider
            ("msls", -100_000, 100_000),   # index 3 - far bounding provider
            ("msls", 1_100_000, -100_000),  # index 4 - far bounding provider
            ("msls", 1_100_000, 100_000),   # index 5 - far bounding provider
            ("hospital", -100, 0),  # index 6 - closer to west provider
            ("hospital", 1100, 0),  # index 7 - closer to east provider
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    by_target = index_edges_by_target(edges)
    assert len(by_target[6]) == 1
    assert len(by_target[7]) == 1

    west_edge = by_target[6][0]
    east_edge = by_target[7][0]
    assert len(west_edge.provider_indices) == 1
    assert len(east_edge.provider_indices) == 1
    assert west_edge.provider_indices == (0,)
    assert east_edge.provider_indices == (1,)
    # Exclusive: exactly one governing provider, never more than one.
    assert all(len(e.provider_indices) <= 1 for e in edges)


def test_voronoi_topology_no_providers_when_targets_exist():
    gdf = _make_assets([("hospital", 0, 0), ("hospital", 10, 0)])
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 2
    assert all(e.provider_indices == () for e in edges)
    assert sorted(targets_without_providers(edges)) == [0, 1]


def test_voronoi_topology_no_targets_returns_no_edges():
    gdf = _make_assets([("msls", 0, 0), ("msls", 1000, 0)])
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "voronoi",
            "availability_policy": "exclusive",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert edges == []


# ---------------------------------------------------------------------------
# Radius topology
# ---------------------------------------------------------------------------

def test_radius_topology_multiple_qualifying_providers():
    gdf = _make_assets(
        [
            ("ms", 0, 50),    # index 0 - distance 50 from target
            ("ms", 100, 0),   # index 1 - distance 100 from target
            ("ms", 0, 140),   # index 2 - distance 140 from target
            ("ms", 200, 0),   # index 3 - distance 200 from target (excluded)
            ("hospital", 0, 0),  # index 4 - target
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 150.0,
            "availability_policy": "any",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.target_index == 4
    assert edge.provider_indices == (0, 1, 2)  # deterministic ascending order
    assert edge.has_provider


def test_radius_topology_includes_provider_exactly_on_boundary():
    gdf = _make_assets(
        [
            ("ms", 0, 100),   # index 0 - distance exactly 100 from target
            ("hospital", 0, 0),  # index 1 - target
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 100.0,
            "availability_policy": "any",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 1
    assert edges[0].provider_indices == (0,)


def test_radius_topology_no_providers_when_targets_exist():
    gdf = _make_assets(
        [
            ("ms", 0, 500),   # too far away
            ("hospital", 0, 0),
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 50.0,
            "availability_policy": "any",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 1
    assert edges[0].provider_indices == ()
    assert not edges[0].has_provider
    assert targets_without_providers(edges) == [1]


def test_radius_topology_missing_crs_raises_clear_error():
    gdf = _make_assets([("ms", 0, 0), ("hospital", 10, 0)], crs=None)
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 50.0,
            "availability_policy": "any",
        }
    )
    with pytest.raises(ValueError, match="gdf_assets.crs"):
        expand_dependency_edges(gdf, kg)


def test_radius_topology_unsuitable_projected_crs_raises_clear_error():
    # EPSG:2263 (NY Long Island, US survey feet) is projected but not metric.
    gdf = _make_assets([("ms", 0, 0), ("hospital", 10, 0)], crs="EPSG:2263")
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 50.0,
            "availability_policy": "any",
        }
    )
    with pytest.raises(ValueError, match="projected metric CRS"):
        expand_dependency_edges(gdf, kg)


def test_radius_topology_reprojects_geographic_crs_automatically():
    # Two points ~ a few hundred metres apart near Delft, NL, in EPSG:4326.
    gdf = _make_assets(
        [
            ("ms", 4.35, 52.01),
            ("hospital", 4.351, 52.011),
        ],
        crs="EPSG:4326",
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 5000.0,
            "availability_policy": "any",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 1
    assert edges[0].provider_indices == (0,)


# ---------------------------------------------------------------------------
# Availability-policy metadata preservation
# ---------------------------------------------------------------------------

def test_any_and_at_least_n_policy_metadata_preserved_on_edges():
    gdf = _make_assets(
        [
            ("ms", 0, 0),
            ("ms", 10, 0),
            ("hospital", 5, 0),
        ]
    )
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "relation": "dependency",
                "source_type": "ms",
                "target_type": "hospital",
                "topology": "radius",
                "radius_m": 100.0,
                "availability_policy": "any",
            }
        ]
    )
    edges = expand_dependency_edges(gdf, kg)
    assert edges[0].availability_policy == POLICY_ANY
    assert edges[0].minimum_available is None

    kg_quorum = DependencyKnowledgeGraph.from_config(
        [
            {
                "relation": "dependency",
                "source_type": "ms",
                "target_type": "hospital",
                "topology": "radius",
                "radius_m": 100.0,
                "availability_policy": "at_least_n",
                "minimum_available": 2,
            }
        ]
    )
    edges_quorum = expand_dependency_edges(gdf, kg_quorum)
    assert edges_quorum[0].availability_policy == POLICY_AT_LEAST_N
    assert edges_quorum[0].minimum_available == 2
    assert edges_quorum[0].provider_indices == (0, 1)


# ---------------------------------------------------------------------------
# Deterministic ordering and stable keys
# ---------------------------------------------------------------------------

def test_edge_ordering_and_keys_are_deterministic_across_repeated_expansion():
    gdf = _make_assets(
        [
            ("msls", 0, 0),
            ("hospital", 30, 0),
            ("hospital", -30, 0),
            ("hospital", 0, 30),
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    edges_first = expand_dependency_edges(gdf, kg)
    edges_second = expand_dependency_edges(gdf, kg)

    assert [e.edge_key for e in edges_first] == [e.edge_key for e in edges_second]
    assert [e.target_index for e in edges_first] == [1, 2, 3]

    lookup = index_edges_by_key(edges_first)
    assert len(lookup) == len(edges_first)
    for edge in edges_first:
        assert lookup[edge.edge_key] is edge


def test_edge_key_distinguishes_different_rules_for_same_target_type():
    gdf = _make_assets(
        [
            ("msls", 0, 0),
            ("ms", 5, 5),
            ("hospital", 1, 1),
        ]
    )
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "relation": "dependency",
                "source_type": "msls",
                "target_type": "hospital",
                "topology": "direct",
                "availability_policy": "exclusive",
            },
            {
                "relation": "dependency",
                "source_type": "ms",
                "target_type": "hospital",
                "topology": "radius",
                "radius_m": 100.0,
                "availability_policy": "any",
            },
        ]
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 2
    keys = {e.edge_key for e in edges}
    assert len(keys) == 2  # both edges target the same asset but keys differ
    by_target = index_edges_by_target(edges)
    assert len(by_target[2]) == 2


def test_edge_key_includes_relation_and_differs_across_relations():
    """Two otherwise-identical rule descriptions with different `relation`
    values must never collide on edge_key, even though only relation=
    "dependency" rules are expanded by expand_dependency_edges today."""
    import types

    from src.dependency_topology import _build_edge_key

    common_fields = dict(
        source_type="msls",
        target_type="hospital",
        topology=TOPOLOGY_DIRECT,
        availability_policy=POLICY_EXCLUSIVE,
        minimum_available=None,
    )
    dependency_like = types.SimpleNamespace(relation="dependency", **common_fields)
    hazard_like = types.SimpleNamespace(relation="hazard", **common_fields)

    key_dependency = _build_edge_key(dependency_like, target_index=0)
    key_hazard = _build_edge_key(hazard_like, target_index=0)

    assert key_dependency != key_hazard
    assert key_dependency.startswith("dependency|")
    assert key_hazard.startswith("hazard|")


def test_edge_key_includes_minimum_available_and_differs_across_quorum_values():
    gdf = _make_assets(
        [
            ("ms", 0, 0),
            ("ms", 10, 0),
            ("ms", 20, 0),
            ("hospital", 5, 0),
        ]
    )
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "relation": "dependency",
                "source_type": "ms",
                "target_type": "hospital",
                "topology": "radius",
                "radius_m": 100.0,
                "availability_policy": "at_least_n",
                "minimum_available": 2,
            },
            {
                "relation": "dependency",
                "source_type": "ms",
                "target_type": "hospital",
                "topology": "radius",
                "radius_m": 100.0,
                "availability_policy": "at_least_n",
                "minimum_available": 3,
            },
        ]
    )
    edges = expand_dependency_edges(gdf, kg)
    assert len(edges) == 2  # one edge per rule for the same target asset

    keys = {e.edge_key for e in edges}
    assert len(keys) == 2  # quorum value must be part of the key
    minimums = {e.minimum_available for e in edges}
    assert minimums == {2, 3}

    key_by_minimum = {e.minimum_available: e.edge_key for e in edges}
    assert "min=2" in key_by_minimum[2]
    assert "min=3" in key_by_minimum[3]

    # The two rules also target the same asset -- confirm both survive
    # independently rather than one overwriting the other.
    by_target = index_edges_by_target(edges)
    assert len(by_target[3]) == 2


def test_edge_key_omits_minimum_available_token_for_non_quorum_policies():
    gdf = _make_assets([("ms", 0, 0), ("hospital", 5, 0)])
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "ms",
            "target_type": "hospital",
            "topology": "radius",
            "radius_m": 100.0,
            "availability_policy": "any",
        }
    )
    edges = expand_dependency_edges(gdf, kg)
    assert "min=" not in edges[0].edge_key
    assert edges[0].edge_key == (
        "dependency|ms->hospital|radius|any|target=1"
    )


def test_edge_key_determinism_is_preserved_after_the_fix():
    gdf = _make_assets(
        [
            ("msls", 0, 0),
            ("hospital", 30, 0),
            ("hospital", -30, 0),
        ]
    )
    kg = _single_dependency_graph(
        {
            "relation": "dependency",
            "source_type": "msls",
            "target_type": "hospital",
            "topology": "direct",
            "availability_policy": "exclusive",
        }
    )
    edges_a = expand_dependency_edges(gdf, kg)
    edges_b = expand_dependency_edges(gdf, kg)
    assert [e.edge_key for e in edges_a] == [e.edge_key for e in edges_b]
    assert edges_a[0].edge_key == "dependency|msls->hospital|direct|exclusive|target=1"
    assert edges_a[1].edge_key == "dependency|msls->hospital|direct|exclusive|target=2"


# ---------------------------------------------------------------------------
# Static configuration purity (no asset IDs / indices)
# ---------------------------------------------------------------------------

def test_static_dependency_rule_never_carries_asset_ids_or_indices():
    rule = KnowledgeGraphRule(
        relation=RELATION_DEPENDENCY,
        source_type="msls",
        target_type="hospital",
        topology=TOPOLOGY_DIRECT,
        availability_policy=POLICY_EXCLUSIVE,
    )
    field_names = set(KnowledgeGraphRule.__dataclass_fields__.keys())
    forbidden = {
        "asset_id",
        "asset_index",
        "supplier_index",
        "dependent_index",
        "source_index",
        "target_index",
    }
    assert field_names.isdisjoint(forbidden)
    # Asset indices only ever appear on the runtime-generated DependencyEdge.
    assert "target_index" in DependencyEdge.__dataclass_fields__
    assert "provider_indices" in DependencyEdge.__dataclass_fields__

