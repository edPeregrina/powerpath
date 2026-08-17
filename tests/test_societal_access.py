"""Scenario tests for src/societal_access.py.

Tests cover both the graph-native Layer A path (build_island_assignment,
build_destination_function_map, compute_origin_access,
compute_access_matrix_from_origins) and the spatial Layer B path
(assign_destinations_to_islands_spatial, assign_origins_to_islands_spatial,
compute_access_matrix).  No real data files are required.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

# Make src importable when tests are run from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.societal_access import (
    SERVICE_NODE_TAXONOMY,
    POPULATION_GROUP_COLUMNS,
    # Layer A — graph-native
    build_island_assignment,
    build_destination_function_map,
    compute_origin_access,
    compute_access_matrix_from_origins,
    # Layer B — spatial helpers (also exposed via backward-compat aliases)
    assign_destinations_to_islands_spatial,
    assign_origins_to_islands_spatial,
    compute_function_access_per_island,   # alias
    join_population_to_islands,           # alias
    # Layer C — shared metrics
    compute_access_matrix,
    compute_equity_gaps,
    # Layer D — wrapper
    analyse_societal_access,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CRS = "EPSG:28992"


def _make_islands(polygons_by_id: dict) -> gpd.GeoDataFrame:
    """Create a minimal islands GeoDataFrame from {island_id: shapely_polygon}."""
    rows = [{"island_id": iid, "geometry": geom} for iid, geom in polygons_by_id.items()]
    return gpd.GeoDataFrame(rows, crs=CRS)


def _make_service_nodes(points_by_type: dict) -> gpd.GeoDataFrame:
    """Create service nodes from {node_type: list_of_(x,y)}."""
    rows = []
    for ntype, coords in points_by_type.items():
        for x, y in coords:
            rows.append({"type": ntype, "geometry": Point(x, y)})
    return gpd.GeoDataFrame(rows, crs=CRS)


def _make_population(cells: list) -> gpd.GeoDataFrame:
    """Create population grid cells."""
    return gpd.GeoDataFrame(cells, crs=CRS)


def _make_nx_graph(edges, directed=False):
    """Build a lightweight NetworkX graph from an edge list."""
    import networkx as nx
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from(edges)
    return G


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_island_scenario():
    """
    Two non-overlapping islands; one hospital on island 0 only.

    Island 0 (main):  box(0, 0, 100, 100)   – 600 people (400 total + 200 elderly)
    Island 1 (small): box(200, 0, 300, 100)  – 400 people (300 total + 100 elderly)

    Hospital at (50, 50) → inside island 0.
    Fire station at (50, 50) → inside island 0.

    Expected health access:
        total   = 400 / 700 ≈ 57.14 %
        elderly = 200 / 300 ≈ 66.67 %
    """
    island0 = box(0, 0, 100, 100)
    island1 = box(200, 0, 300, 100)

    islands = _make_islands({0: island0, 1: island1})

    service_nodes = _make_service_nodes(
        {
            "hospital": [(50, 50)],
            "fire_station": [(50, 50)],
        }
    )

    pop_cells = [
        # Inside island 0
        {
            "geometry": box(10, 10, 40, 40),
            "aantal_inwoners": 250,
            "aantal_inwoners_65_jaar_en_ouder": 100,
            "aantal_inwoners_0_tot_15_jaar": 60,
            "aantal_inwoners_25_tot_45_jaar": 90,
        },
        {
            "geometry": box(50, 50, 80, 80),
            "aantal_inwoners": 150,
            "aantal_inwoners_65_jaar_en_ouder": 100,
            "aantal_inwoners_0_tot_15_jaar": 30,
            "aantal_inwoners_25_tot_45_jaar": 20,
        },
        # Inside island 1
        {
            "geometry": box(210, 10, 290, 90),
            "aantal_inwoners": 300,
            "aantal_inwoners_65_jaar_en_ouder": 100,
            "aantal_inwoners_0_tot_15_jaar": 80,
            "aantal_inwoners_25_tot_45_jaar": 120,
        },
    ]
    population = _make_population(pop_cells)

    return islands, service_nodes, population


# ---------------------------------------------------------------------------
# Tests — Layer A: build_island_assignment
# ---------------------------------------------------------------------------

class TestBuildIslandAssignment:

    def test_two_components(self):
        # Nodes 0-1-2 connected; node 3 isolated → 2 islands
        G = _make_nx_graph([(0, 1), (1, 2)])
        G.add_node(3)
        assignment = build_island_assignment(G)
        assert set(assignment.keys()) == {0, 1, 2, 3}
        # 0, 1, 2 share one island; 3 is alone
        assert assignment[0] == assignment[1] == assignment[2]
        assert assignment[3] != assignment[0]

    def test_each_node_exactly_one_island(self):
        G = _make_nx_graph([(0, 1), (2, 3), (4, 5)])
        assignment = build_island_assignment(G)
        assert len(assignment) == 6
        # Each node has exactly one island id (no duplicates in assignment)
        assert len(set(assignment.keys())) == 6

    def test_isolated_node_is_single_island(self):
        G = _make_nx_graph([])
        G.add_nodes_from([10, 20])
        assignment = build_island_assignment(G)
        assert assignment[10] != assignment[20]

    def test_fully_connected_graph_single_island(self):
        G = _make_nx_graph([(0, 1), (1, 2), (2, 0)])
        assignment = build_island_assignment(G)
        assert assignment[0] == assignment[1] == assignment[2]
        assert len(set(assignment.values())) == 1

    def test_directed_graph_weakly_connected(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (2, 3)])
        assignment = build_island_assignment(G)
        assert assignment[0] == assignment[1]
        assert assignment[2] == assignment[3]
        assert assignment[0] != assignment[2]

    def test_partition_property(self):
        """Every node in one island; no node in multiple islands."""
        G = _make_nx_graph([(0, 1), (1, 2), (3, 4)])
        assignment = build_island_assignment(G)
        # Union of all islands == all nodes
        assert set(assignment.keys()) == set(G.nodes())
        # Each node appears exactly once
        from collections import Counter
        counts = Counter(assignment.keys())
        assert max(counts.values()) == 1


# ---------------------------------------------------------------------------
# Tests — Layer A: build_destination_function_map
# ---------------------------------------------------------------------------

class TestBuildDestinationFunctionMap:

    def test_hospital_on_island_0(self):
        # node 0 is hospital, on island 0
        dest = {0: "hospital", 5: "fire_station"}
        assignment = {0: 0, 1: 0, 5: 1}
        result = build_destination_function_map(dest, assignment)
        assert "health" in result[0]
        assert "emergency_response" in result[1]

    def test_disrupted_destination_skipped(self):
        # node 99 is a hospital but was removed by disruption
        dest = {99: "hospital"}
        assignment = {0: 0, 1: 0}  # node 99 not present
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = build_destination_function_map(dest, assignment)
            assert any("not found in island assignment" in str(x.message) for x in w)
        assert result == {}

    def test_unknown_type_ignored(self):
        dest = {0: "spaceship"}
        assignment = {0: 0}
        result = build_destination_function_map(dest, assignment)
        assert result == {}

    def test_custom_taxonomy(self):
        dest = {0: "my_type"}
        assignment = {0: 0}
        result = build_destination_function_map(dest, assignment, taxonomy={"my_type": "my_func"})
        assert "my_func" in result[0]


# ---------------------------------------------------------------------------
# Tests — Layer A: compute_origin_access
# ---------------------------------------------------------------------------

class TestComputeOriginAccess:

    def test_shared_island_gives_access(self):
        # Origin 10 and hospital are both on island 0
        origin_island_ids = {10: 0, 20: 1}
        island_function_map = {0: frozenset(["health"])}
        result = compute_origin_access(origin_island_ids, island_function_map)
        assert result.loc[10, "health"] == True
        assert result.loc[20, "health"] == False

    def test_no_functions_means_no_access(self):
        origin_island_ids = {10: 0}
        island_function_map = {}
        result = compute_origin_access(origin_island_ids, island_function_map,
                                       all_functions=["health"])
        assert result.loc[10, "health"] == False

    def test_all_functions_explicit(self):
        origin_island_ids = {1: 0}
        island_function_map = {0: frozenset(["health"])}
        result = compute_origin_access(origin_island_ids, island_function_map,
                                       all_functions=["health", "education"])
        assert "education" in result.columns
        assert result.loc[1, "education"] == False


# ---------------------------------------------------------------------------
# Tests — Layer A: compute_access_matrix_from_origins
# ---------------------------------------------------------------------------

class TestComputeAccessMatrixFromOrigins:

    def test_full_access(self):
        # All origins have health access
        origin_access_df = pd.DataFrame(
            {"health": [True, True, True]},
            index=pd.Index([0, 1, 2], name="origin_id"),
        )
        origin_access_df.columns.name = "function"
        groups = {
            "total": {0: 100, 1: 200, 2: 300},
        }
        matrix = compute_access_matrix_from_origins(origin_access_df, groups)
        assert matrix.loc["health", "total"] == pytest.approx(100.0)

    def test_partial_access_weighted(self):
        # Origin 0 has access (100 people), origin 1 does not (300 people)
        origin_access_df = pd.DataFrame(
            {"health": [True, False]},
            index=pd.Index([0, 1], name="origin_id"),
        )
        origin_access_df.columns.name = "function"
        groups = {
            "total": {0: 100, 1: 300},
        }
        matrix = compute_access_matrix_from_origins(origin_access_df, groups)
        assert matrix.loc["health", "total"] == pytest.approx(25.0)

    def test_missing_origins_give_nan(self):
        origin_access_df = pd.DataFrame(
            {"health": [True]},
            index=pd.Index([99], name="origin_id"),
        )
        origin_access_df.columns.name = "function"
        # Group references origins not in origin_access_df
        groups = {"total": {0: 100, 1: 200}}
        matrix = compute_access_matrix_from_origins(origin_access_df, groups)
        assert pd.isna(matrix.loc["health", "total"])

    def test_zero_weight_group_gives_nan(self):
        origin_access_df = pd.DataFrame(
            {"health": [True]},
            index=pd.Index([0], name="origin_id"),
        )
        origin_access_df.columns.name = "function"
        groups = {"empty_group": {0: 0}}
        matrix = compute_access_matrix_from_origins(origin_access_df, groups)
        assert pd.isna(matrix.loc["health", "empty_group"])


# ---------------------------------------------------------------------------
# Tests — Layer B (spatial): assign_destinations_to_islands_spatial
# ---------------------------------------------------------------------------

class TestAssignDestinationsToIslandsSpatial:

    def test_hospital_on_island_0_only(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        result = assign_destinations_to_islands_spatial(service_nodes, islands)

        assert "health" in result.get(0, frozenset()), (
            "Island 0 should have 'health' because hospital is inside it"
        )
        assert "health" not in result.get(1, frozenset()), (
            "Island 1 has no hospital, should not have 'health'"
        )

    def test_fire_station_on_island_0(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        result = assign_destinations_to_islands_spatial(service_nodes, islands)
        assert "emergency_response" in result.get(0, frozenset())

    def test_empty_service_nodes_returns_empty(self, two_island_scenario):
        islands, _, _ = two_island_scenario
        empty_nodes = gpd.GeoDataFrame({"type": [], "geometry": []}, crs=CRS)
        result = assign_destinations_to_islands_spatial(empty_nodes, islands)
        assert result == {}

    def test_custom_taxonomy(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        custom_tax = {"hospital": "my_health_cat"}
        result = assign_destinations_to_islands_spatial(
            service_nodes, islands, taxonomy=custom_tax
        )
        assert "my_health_cat" in result.get(0, frozenset())
        assert "emergency_response" not in result.get(0, frozenset())

    def test_unknown_node_types_ignored(self, two_island_scenario):
        islands, _, _ = two_island_scenario
        unknown_nodes = _make_service_nodes({"unknown_thing": [(50, 50)]})
        result = assign_destinations_to_islands_spatial(unknown_nodes, islands)
        for cats in result.values():
            assert len(cats) == 0

    def test_backward_compat_alias(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        r1 = assign_destinations_to_islands_spatial(service_nodes, islands)
        r2 = compute_function_access_per_island(service_nodes, islands)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests — Layer B (spatial): assign_origins_to_islands_spatial
# ---------------------------------------------------------------------------

class TestAssignOriginsToIslandsSpatial:

    def test_cells_assigned_to_correct_islands(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)

        assert "island_id" in result.columns
        assert (result["island_id"] == 0).sum() == 2
        assert (result["island_id"] == 1).sum() == 1

    def test_total_population_preserved(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        assert population["aantal_inwoners"].sum() == result["aantal_inwoners"].sum()

    def test_negative_cbs_values_replaced_with_zero(self, two_island_scenario):
        islands, _, population = two_island_scenario
        population = population.copy()
        population.at[0, "aantal_inwoners"] = -99997
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        assert (result["aantal_inwoners"] >= 0).all()

    def test_backward_compat_alias(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        r1 = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        r2 = join_population_to_islands(population, islands, pop_columns=pop_cols)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Tests — Layer C: compute_access_matrix (spatial path)
# ---------------------------------------------------------------------------

class TestComputeAccessMatrix:

    def test_full_access_when_all_on_connected_island(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        service_nodes2 = _make_service_nodes(
            {"hospital": [(50, 50), (250, 50)]}
        )
        island_function_map = assign_destinations_to_islands_spatial(service_nodes2, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        assert matrix.loc["health", "total"] == pytest.approx(100.0)

    def test_partial_access(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        island_function_map = assign_destinations_to_islands_spatial(service_nodes, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        assert 0 < matrix.loc["health", "total"] < 100

    def test_zero_access_when_no_service_nodes(self, two_island_scenario):
        islands, _, population = two_island_scenario
        island_function_map: dict = {}
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map,
            island_pop,
            pop_columns=POPULATION_GROUP_COLUMNS,
            all_functions=["health"],
        )
        assert matrix.loc["health", "total"] == pytest.approx(0.0)

    def test_access_values_in_0_100_range(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        island_function_map = assign_destinations_to_islands_spatial(service_nodes, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = assign_origins_to_islands_spatial(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        assert ((matrix >= 0) | matrix.isna()).all().all()
        assert ((matrix <= 100) | matrix.isna()).all().all()


# ---------------------------------------------------------------------------
# Tests — Layer C: compute_equity_gaps
# ---------------------------------------------------------------------------

class TestComputeEquityGaps:

    def _make_matrix(self):
        return pd.DataFrame(
            {
                "total": [92.0, 80.0],
                "elderly": [74.0, 60.0],
                "children": [85.0, 75.0],
            },
            index=pd.Index(["health", "education"], name="function"),
        )

    def test_absolute_gap_direction(self):
        matrix = self._make_matrix()
        gaps = compute_equity_gaps(matrix, reference_group="total")
        assert gaps.loc["health", "elderly_absolute_gap"] == pytest.approx(18.0)

    def test_relative_gap_below_one_for_disadvantaged(self):
        matrix = self._make_matrix()
        gaps = compute_equity_gaps(matrix, reference_group="total")
        assert gaps.loc["health", "elderly_relative_gap"] < 1.0

    def test_most_disadvantaged_group_identified(self):
        matrix = self._make_matrix()
        gaps = compute_equity_gaps(matrix, reference_group="total")
        assert gaps.loc["health", "most_disadvantaged_group"] == "elderly"

    def test_invalid_reference_group_raises(self):
        matrix = self._make_matrix()
        with pytest.raises(ValueError, match="Reference group"):
            compute_equity_gaps(matrix, reference_group="nonexistent")

    def test_equal_access_gives_zero_gap(self):
        matrix = pd.DataFrame(
            {"total": [100.0], "elderly": [100.0]},
            index=pd.Index(["health"], name="function"),
        )
        gaps = compute_equity_gaps(matrix, reference_group="total")
        assert gaps.loc["health", "elderly_absolute_gap"] == pytest.approx(0.0)
        assert gaps.loc["health", "elderly_relative_gap"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests — Layer D: analyse_societal_access (spatial path)
# ---------------------------------------------------------------------------

class TestAnalyseSocietalAccessSpatialPath:

    def test_returns_all_expected_keys(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            islands_gdf=islands,
            population_gdf=population,
            service_nodes_gdf=service_nodes,
            pop_columns=POPULATION_GROUP_COLUMNS,
        )
        assert set(result.keys()) == {
            "island_functions",
            "island_population",
            "origin_access",
            "access_matrix",
            "equity_gaps",
        }

    def test_access_matrix_has_function_rows(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            islands_gdf=islands,
            population_gdf=population,
            service_nodes_gdf=service_nodes,
            pop_columns=POPULATION_GROUP_COLUMNS,
        )
        matrix = result["access_matrix"]
        assert "health" in matrix.index
        assert "emergency_response" in matrix.index

    def test_equity_gaps_columns_present(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            islands_gdf=islands,
            population_gdf=population,
            service_nodes_gdf=service_nodes,
            pop_columns=POPULATION_GROUP_COLUMNS,
        )
        gaps = result["equity_gaps"]
        assert "most_disadvantaged_group" in gaps.columns
        assert "max_absolute_gap" in gaps.columns


# ---------------------------------------------------------------------------
# Tests — Layer D: analyse_societal_access (graph-native path)
# ---------------------------------------------------------------------------

class TestAnalyseSocietalAccessGraphPath:

    def _build_scenario(self):
        """
        Graph: nodes 0-1-2 connected (island A), node 3 isolated (island B).
        Hospital on node 1 (island A) → origins on island A have health access.
        Origins: nodes 0, 2, 3.
        """
        G = _make_nx_graph([(0, 1), (1, 2)])
        G.add_node(3)
        destination_nodes = {1: "hospital"}
        # Weights = population
        stakeholder_groups = {
            "total":   {0: 100, 2: 200, 3: 300},
            "elderly": {0: 20, 3: 80},
        }
        return G, destination_nodes, stakeholder_groups

    def test_graph_path_returns_expected_keys(self):
        G, dest, groups = self._build_scenario()
        from shapely.geometry import box as sbox
        dummy_islands = _make_islands({0: sbox(0, 0, 1, 1)})
        result = analyse_societal_access(
            islands_gdf=dummy_islands,
            population_gdf=None,
            graph=G,
            destination_nodes=dest,
            stakeholder_groups=groups,
        )
        assert set(result.keys()) == {
            "island_functions", "island_population", "origin_access",
            "access_matrix", "equity_gaps",
        }

    def test_graph_path_access_by_island_membership(self):
        G, dest, groups = self._build_scenario()
        from shapely.geometry import box as sbox
        dummy_islands = _make_islands({0: sbox(0, 0, 1, 1)})
        result = analyse_societal_access(
            islands_gdf=dummy_islands,
            population_gdf=None,
            graph=G,
            destination_nodes=dest,
            stakeholder_groups=groups,
        )
        matrix = result["access_matrix"]
        # Origins 0, 2 are on island with hospital → 300 accessible out of 600 total
        total_access = matrix.loc["health", "total"]
        assert 0 < total_access < 100

    def test_graph_path_origin_access_df_indexed_by_origin(self):
        G, dest, groups = self._build_scenario()
        from shapely.geometry import box as sbox
        dummy_islands = _make_islands({0: sbox(0, 0, 1, 1)})
        result = analyse_societal_access(
            islands_gdf=dummy_islands,
            population_gdf=None,
            graph=G,
            destination_nodes=dest,
            stakeholder_groups=groups,
        )
        origin_access = result["origin_access"]
        assert origin_access is not None
        assert origin_access.index.name == "origin_id"
        assert "health" in origin_access.columns


# ---------------------------------------------------------------------------
# Tests — taxonomy and constants
# ---------------------------------------------------------------------------

class TestTaxonomyConstants:

    def test_taxonomy_covers_all_planned_node_types(self):
        required = [
            "hospital",
            "fire_station",
            "cooling_centre",
            "water_supply_point",
            "school",
            "repair_depot",
            "emergency_operations_centre",
        ]
        for ntype in required:
            assert ntype in SERVICE_NODE_TAXONOMY, (
                f"'{ntype}' not in SERVICE_NODE_TAXONOMY"
            )

    def test_taxonomy_values_are_non_empty_strings(self):
        for ntype, cat in SERVICE_NODE_TAXONOMY.items():
            assert isinstance(cat, str) and cat

    def test_population_group_columns_has_total(self):
        assert "total" in POPULATION_GROUP_COLUMNS

    def test_population_group_column_names_are_strings(self):
        for label, col in POPULATION_GROUP_COLUMNS.items():
            assert isinstance(label, str) and isinstance(col, str)

