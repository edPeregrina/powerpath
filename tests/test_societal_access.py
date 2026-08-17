"""Scenario tests for src/societal_access.py.

Each test builds minimal synthetic GeoDataFrames and verifies that the
access-matrix and equity-gap computations produce the expected numerical
results.  No real data files are required.
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
    compute_access_matrix,
    compute_equity_gaps,
    compute_function_access_per_island,
    join_population_to_islands,
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
    """Create population grid cells.

    cells: list of dicts with keys 'geometry', 'aantal_inwoners',
    'aantal_inwoners_65_jaar_en_ouder', 'aantal_inwoners_0_tot_15_jaar',
    'aantal_inwoners_25_tot_45_jaar'.
    """
    return gpd.GeoDataFrame(cells, crs=CRS)


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
# Tests – compute_function_access_per_island
# ---------------------------------------------------------------------------

class TestComputeFunctionAccessPerIsland:

    def test_hospital_on_island_0_only(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        result = compute_function_access_per_island(service_nodes, islands)

        assert "health" in result.get(0, frozenset()), (
            "Island 0 should have 'health' because hospital is inside it"
        )
        assert "health" not in result.get(1, frozenset()), (
            "Island 1 has no hospital, should not have 'health'"
        )

    def test_fire_station_on_island_0(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        result = compute_function_access_per_island(service_nodes, islands)
        assert "emergency_response" in result.get(0, frozenset())

    def test_empty_service_nodes_returns_empty(self, two_island_scenario):
        islands, _, _ = two_island_scenario
        empty_nodes = gpd.GeoDataFrame({"type": [], "geometry": []}, crs=CRS)
        result = compute_function_access_per_island(empty_nodes, islands)
        assert result == {}

    def test_custom_taxonomy(self, two_island_scenario):
        islands, service_nodes, _ = two_island_scenario
        custom_tax = {"hospital": "my_health_cat"}
        result = compute_function_access_per_island(
            service_nodes, islands, taxonomy=custom_tax
        )
        assert "my_health_cat" in result.get(0, frozenset())
        # fire_station not in custom taxonomy → should not appear
        assert "emergency_response" not in result.get(0, frozenset())

    def test_unknown_node_types_ignored(self, two_island_scenario):
        islands, _, _ = two_island_scenario
        unknown_nodes = _make_service_nodes({"unknown_thing": [(50, 50)]})
        result = compute_function_access_per_island(unknown_nodes, islands)
        # Should not crash; unknown types simply produce no function entries
        for cats in result.values():
            assert len(cats) == 0


# ---------------------------------------------------------------------------
# Tests – join_population_to_islands
# ---------------------------------------------------------------------------

class TestJoinPopulationToIslands:

    def test_cells_assigned_to_correct_islands(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = join_population_to_islands(population, islands, pop_columns=pop_cols)

        assert "island_id" in result.columns
        # Two cells inside island 0
        assert (result["island_id"] == 0).sum() == 2
        # One cell inside island 1
        assert (result["island_id"] == 1).sum() == 1

    def test_total_population_preserved(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = join_population_to_islands(population, islands, pop_columns=pop_cols)
        original_total = population["aantal_inwoners"].sum()
        result_total = result["aantal_inwoners"].sum()
        assert original_total == result_total

    def test_negative_cbs_values_replaced_with_zero(self, two_island_scenario):
        islands, _, population = two_island_scenario
        # Simulate CBS suppression codes
        population = population.copy()
        population.at[0, "aantal_inwoners"] = -99997
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        result = join_population_to_islands(population, islands, pop_columns=pop_cols)
        assert (result["aantal_inwoners"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests – compute_access_matrix
# ---------------------------------------------------------------------------

class TestComputeAccessMatrix:

    def test_full_access_when_all_on_connected_island(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        # Override: put hospital on BOTH islands
        service_nodes2 = _make_service_nodes(
            {"hospital": [(50, 50), (250, 50)]}
        )
        island_function_map = compute_function_access_per_island(service_nodes2, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = join_population_to_islands(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        assert matrix.loc["health", "total"] == pytest.approx(100.0)

    def test_partial_access(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        island_function_map = compute_function_access_per_island(service_nodes, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = join_population_to_islands(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        # Health access: only island 0 (400 out of 700 people)
        assert 0 < matrix.loc["health", "total"] < 100

    def test_zero_access_when_no_service_nodes(self, two_island_scenario):
        islands, _, population = two_island_scenario
        # No service nodes → empty function map
        island_function_map: dict = {}
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = join_population_to_islands(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map,
            island_pop,
            pop_columns=POPULATION_GROUP_COLUMNS,
            all_functions=["health"],
        )
        assert matrix.loc["health", "total"] == pytest.approx(0.0)

    def test_access_values_in_0_100_range(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        island_function_map = compute_function_access_per_island(service_nodes, islands)
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = join_population_to_islands(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            island_function_map, island_pop, pop_columns=POPULATION_GROUP_COLUMNS
        )
        assert ((matrix >= 0) | matrix.isna()).all().all()
        assert ((matrix <= 100) | matrix.isna()).all().all()

    def test_empty_function_map_with_explicit_functions(self, two_island_scenario):
        islands, _, population = two_island_scenario
        pop_cols = list(POPULATION_GROUP_COLUMNS.values())
        island_pop = join_population_to_islands(population, islands, pop_columns=pop_cols)
        matrix = compute_access_matrix(
            {},
            island_pop,
            pop_columns=POPULATION_GROUP_COLUMNS,
            all_functions=["health", "education"],
        )
        assert set(matrix.index.tolist()) == {"health", "education"}
        assert (matrix == 0).all().all()


# ---------------------------------------------------------------------------
# Tests – compute_equity_gaps
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
        # Elderly access < total → positive gap
        assert gaps.loc["health", "elderly_absolute_gap"] == pytest.approx(18.0)

    def test_relative_gap_below_one_for_disadvantaged(self):
        matrix = self._make_matrix()
        gaps = compute_equity_gaps(matrix, reference_group="total")
        assert gaps.loc["health", "elderly_relative_gap"] < 1.0

    def test_most_disadvantaged_group_identified(self):
        matrix = self._make_matrix()
        gaps = compute_equity_gaps(matrix, reference_group="total")
        # elderly gap = 18; children gap = 7 → elderly is most disadvantaged
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
# Tests – analyse_societal_access (integration)
# ---------------------------------------------------------------------------

class TestAnalyseSocietalAccess:

    def test_returns_all_expected_keys(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            service_nodes,
            islands,
            population,
            pop_columns=POPULATION_GROUP_COLUMNS,
        )
        assert set(result.keys()) == {
            "island_functions",
            "island_population",
            "access_matrix",
            "equity_gaps",
        }

    def test_access_matrix_has_function_rows(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            service_nodes, islands, population, pop_columns=POPULATION_GROUP_COLUMNS
        )
        matrix = result["access_matrix"]
        assert "health" in matrix.index
        assert "emergency_response" in matrix.index

    def test_equity_gaps_columns_present(self, two_island_scenario):
        islands, service_nodes, population = two_island_scenario
        result = analyse_societal_access(
            service_nodes, islands, population, pop_columns=POPULATION_GROUP_COLUMNS
        )
        gaps = result["equity_gaps"]
        assert "most_disadvantaged_group" in gaps.columns
        assert "max_absolute_gap" in gaps.columns


# ---------------------------------------------------------------------------
# Tests – SERVICE_NODE_TAXONOMY and POPULATION_GROUP_COLUMNS constants
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
            assert isinstance(cat, str) and cat, (
                f"Taxonomy value for '{ntype}' must be a non-empty string"
            )

    def test_population_group_columns_has_total(self):
        assert "total" in POPULATION_GROUP_COLUMNS

    def test_population_group_column_names_are_strings(self):
        for label, col in POPULATION_GROUP_COLUMNS.items():
            assert isinstance(label, str) and isinstance(col, str)
