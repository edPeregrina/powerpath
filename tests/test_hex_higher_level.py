"""
Tests for the hierarchical-station ring-lookup functions in
grid_based_accessibility_hex.py.

These tests use simple square-cell grids (4 × 4) to keep geometry trivial and
avoid any dependency on geohexgrid, RA2CE, or external data.  The adjacency
relationships mirror what you would see in a real hex grid:

    0  1  2  3
    4  5  6  7
    8  9  10 11
   12 13  14 15

Each cell is a 10 × 10-unit square; adjacent cells share an edge (touch).
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from src.grid_based_accessibility_hex import (
    build_hex_adjacency_graph,
    higher_level_station_within_rings,
)


def _make_grid_4x4(cell_size: float = 10.0) -> gpd.GeoDataFrame:
    """Return a 4 × 4 square-cell grid in EPSG:28992."""
    cells = []
    for row in range(4):
        for col in range(4):
            minx = col * cell_size
            miny = row * cell_size
            cells.append(
                box(minx, miny, minx + cell_size, miny + cell_size)
            )
    return gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:28992")


def _assets_at(*cell_indices, asset_type: str = "ls", cell_size: float = 10.0) -> gpd.GeoDataFrame:
    """Create a point-asset GeoDataFrame whose points sit at the centre of the
    specified grid cells (0-indexed, row-major order in the 4 × 4 grid)."""
    points = []
    for idx in cell_indices:
        row, col = divmod(idx, 4)
        cx = col * cell_size + cell_size / 2
        cy = row * cell_size + cell_size / 2
        points.append(Point(cx, cy))
    return gpd.GeoDataFrame(
        {"type": [asset_type] * len(cell_indices), "geometry": points},
        crs="EPSG:28992",
    )


# ---------------------------------------------------------------------------
# build_hex_adjacency_graph
# ---------------------------------------------------------------------------

class TestBuildHexAdjacencyGraph:
    def test_node_count_matches_cells(self):
        grid = _make_grid_4x4()
        G = build_hex_adjacency_graph(grid)
        assert G.number_of_nodes() == 16

    def test_corner_cell_has_two_neighbours(self):
        """Cell 0 (top-left corner of the 4×4 grid) touches cell 1 and cell 4."""
        grid = _make_grid_4x4()
        G = build_hex_adjacency_graph(grid)
        # Cell 0 should be connected to cells 1 (right) and 4 (below)
        assert G.degree(0) == 2

    def test_edge_cell_has_three_neighbours(self):
        """Cell 1 (top edge, not corner) touches cells 0, 2, and 5."""
        grid = _make_grid_4x4()
        G = build_hex_adjacency_graph(grid)
        assert G.degree(1) == 3

    def test_interior_cell_has_four_neighbours(self):
        """Cell 5 (interior) touches cells 1, 4, 6, and 9."""
        grid = _make_grid_4x4()
        G = build_hex_adjacency_graph(grid)
        assert G.degree(5) == 4

    def test_symmetry(self):
        grid = _make_grid_4x4()
        G = build_hex_adjacency_graph(grid)
        assert G.has_edge(0, 1) == G.has_edge(1, 0)
        assert G.has_edge(5, 9) == G.has_edge(9, 5)


# ---------------------------------------------------------------------------
# higher_level_station_within_rings
# ---------------------------------------------------------------------------

class TestHigherLevelStationWithinRings:
    def test_station_in_own_cell_gives_ring_zero(self):
        """When a cell hosts a target station it should report ring distance 0."""
        grid = _make_grid_4x4()
        assets = _assets_at(5, asset_type="ls")
        result = higher_level_station_within_rings(
            grid, assets, target_types=["ls"], max_rings=3
        )
        assert result.loc[5, "higher_level_rings"] == 0

    def test_immediate_neighbour_gives_ring_one(self):
        """Cells touching a cell with a station should report ring 1."""
        grid = _make_grid_4x4()
        assets = _assets_at(5, asset_type="ls")
        result = higher_level_station_within_rings(
            grid, assets, target_types=["ls"], max_rings=3
        )
        for nb in [1, 4, 6, 9]:  # the four neighbours of cell 5
            assert result.loc[nb, "higher_level_rings"] == 1, (
                f"cell {nb} should be ring 1 from cell 5"
            )

    def test_two_rings_away_gives_ring_two(self):
        """Cell 0 is 2 hops from cell 5 via cell 1 or cell 4."""
        grid = _make_grid_4x4()
        assets = _assets_at(5, asset_type="ls")
        result = higher_level_station_within_rings(
            grid, assets, target_types=["ls"], max_rings=3
        )
        assert result.loc[0, "higher_level_rings"] == 2

    def test_beyond_max_rings_gives_minus_one(self):
        """Cells beyond max_rings should be -1."""
        grid = _make_grid_4x4()
        # Station only at cell 0 (top-left corner)
        assets = _assets_at(0, asset_type="hs")
        result = higher_level_station_within_rings(
            grid, assets, target_types=["hs"], max_rings=1
        )
        # Cell 15 is at least 6 hops from cell 0
        assert result.loc[15, "higher_level_rings"] == -1

    def test_no_target_assets_all_minus_one(self):
        """When no assets match target_types every cell should be -1."""
        grid = _make_grid_4x4()
        assets = _assets_at(5, asset_type="msls")  # msls, not ls/hs
        result = higher_level_station_within_rings(
            grid, assets, target_types=["ls", "hs"], max_rings=3
        )
        assert (result["higher_level_rings"] == -1).all()

    def test_ls_but_not_ms_matches_only_ls(self):
        """Only the specified target types contribute."""
        grid = _make_grid_4x4()
        # ls at cell 5, ms at cell 10 – querying only for "ls"
        ls_assets = _assets_at(5, asset_type="ls")
        ms_assets = _assets_at(10, asset_type="ms")
        import pandas as pd
        all_assets = gpd.GeoDataFrame(
            pd.concat([ls_assets, ms_assets], ignore_index=True),
            crs="EPSG:28992",
        )
        result = higher_level_station_within_rings(
            grid, all_assets, target_types=["ls"], max_rings=3
        )
        # Cell 10 hosts an ms station, which is NOT in target_types — it must
        # be measured relative to the ls station at cell 5, not 0.
        # Cell 5 has ring 0; cell 10 is 3 hops away (5→6→10 or 5→9→10).
        assert result.loc[5, "higher_level_rings"] == 0
        assert result.loc[10, "higher_level_rings"] >= 2

    def test_reuses_supplied_adjacency_graph(self):
        """Supplying a pre-built graph should give identical results."""
        grid = _make_grid_4x4()
        assets = _assets_at(3, asset_type="hs")
        G = build_hex_adjacency_graph(grid)
        result_prebuilt = higher_level_station_within_rings(
            grid, assets, target_types=["hs"], max_rings=4,
            adjacency_graph=G,
        )
        result_auto = higher_level_station_within_rings(
            grid, assets, target_types=["hs"], max_rings=4,
        )
        assert result_prebuilt["higher_level_rings"].equals(
            result_auto["higher_level_rings"]
        )

    def test_result_column_name_is_configurable(self):
        grid = _make_grid_4x4()
        assets = _assets_at(0, asset_type="hs")
        result = higher_level_station_within_rings(
            grid, assets, target_types=["hs"], max_rings=2,
            result_column="hs_ring_distance",
        )
        assert "hs_ring_distance" in result.columns
        assert "higher_level_rings" not in result.columns

    def test_multiple_stations_reduce_distances(self):
        """Two stations should shorten the ring distance for distant cells."""
        grid = _make_grid_4x4()
        # Stations at cells 0 and 15 (opposite corners)
        assets = gpd.GeoDataFrame(
            {
                "type": ["hs", "hs"],
                "geometry": [
                    Point(5, 5),   # centre of cell 0
                    Point(35, 35), # centre of cell 15
                ],
            },
            crs="EPSG:28992",
        )
        result_two = higher_level_station_within_rings(
            grid, assets, target_types=["hs"], max_rings=6
        )
        result_one = higher_level_station_within_rings(
            grid, _assets_at(0, asset_type="hs"), target_types=["hs"], max_rings=6
        )
        # Cell 15 should be closer with two stations (ring 0) than with one
        assert result_two.loc[15, "higher_level_rings"] == 0
        assert result_one.loc[15, "higher_level_rings"] > 0

    def test_crs_mismatch_is_handled(self):
        """Assets in a different CRS should be reprojected transparently."""
        grid = _make_grid_4x4()  # EPSG:28992
        # Place asset at the centre of cell 5 projected to 4326
        asset_28992 = _assets_at(5, asset_type="ls")
        asset_4326 = asset_28992.to_crs("EPSG:4326")
        result = higher_level_station_within_rings(
            grid, asset_4326, target_types=["ls"], max_rings=3
        )
        assert result.loc[5, "higher_level_rings"] == 0
