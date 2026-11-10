"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import numpy as np

from src.road_islands import (
    attach_assets_undisrupted,
    compute_island_geodataframe_from_graph,
    road_state_asset_islands,
)

BUFFER_M = 20
PROJECTED_CRS = "EPSG:28992"


def undisrupted_state(assets_gdf, graph_path, l1_area_geojson=None, **exemptions):
    """Asset attachment on the undisrupted network (static per extent).
    Returns ``(access_rfids, boundary_asset_indices, boundary_islands_rfids)``."""
    roads = compute_island_geodataframe_from_graph(
        graph_path, hazard_threshold=np.inf, hazard_column="__undisrupted__",
        buffer_distance=BUFFER_M, l1_area_geojson=l1_area_geojson, **exemptions,
    )
    return attach_assets_undisrupted(assets_gdf.to_crs(PROJECTED_CRS), roads.to_crs(PROJECTED_CRS))


def map_asset_islands(undisrupted, graph_path, hazard_column, hazard_threshold,
                      l1_area_geojson=None, **kwargs):
    """Asset island ids and island gdf for one map. ``kwargs``: footprint/exemptions."""
    access_rfids, boundary_assets, boundary_rfids = undisrupted
    roads = compute_island_geodataframe_from_graph(
        graph_path, hazard_threshold=hazard_threshold, hazard_column=hazard_column,
        buffer_distance=BUFFER_M, l1_area_geojson=l1_area_geojson, **kwargs,
    )
    ids, _, roads = road_state_asset_islands(roads, access_rfids, boundary_assets, boundary_rfids)
    return ids, roads
