"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import numpy as np

from src.damage_recovery import _build_failure_probability, resolve_fragility_model
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


def failure_probability(depth, asset_type, k, fragility_models=None):
    """Daily (peak-map) failure probability per asset; ``k`` fixed so draws are comparable."""
    depth = np.asarray(depth, dtype=float)
    asset_type = np.asarray(asset_type)
    prob = np.zeros_like(depth)
    for name in np.unique(asset_type):
        mask = asset_type == name
        model = resolve_fragility_model(name, fragility_models)
        prob[mask] = _build_failure_probability(depth[mask], model, k=k, major_timestep=24)
    return prob


def operational_status(prob, uniforms):
    """1 operational / 0 failed. Common random numbers: same ``uniforms`` across policies."""
    return (np.asarray(uniforms) >= prob).astype(int)
