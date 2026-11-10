"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import numpy as np

from src.damage_recovery import default_fragility_function, update_damage_and_repair
from src.island_analysis import _compute_initial_distribution
from src.road_islands import (
    attach_assets_undisrupted,
    compute_island_geodataframe_from_graph,
    road_state_asset_islands,
)
from src.simulation import update_repair_crew_assignment_optimized

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


def simulate_single_map(hazard_values, asset_type, island_ids, roads, n_crews, random_values,
                        recovery_parameters, flood_threshold, flood_duration=0.0,
                        fragility_k=None, fragility_exclusions=None, method="random"):
    """Peak-state outcome for one map via the shared multi-map building blocks (no time stepping).

    ``random_values``: one uniform draw per asset, reused across policies (common random numbers).
    ``recovery_time`` = flood duration + ``repair_time`` for assets that are not intrinsically operational.
    """
    hazard_values = np.asarray(hazard_values, dtype=float)
    asset_type = np.asarray(asset_type)
    island_ids = np.asarray(island_ids)
    flooded = hazard_values > flood_threshold

    intrinsic_operational = np.ones(len(hazard_values), dtype=bool)
    if flooded.any():
        intrinsic_operational[flooded] = default_fragility_function(
            hazard_values[flooded], asset_type[flooded], k=fragility_k, major_timestep=24,
            fragility_models=recovery_parameters.get("fragility_models"),
            fragility_exclusions=None if fragility_exclusions is None else fragility_exclusions[flooded],
            random_values=np.asarray(random_values)[flooded],
        ).astype(bool)

    damage_ratio = np.zeros(len(hazard_values))
    repair_time = np.zeros(len(hazard_values))
    update_damage_and_repair(
        hazard_values, damage_ratio, repair_time, flood_threshold,
        recovery_parameters.get("damage_ratio_coefficients", (0.0468, 0.0077)),
        recovery_parameters.get("repair_time_coefficients", [702.72, 3.14, 1.9891]),
        recovery_parameters.get("repair_threshold", 2.0),
    )
    repair_time = np.where(intrinsic_operational, 0.0, repair_time)

    rfids_islands = dict(zip(roads["rfid"], roads["island_id"]))
    available_repair_crews, _ = _compute_initial_distribution(
        n_crews, rfids_islands, dict(zip(roads["rfid"], roads["length_m"]))
    )
    _, repair_crews_assigned = update_repair_crew_assignment_optimized(
        0, available_repair_crews, np.zeros(len(hazard_values), dtype=bool), island_ids != -1,
        np.zeros(len(hazard_values), dtype=bool), repair_time, island_ids, method=method,
    )
    return {
        "flooded": flooded,
        "intrinsic_operational": intrinsic_operational,
        "damage_ratio": damage_ratio,
        "repair_time": repair_time,
        "repair_crews_assigned": repair_crews_assigned,
        "recovery_time": np.where(intrinsic_operational, 0.0, repair_time + flood_duration),
    }
