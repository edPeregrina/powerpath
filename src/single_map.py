"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import numpy as np

from src.adaptation import build_l1_l2_reduction_array
from src.damage_recovery import (
    default_fragility_function,
    update_damage_and_repair,
    with_steepness_overrides,
)
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.island_analysis import _compute_initial_distribution
from src.road_islands import (
    attach_assets_undisrupted,
    compute_island_geodataframe_from_graph,
    road_state_asset_islands,
)
from src.societal_access import postprocess_societal_access_from_config
from src.simulation import (
    SimulationState,
    _update_operational_state,
    update_repair_crew_assignment_optimized,
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


def adaptation_depth_reduction(gdf_assets, l1_area_geojson=None, l2_asset_geojson=None, config=None):
    """Per-asset depth reduction (m) from L1 areas and L2 assets, as multi-map applies it.

    Depends only on the policy and the asset set, not on the map, so compute it once per
    policy and reuse it for every map.
    """
    if l1_area_geojson is None and l2_asset_geojson is None:
        return np.zeros(len(gdf_assets))
    reductions = build_l1_l2_reduction_array(
        gdf_assets, l1_area_geojson=l1_area_geojson, l2_asset_geojson=l2_asset_geojson,
        hazard_maps=[None], major_timestep=1, config=config,
    )
    return reductions.sum(axis=2)[0].astype(float)


def map_asset_depths(hazard_map, gdf_assets, map_counter, config, hazard_cache=None, depth_reduction=None):
    """Depth (m) at each asset for one map, extracted as in multi-map.

    Extraction is cached per (map, asset set), independent of the policy;
    ``depth_reduction`` (see :func:`adaptation_depth_reduction`) is applied afterwards.
    """
    extracted = find_hazard_value_at_points_optimized(
        hazard_map, gdf_assets[["type", "geometry"]].copy(), map_counter,
        extraction_method=config["analysis_config"]["hazard_extraction_method"],
        hazard_cache=hazard_cache, hazard_dir=config["hazard_dir"], _config=config,
    )
    depths = extracted[f"EV{map_counter}_ma"].fillna(0.0).to_numpy(dtype=float)
    if depth_reduction is not None:
        depths = np.maximum(0.0, depths - depth_reduction)
    return depths


def common_random_numbers(seed, map_id, asset_ids=None, stream="fragility"):
    """Uniform draws in [0, 1) that depend only on ``(seed, stream, map_id, asset_id)``.

    Each draw is a hash of its own key, so it is the same in every policy and does not change
    if assets or maps are added, removed or reordered. ``asset_ids=None`` returns one draw
    for the map (e.g. ``stream="duration"``).
    """
    import pandas as pd

    ids = [""] if asset_ids is None else list(asset_ids)
    keys = np.array([f"{seed}|{stream}|{map_id}|{i}" for i in ids], dtype=object)
    # hash_array gives a 64-bit integer; keep its top 53 bits (exact in a float) and scale to [0, 1)
    u = (pd.util.hash_array(keys) >> np.uint64(11)).astype(float) / 2.0**53
    return u[0] if asset_ids is None else u


def sample_flood_durations(uniforms, params=None):
    """Flood duration in hours per uniform draw (one per map; reuse across policies).

    Truncated lognormal in days, parameters from ``recovery_parameters["flood_duration_days"]``
    (``min``, ``max``, ``median``, ``sigma``); inverse-CDF so draws stay common random numbers.
    """
    from scipy.stats import norm

    p = {"min": 0.0, "max": 7.0, "median": 1.0, "sigma": 1.0, **(params or {})}
    mu, sigma = np.log(p["median"]), p["sigma"]
    cdf = lambda d: norm.cdf((np.log(d) - mu) / sigma) if d > 0 else 0.0
    lo, hi = cdf(p["min"]), cdf(p["max"])
    q = lo + np.asarray(uniforms, dtype=float) * (hi - lo)
    return np.exp(mu + sigma * norm.ppf(q)) * 24.0


def map_draws(seed, map_id, asset_ids, duration_params=None):
    """``(random_values, flood_duration_hours)`` for one map, from independent hashed streams.

    Feed to :func:`simulate_single_map`; ``seed`` is the EMA parameter that varies scenarios.
    """
    u = common_random_numbers(seed, map_id, asset_ids, stream="fragility")
    duration = sample_flood_durations(common_random_numbers(seed, map_id, stream="duration"), duration_params)
    return u, float(duration)


def simulate_single_map(hazard_values, asset_type, island_ids, roads, n_crews, random_values,
                        recovery_parameters, flood_threshold, flood_duration=0.0,
                        fragility_k=None, fragility_steepness_overrides=None,
                        fragility_exclusions=None, method="random",
                        config=None, knowledge_graph=None, dependency_edges=None):
    """Peak-state outcome for one map via the shared multi-map building blocks (no time stepping).

    ``random_values``: one uniform draw per asset, reused across policies (common random numbers).
    Fragility is configured as in multi-map: ``recovery_parameters["fragility_models"]``,
    per-type ``fragility_steepness_overrides`` and ``fragility_exclusions`` (see
    ``simulation._fragility_exclusions_from_knowledge_graph``).
    With ``config`` (plus ``knowledge_graph``/``dependency_edges`` from
    ``simulation.build_dependency_runtime``), ``operational`` also applies hazard blocking
    (e.g. flooded hospitals) and dependencies; otherwise it equals ``intrinsic_operational``.
    ``recovery_time`` = flood duration + ``repair_time`` if not intrinsically operational,
    flood duration alone if only blocked while flooded, else 0 (dependency waits excluded).
    """
    hazard_values = np.asarray(hazard_values, dtype=float)
    asset_type = np.asarray(asset_type)
    island_ids = np.asarray(island_ids)
    flooded = hazard_values > flood_threshold

    intrinsic_operational = np.ones(len(hazard_values), dtype=bool)
    if flooded.any():
        if isinstance(fragility_exclusions, np.ndarray):
            fragility_exclusions = fragility_exclusions[flooded]
        intrinsic_operational[flooded] = default_fragility_function(
            hazard_values[flooded], asset_type[flooded], k=fragility_k, major_timestep=24,
            fragility_models=with_steepness_overrides(
                recovery_parameters.get("fragility_models"), fragility_steepness_overrides),
            fragility_exclusions=fragility_exclusions,
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

    rfids_islands = dict(zip(roads["rfid"], roads["island_id"]))
    available_repair_crews, _ = _compute_initial_distribution(
        n_crews, rfids_islands, dict(zip(roads["rfid"], roads["length_m"]))
    )
    _, repair_crews_assigned = update_repair_crew_assignment_optimized(
        0, available_repair_crews, np.zeros(len(hazard_values), dtype=bool), island_ids != -1,
        np.zeros(len(hazard_values), dtype=bool), repair_time, island_ids, method=method,
    )

    # Hazard blocking and dependencies: the multi-map evaluator, run once on the peak state.
    operational = intrinsic_operational
    if config is not None:
        state = SimulationState(None, len(hazard_values))
        state.intrinsic_operational = intrinsic_operational.copy()
        state.recovery_wait_vectors["repair_time"] = repair_time.copy()
        _update_operational_state(
            state, asset_type, flooded, config, recovery_parameters.get("repair_threshold", 2.0),
            knowledge_graph=knowledge_graph, dependency_edges=dependency_edges,
        )
        operational = state.operational

    return {
        "flooded": flooded,
        "intrinsic_operational": intrinsic_operational,
        "operational": operational,
        "damage_ratio": damage_ratio,
        "repair_time": repair_time,
        "repair_crews_assigned": repair_crews_assigned,
        "recovery_time": np.where(
            ~intrinsic_operational, repair_time + flood_duration,
            np.where(~operational & flooded, flood_duration, 0.0),
        ),
    }


def single_map_societal_metrics(societal_access_config, gdf_assets, outcome, island_ids, roads,
                                road_state_key, map_name="map", allocation_cache=None, **overrides):
    """Societal/population metrics for one map via the shared multi-map postprocessor.

    The map is one timestep (0); uses the outcome's ``operational`` (falls back to
    ``intrinsic_operational``). Returns the flat metrics dict.
    """
    detail = {
        "timestep": 0, "map": map_name, "road_state_key": road_state_key,
        "operational": np.asarray(outcome.get("operational", outcome["intrinsic_operational"])),
        "island_id": np.asarray(island_ids),
    }
    results, _ = postprocess_societal_access_from_config(
        societal_access_config, summary_results=[{"timestep": 0, "map": map_name}],
        detailed_results=[detail], gdf_assets=gdf_assets,
        allocation_cache={} if allocation_cache is None else allocation_cache,
        islands_gdf_cache={road_state_key: roads}, **overrides,
    )
    return results[0]
