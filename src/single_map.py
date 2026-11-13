"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.adaptation import build_l1_l2_reduction_array
from src.damage_recovery import (
    default_fragility_function,
    update_damage_and_repair,
    with_steepness_overrides,
)
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.caching import _compute_l1_hash, create_island_cache_key, get_asset_centroid_hash
from src.island_analysis import _compute_initial_distribution, _road_exemption_kwargs
from src.road_islands import (
    _load_and_project_graph,
    attach_assets_undisrupted,
    canonical_island_ids,
    compute_island_geodataframe_from_graph,
    edge_removal,
    filter_hazard_graph,
    l1_node_islands,
    label_road_islands,
    road_state_asset_islands,
)
from src.impacts import calculate_cumulative_monetized_impacts_ema
from src.graph_manager import file_fingerprint
from src.societal_access import postprocess_societal_access_from_config
from src.simulation import (
    SimulationState,
    _fragility_exclusions_from_knowledge_graph,
    _update_operational_state,
    build_dependency_runtime,
    combine_per_type_crew_kwargs,
    combine_per_type_fragility_steepness_kwargs,
    update_repair_crew_assignment_optimized,
)

BUFFER_M = 20
PROJECTED_CRS = "EPSG:28992"
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def undisrupted_state(assets_gdf, graph_path, l1_area_geojson=None, **exemptions):
    """Asset attachment on the undisrupted network (static per extent).
    Returns ``(access_rfids, boundary_asset_indices, boundary_islands_rfids)``."""
    roads = compute_island_geodataframe_from_graph(
        graph_path, hazard_threshold=np.inf, hazard_column="__undisrupted__",
        buffer_distance=BUFFER_M, l1_area_geojson=l1_area_geojson, **exemptions,
    )
    return attach_assets_undisrupted(assets_gdf.to_crs(PROJECTED_CRS), roads.to_crs(PROJECTED_CRS))


def road_island_table(graph_path, hazard_column, threshold, l1_area_geojson=None,
                      cache=None, **kwargs):
    """Road sections with their island on one map: DataFrame ``rfid, island_id, length_m``.

    Same rows as :func:`compute_island_geodataframe_from_graph` (including its
    drop of duplicate geometries) but without geometry, and island ids are canonical
    (see :func:`canonical_island_ids`), so equal partitions share ids and allocation-cache
    entries. With an L1 file and no footprint, islands are derived from the cached no-L1
    labels of the same map and threshold instead of recomputing connected components.
    ``cache`` is a dict (e.g. the island cache) keyed by map column, threshold, L1 and
    ``kwargs`` (footprint/exemptions), so changing any of them is a miss.
    """
    return _road_islands(graph_path, hazard_column, threshold, l1_area_geojson, cache, kwargs)[0]


def _road_islands(graph_path, hazard_column, threshold, l1, cache, kwargs):
    """``(table, node_labels)``; node labels are kept so an L1 variant can build on them."""
    footprint = kwargs.get("event_footprint")
    key = create_island_cache_key(hazard_column, threshold, "road_table", l1_area_geojson=l1) + (
        f"_graph{file_fingerprint(graph_path)}_" + repr(sorted(
            (k, _compute_l1_hash(footprint) if k == "event_footprint" and v is not None else v)
            for k, v in kwargs.items()
        ))
    )
    if cache is not None and key in cache:
        return cache[key]

    G = _load_and_project_graph(graph_path, from_crs="EPSG:4326", to_crs=PROJECTED_CRS)
    edges, remove, _ = edge_removal(G, threshold, hazard_column, l1_area_geojson=l1, **kwargs)
    if l1 is not None and footprint is None:
        _, baseline = _road_islands(graph_path, hazard_column, threshold, None, cache, kwargs)
        labels = l1_node_islands(G, baseline, threshold, hazard_column, l1, **kwargs)
    else:
        labels = label_road_islands(
            filter_hazard_graph(G, threshold, hazard_column, l1_area_geojson=l1, **kwargs)
        )
    labels = canonical_island_ids(labels)

    rows, seen = [], set()
    for (u, v, _, data), removed in zip(edges, remove):
        if removed:
            continue
        island = labels.get(v, -1)
        geom = data.get("geometry_m")
        dedup = (island, geom.wkb if geom is not None else (u, v))
        if dedup in seen:  # same as compute_island_geodataframe_from_graph
            continue
        seen.add(dedup)
        length = data.get("length")
        rows.append((data["rfid"], island, geom.length if length is None else length))
    table = pd.DataFrame(rows, columns=["rfid", "island_id", "length_m"])
    if cache is not None:
        cache[key] = (table, labels)
    return table, labels


_road_geometry_cache = {}


def _road_geometries(graph_path):
    """rfid -> buffered road geometry (static per graph, built once per process)."""
    geoms = _road_geometry_cache.get(str(graph_path))
    if geoms is None:
        G = _load_and_project_graph(graph_path, from_crs="EPSG:4326", to_crs=PROJECTED_CRS)
        geoms = {
            d["rfid"]: d["geometry_m"].buffer(BUFFER_M, cap_style="square", join_style="mitre")
            for _, _, d in G.edges(data=True) if d.get("geometry_m") is not None
        }
        _road_geometry_cache[str(graph_path)] = geoms
    return geoms


def map_asset_islands(undisrupted, graph_path, hazard_column, hazard_threshold,
                      l1_area_geojson=None, island_cache=None, **kwargs):
    """Asset island ids and island gdf for one map. ``kwargs``: footprint/exemptions.
    ``island_cache``: dict for :func:`road_island_table`."""
    access_rfids, boundary_assets, boundary_rfids = undisrupted
    table = road_island_table(
        graph_path, hazard_column, hazard_threshold, l1_area_geojson, island_cache, **kwargs
    )
    geoms = _road_geometries(graph_path)
    roads = gpd.GeoDataFrame(
        table.assign(geometry=table["rfid"].map(geoms)), geometry="geometry", crs=PROJECTED_CRS
    )
    sizes = roads.groupby("island_id")["length_m"].transform("sum") / 1000.0
    roads["island_size_km"] = sizes
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

    Extraction is cached per (map file, asset set), independent of the policy;
    ``depth_reduction`` (see :func:`adaptation_depth_reduction`) is applied afterwards.
    The entry sits in ``hazard_cache`` under its own key, which includes the map's
    fingerprint (a changed file is a miss) and never collides with multi-map's entries.
    """
    method = config["analysis_config"]["hazard_extraction_method"]
    key = (f"single_map_depths|{method}|{Path(hazard_map).name}|{file_fingerprint(hazard_map)}"
           f"|map{map_counter}|{get_asset_centroid_hash(gdf_assets)}")
    depths = hazard_cache.get(key) if hazard_cache is not None else None
    if depths is None:
        extracted = find_hazard_value_at_points_optimized(
            hazard_map, gdf_assets[["type", "geometry"]].copy(), map_counter,
            extraction_method=method, hazard_cache=None, hazard_dir=config["hazard_dir"], _config=config,
        )
        depths = extracted[f"EV{map_counter}_ma"].fillna(0.0).to_numpy(dtype=float)
        if hazard_cache is not None:
            hazard_cache[key] = depths
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


RETURN_PERIOD_PATTERN = re.compile(r"RP[_-]?(\d+)", re.IGNORECASE)


def return_periods(map_names):
    """Return periods if all maps are the same name except for their ``RP<n>`` (>= 2 distinct), else None.

    ``["flood_RP10", "flood_RP100"]`` -> ``[10, 100]``.
    """
    bases, rps = set(), []
    for name in map_names:
        match = RETURN_PERIOD_PATTERN.search(name)
        if match is None:
            return None
        bases.add(RETURN_PERIOD_PATTERN.sub("RP", name))
        rps.append(int(match.group(1)))
    return rps if len(bases) == 1 and len(set(rps)) >= 2 else None


def annualised(values, rps):
    """Integral of a per-map metric over annual exceedance probability ``1/RP``.

    Trapezoid rule between the rarest and most frequent map only (no extrapolation),
    so it is a lower bound on the full annualised value.
    """
    p = 1.0 / np.asarray(rps, dtype=float)
    order = np.argsort(p)
    return float(trapezoid(np.asarray(values, dtype=float)[order], p[order]))


_extent_state_cache = {}


def _extent_state(gdf_assets, graph_path, config):
    """Per-extent state shared by all experiments: undisrupted attachment and dependency runtime."""
    key = (str(graph_path), file_fingerprint(graph_path), get_asset_centroid_hash(gdf_assets))
    state = _extent_state_cache.get(key)
    if state is None:
        knowledge_graph, edges = build_dependency_runtime(config, gdf_assets)
        state = {
            "undisrupted": undisrupted_state(gdf_assets, graph_path, **_road_exemption_kwargs(config)),
            "knowledge_graph": knowledge_graph,
            "dependency_edges": edges,
            "exclusions": _fragility_exclusions_from_knowledge_graph(config, gdf_assets["type"].to_numpy()),
        }
        _extent_state_cache[key] = state
    return state


MONETARY_CATEGORIES = ["residential", "commercial", "industrial", "transport", "public_sector"]


def single_map_monetary_rate(operational, asset_ids, asset_to_lu):
    """Monetary impact rate [EUR/h] of the non-operational assets, per land-use category and total.

    One timestep through the multi-map cumulative function, so the cumsum is the identity and the
    result is a rate, not an accumulated cost (that would need outage durations).
    """
    rate = {f"monetary_impact_{c}": 0.0 for c in [*MONETARY_CATEGORIES, "total"]}
    if asset_to_lu:
        step = calculate_cumulative_monetized_impacts_ema(
            [{"asset_id": asset_ids, "operational": operational}], asset_to_lu
        ).iloc[0]
        rate.update({k: float(v) for k, v in step.items() if k.startswith("monetary_impact_")})
    return rate


def run_single_map_experiment(*, gdf_assets, hazard_maps, config, graph_path, societal_access_config,
                              flood_threshold=0.2, seed=0, l1_area_geojson=None, l2_asset_geojson=None,
                              repair_crew_assignment_method="random", island_cache=None,
                              hazard_extraction_cache=None, asset_to_lu=None, **kwargs):
    """EMA model function: every map of one experiment (scenario x policy), independent of each other.

    Per-type uncertainties arrive as in multi-map (``repair_crews__<type>``,
    ``fragility_param_k_divby2__<type>``); ``seed`` varies the common random numbers.
    Returns ``{outcome: array of length n_maps}`` (societal metrics plus asset counts and
    ``max_recovery_time_h``). If the maps are return periods of one flood (see
    :func:`return_periods`), each outcome also gets an ``annualised_`` counterpart.
    """
    combine_per_type_crew_kwargs(kwargs)
    combine_per_type_fragility_steepness_kwargs(kwargs)
    n_crews = kwargs.get("number_repair_crews", 1)
    steepness = kwargs.get("fragility_steepness_overrides")
    params = config["recovery_parameters"]
    state = _extent_state(gdf_assets, graph_path, config)
    asset_hash = get_asset_centroid_hash(gdf_assets)
    reduction = adaptation_depth_reduction(gdf_assets, l1_area_geojson, l2_asset_geojson, config)
    asset_ids = np.arange(len(gdf_assets))
    allocation_cache = societal_access_config.get("allocation_cache")

    names, per_map = [], []
    for counter, hazard_map in enumerate(hazard_maps):
        name = Path(hazard_map).stem
        column = f"EV{counter}_ma"
        depths = map_asset_depths(hazard_map, gdf_assets, counter, config, hazard_extraction_cache, reduction)
        ids, roads = map_asset_islands(
            state["undisrupted"], graph_path, column, flood_threshold, l1_area_geojson,
            island_cache, **_road_exemption_kwargs(config),
        )
        draws, duration = map_draws(seed, name, asset_ids, params.get("flood_duration_days"))
        outcome = simulate_single_map(
            depths, gdf_assets["type"], ids, roads, n_crews, draws, params, flood_threshold,
            flood_duration=duration, fragility_steepness_overrides=steepness,
            fragility_exclusions=state["exclusions"], method=repair_crew_assignment_method,
            config=config, knowledge_graph=state["knowledge_graph"],
            dependency_edges=state["dependency_edges"],
        )
        road_state_key = (create_island_cache_key(column, flood_threshold, asset_hash, l1_area_geojson)
                          + f"_graph{file_fingerprint(graph_path)}")
        metrics = single_map_societal_metrics(
            societal_access_config, gdf_assets, outcome, ids, roads, road_state_key, name,
            allocation_cache,
        )
        metrics = {k: v for k, v in metrics.items()
                   if k != "timestep" and isinstance(v, (int, float, np.number))}
        metrics.update(
            flooded_assets=int(outcome["flooded"].sum()),
            non_operational_assets=int((~outcome["operational"]).sum()),
            max_recovery_time_h=float(outcome["recovery_time"].max(initial=0.0)),
            **single_map_monetary_rate(outcome["operational"], asset_ids, asset_to_lu),
        )
        names.append(name)
        per_map.append(metrics)

    results = {key: np.array([m[key] for m in per_map], dtype=float) for key in per_map[0]}
    rps = return_periods(names)
    if rps is not None:
        results.update({f"annualised_{k}": np.array([annualised(v, rps)]) for k, v in list(results.items())})
    return results
