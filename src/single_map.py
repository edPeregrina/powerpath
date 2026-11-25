"""Single-map mode: one independent hazard map, peak depth, no time stepping."""
import hashlib
import heapq
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.adaptation import blocking_reason_codes, build_l1_l2_reduction_array
from src.damage_recovery import (
    default_fragility_function,
    update_damage_and_repair,
    with_steepness_overrides,
)
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.caching import _compute_l1_hash, create_island_cache_key
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
from src.impacts import MONETARY_CATEGORIES, calculate_cumulative_monetized_impacts_ema
from src.graph_manager import file_fingerprint
from src.societal_access import postprocess_societal_access_from_config
from src.simulation import (
    SimulationState,
    _fragility_exclusions_from_knowledge_graph,
    _assign_repair_crews,
    _build_crew_pools,
    _normalize_number_repair_crews_config,
    _update_operational_state,
    _update_unreachable_assets,
    build_dependency_runtime,
    combine_per_type_crew_kwargs,
    combine_per_type_fragility_steepness_kwargs,
    repair_impact_map,
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
    if hazard_column == "__undisrupted__" and threshold == np.inf:
        edges = list(
            G.edges(keys=True, data=True) if G.is_multigraph()
            else ((u, v, None, data) for u, v, data in G.edges(data=True))
        )
        remove = np.zeros(len(edges), dtype=bool)
        labels = label_road_islands(G)
    elif l1 is not None and footprint is None:
        edges, remove, _ = edge_removal(G, threshold, hazard_column, l1_area_geojson=l1, **kwargs)
        _, baseline = _road_islands(graph_path, hazard_column, threshold, None, cache, kwargs)
        labels = l1_node_islands(G, baseline, threshold, hazard_column, l1, **kwargs)
    else:
        edges, remove, _ = edge_removal(G, threshold, hazard_column, l1_area_geojson=l1, **kwargs)
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
    """rfid -> (centreline, buffered geometry); static per graph, built once per process."""
    geoms = _road_geometry_cache.get(str(graph_path))
    if geoms is None:
        G = _load_and_project_graph(graph_path, from_crs="EPSG:4326", to_crs=PROJECTED_CRS)
        geoms = {}
        for u, v, d in G.edges(data=True):
            line = d.get("geometry_m")
            if line is None:  # endpoint fallback, as in compute_island_geodataframe_from_graph
                line = LineString([(G.nodes[u]["x_m"], G.nodes[u]["y_m"]), (G.nodes[v]["x_m"], G.nodes[v]["y_m"])])
            geoms[d["rfid"]] = (line, line.buffer(BUFFER_M, cap_style="square", join_style="mitre"))
        _road_geometry_cache[str(graph_path)] = geoms
    return geoms


def island_road_sections(graph_path, hazard_column, threshold, l1_area_geojson=None, island_cache=None,
                         **kwargs):
    """Buffered road sections (GeoDataFrame ``rfid, island_id, length_m, island_size_km``) of one map,
    with the unbuffered line in ``original_geometry`` as in :func:`compute_island_geodataframe_from_graph`."""
    table = road_island_table(
        graph_path, hazard_column, threshold, l1_area_geojson, island_cache, **kwargs
    )
    geoms = _road_geometries(graph_path)
    roads = gpd.GeoDataFrame(
        table.assign(
            geometry=table["rfid"].map(lambda r: geoms[r][1] if r in geoms else None),
            original_geometry=table["rfid"].map(lambda r: geoms[r][0] if r in geoms else None),
        ),
        geometry="geometry", crs=PROJECTED_CRS,
    )
    roads["island_size_km"] = roads.groupby("island_id")["length_m"].transform("sum") / 1000.0
    return roads


def road_peak_depths(graph_path, hazard_column):
    """rfid -> peak flood depth [m] of the road section on one map (max over duplicate edges)."""
    G = _load_and_project_graph(graph_path, from_crs="EPSG:4326", to_crs=PROJECTED_CRS)
    depths = {}
    for _, _, data in G.edges(data=True):
        value = data.get(hazard_column)
        if value is not None:
            depths[data["rfid"]] = max(depths.get(data["rfid"], 0.0), float(value))
    return depths


def map_asset_islands(undisrupted, graph_path, hazard_column, hazard_threshold,
                      l1_area_geojson=None, island_cache=None, **kwargs):
    """Asset island ids and island gdf for one map. ``kwargs``: footprint/exemptions.
    ``island_cache``: dict for :func:`road_island_table`."""
    access_rfids, boundary_assets, boundary_rfids = undisrupted
    roads = island_road_sections(
        graph_path, hazard_column, hazard_threshold, l1_area_geojson, island_cache, **kwargs
    )
    ids, _, roads = road_state_asset_islands(roads, access_rfids, boundary_assets, boundary_rfids)
    return ids, roads


def single_map_asset_hash(gdf_assets):
    """Fingerprint of the asset set: CRS, types and geometries in order.

    Single-map cache entries are positional per asset, so unlike
    :func:`src.caching.get_asset_centroid_hash` (count + bbox, kept for multi-map's
    existing keys) a reordered or edited asset set must give a different key.
    """
    digest = hashlib.md5(str(gdf_assets.crs).encode())
    digest.update("\0".join(map(str, gdf_assets["type"])).encode())
    for wkb in gdf_assets.geometry.to_wkb():
        digest.update(wkb)
    return f"n{len(gdf_assets)}_{digest.hexdigest()}"


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
           f"|map{map_counter}|{single_map_asset_hash(gdf_assets)}")
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


def queued_repair_finish(repair_time, island_ids, assigned, crew_pools, order_key):
    """Hours after the flood until each asset's repair ends, with a limited crew queue.

    ``crew_pools``: ``(asset mask, {island_id: crews})`` per pool. Within each pool and island,
    crews take assets in order (``assigned`` first, then ascending ``order_key``); each next
    asset goes to the crew that frees up first. ``inf`` if no crew can reach the asset;
    0 if it needs no repair.
    """
    finish = np.where(repair_time > 0, np.inf, 0.0)
    pending = (repair_time > 0) & (island_ids != -1)
    order = np.lexsort((order_key, ~assigned))
    for mask, island_crews in crew_pools:
        for island_id, crews in island_crews.items():
            if crews <= 0:
                continue
            queue = order[(pending & mask & (island_ids == island_id))[order]]
            free_at = [0.0] * int(crews)
            for i in queue:
                end = heapq.heappop(free_at) + repair_time[i]
                finish[i] = end
                heapq.heappush(free_at, end)
    return finish


def simulate_single_map(hazard_values, asset_type, island_ids, roads, n_crews, random_values,
                        recovery_parameters, flood_threshold, flood_duration=0.0,
                        fragility_k=None, fragility_steepness_overrides=None,
                        fragility_exclusions=None, method="random",
                        config=None, knowledge_graph=None, dependency_edges=None, asset_impact_map=None):
    """Peak-state outcome for one map via the shared multi-map building blocks (no time stepping).

    ``random_values``: one uniform draw per asset, reused across policies (common random numbers).
    Fragility is configured as in multi-map: ``recovery_parameters["fragility_models"]``,
    per-type ``fragility_steepness_overrides`` and ``fragility_exclusions`` (see
    ``simulation._fragility_exclusions_from_knowledge_graph``).
    With ``config`` (plus ``knowledge_graph``/``dependency_edges`` from
    ``simulation.build_dependency_runtime``), ``operational`` also applies hazard blocking
    (e.g. flooded hospitals) and dependencies; otherwise it equals ``intrinsic_operational``.
    ``recovery_time`` = flood duration + wait for a crew + ``repair_time`` if not intrinsically
    operational (see :func:`queued_repair_finish`; ``inf`` if no crew can reach it), flood
    duration alone if only blocked while flooded, else 0 (dependency waits excluded).
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
    rfid_lengths = dict(zip(roads["rfid"], roads["length_m"]))

    def initial_distribution(count):
        return count if isinstance(count, dict) else _compute_initial_distribution(
            count, rfids_islands, rfid_lengths)[0]

    # n_crews: scalar, or per-type dict as built from the per-type crew uncertainties (see multi-map)
    crews = _normalize_number_repair_crews_config(n_crews)
    available_repair_crews = initial_distribution(crews["default"])
    crew_pools = _build_crew_pools(crews["typed"])
    for pool in (crew_pools or {"pools": []})["pools"]:
        pool["available"] = initial_distribution(pool["available"])
    typed_types = list((crew_pools or {"asset_type_to_pool": {}})["asset_type_to_pool"])
    queue_pools = [(~np.isin(asset_type, typed_types), dict(available_repair_crews))] + [
        (np.isin(asset_type, list(pool["asset_types"])), dict(pool["available"]))
        for pool in (crew_pools or {"pools": []})["pools"]
    ]
    available_repair_crews, repair_crews_assigned = _assign_repair_crews(
        timestep=0,
        available_repair_crews=available_repair_crews,
        repair_crews_assigned=np.zeros(len(hazard_values), dtype=bool),
        accessible=island_ids != -1,
        flooded_mask=np.zeros(len(hazard_values), dtype=bool),
        repair_time=repair_time,
        island_ids=island_ids,
        method=method,
        verbose=False,
        asset_type=asset_type,
        crew_pools=crew_pools,
        asset_impact_map=asset_impact_map,
    )
    method_name = str(method)
    if "lowest" in method_name:
        order_key = repair_time
    elif "highest" in method_name:
        order_key = -repair_time
    elif "impact" in method_name and asset_impact_map:
        order_key = -np.array([asset_impact_map.get(i, 0) for i in range(len(hazard_values))], dtype=float)
    else:
        order_key = np.random.random(len(hazard_values))  # random, as multi-map's shuffle
    repair_finish = queued_repair_finish(
        repair_time, island_ids, repair_crews_assigned, queue_pools, order_key)

    # Hazard blocking and dependencies: the multi-map evaluator, run once on the peak state.
    operational = intrinsic_operational
    hazard_available = dependency_available = np.ones(len(hazard_values), dtype=bool)
    if config is not None:
        state = SimulationState(None, len(hazard_values))
        state.intrinsic_operational = intrinsic_operational.copy()
        state.recovery_wait_vectors["repair_time"] = repair_time.copy()
        _update_operational_state(
            state, asset_type, flooded, config, recovery_parameters.get("repair_threshold", 2.0),
            knowledge_graph=knowledge_graph, dependency_edges=dependency_edges,
        )
        operational = state.operational
        if state.hazard_available is not None:
            hazard_available = state.hazard_available
        if state.dependency_available is not None:
            dependency_available = state.dependency_available

    unreachable_state = SimpleNamespace(
        island_ids=island_ids, damage_ratio=damage_ratio, repair_crews_assigned=repair_crews_assigned,
    )
    _update_unreachable_assets(
        unreachable_state, available_repair_crews, flooded, recovery_parameters.get("damage_threshold", 0.001),
        asset_type=asset_type, crew_pools=crew_pools,
    )

    return {
        "flooded": flooded,
        "intrinsic_operational": intrinsic_operational,
        "operational": operational,
        "island_id": island_ids,
        "accessible": np.ones(len(hazard_values), dtype=bool),  # as in multi-map, never lowered
        "unreachable": unreachable_state.unreachable,
        "hazard_available": hazard_available,
        "dependency_available": dependency_available,
        "blocking_reason": blocking_reason_codes(
            operational, intrinsic_operational, hazard_available, dependency_available),
        "damage_ratio": damage_ratio,
        "repair_time": repair_time,
        "repair_crews_assigned": repair_crews_assigned,
        "recovery_time": np.where(
            ~intrinsic_operational, repair_finish + flood_duration,
            np.where(~operational & flooded, flood_duration, 0.0),
        ),
    }


def undisrupted_reference(state, graph_path, island_cache=None, **exemptions):
    """``(asset island ids, roads)`` of the undisrupted network, the reference for provider counts.

    Built once per extent (kept in ``state``); no hazard, so threshold is infinite.
    """
    if "capacity_reference_islands" not in state:
        roads = island_road_sections(
            graph_path, "__undisrupted__", np.inf, None, island_cache, **exemptions
        )
        by_rfid = dict(zip(roads["rfid"], roads["island_id"]))
        access_rfids = state["undisrupted"][0]
        ids = np.array([by_rfid.get(rfid, -1) for rfid in access_rfids], dtype=int)
        state["capacity_reference_islands"] = (ids, roads)
    return state["capacity_reference_islands"]


def single_map_societal_metrics(societal_access_config, gdf_assets, outcome, island_ids, roads,
                                road_state_key, map_name="map", allocation_cache=None, reference=None,
                                graph_path=None, undisrupted=None, island_cache=None,
                                **overrides):
    """Societal/population metrics for one map via the shared multi-map postprocessor.

    The map is one timestep (0); uses the outcome's ``operational`` (falls back to
    ``intrinsic_operational``). ``reference`` is ``(island_ids, roads, road_state_key)`` of the
    undisrupted network: it is a preceding timestep that only sets the provider-count reference.
    When omitted and capacity groups are enabled, build the reference from
    ``graph_path`` without removing edges, using ``undisrupted`` attachments
    if supplied. Returns the flat metrics dict.
    """
    if reference is None and overrides.get("capacity_groups", societal_access_config.get("capacity_groups")):
        if graph_path is None:
            raise ValueError("graph_path is required to build the capacity reference when reference is omitted")
        state = {
            "undisrupted": (
                undisrupted if undisrupted is not None
                else undisrupted_state(gdf_assets, graph_path)
            )
        }
        ref_ids, ref_roads = undisrupted_reference(state, graph_path, island_cache)
        ref_key = (
            create_island_cache_key("__undisrupted__", np.inf, single_map_asset_hash(gdf_assets))
            + f"_graph{file_fingerprint(graph_path)}_capacity_reference_v1"
        )
        reference = (ref_ids, ref_roads, ref_key)
    detail = {
        "timestep": 0, "map": map_name, "road_state_key": road_state_key,
        "operational": np.asarray(outcome.get("operational", outcome["intrinsic_operational"])),
        "island_id": np.asarray(island_ids),
    }
    details, summaries, islands = [detail], [{"timestep": 0, "map": map_name}], {road_state_key: roads}
    if reference is not None:
        ref_ids, ref_roads, ref_key = reference
        details.insert(0, {
            "timestep": -1, "map": "undisrupted", "road_state_key": ref_key,
            "operational": np.ones(len(ref_ids), dtype=bool), "island_id": np.asarray(ref_ids),
            "capacity_reference": True,
        })
        summaries.insert(0, {"timestep": -1, "map": "undisrupted"})
        islands[ref_key] = ref_roads
    results, _ = postprocess_societal_access_from_config(
        societal_access_config, summary_results=summaries,
        detailed_results=details, gdf_assets=gdf_assets,
        allocation_cache={} if allocation_cache is None else allocation_cache,
        islands_gdf_cache=islands, **overrides,
    )
    return results[-1]


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
    """Per-extent state shared by all experiments: undisrupted attachment and dependency runtime.

    Keyed on the graph, the ordered asset set, the road exemptions and the dependency rules.
    """
    exemptions = _road_exemption_kwargs(config)
    setup = hashlib.md5(pickle.dumps((sorted(exemptions.items()), config.get("dependency_parameters"))))
    key = (str(graph_path), file_fingerprint(graph_path), single_map_asset_hash(gdf_assets), setup.hexdigest())
    state = _extent_state_cache.get(key)
    if state is None:
        knowledge_graph, edges = build_dependency_runtime(config, gdf_assets)
        state = {
            "undisrupted": undisrupted_state(gdf_assets, graph_path, **exemptions),
            "knowledge_graph": knowledge_graph,
            "dependency_edges": edges,
            "exclusions": _fragility_exclusions_from_knowledge_graph(config, gdf_assets["type"].to_numpy()),
        }
        _extent_state_cache[key] = state
    return state


def single_map_trace_columns(capture, fields):
    """Long-format trace columns (one row per map x asset) from :func:`run_single_map_experiment`'s
    ``capture``; ``timestep`` is the map index, ``map`` its name, as multi-map traces use ``timestep``."""
    n_assets = len(capture[0]["hazard_value"])
    columns = {
        "timestep": np.repeat(np.arange(len(capture)), n_assets),
        "map": np.repeat([c["map"] for c in capture], n_assets),
        "asset_id": np.tile(np.arange(n_assets), len(capture)),
    }
    for field in fields:
        columns[field] = np.concatenate([
            np.asarray(c["hazard_value"] if field == "hazard_value" else c["outcome"][field]) for c in capture
        ])
    return columns


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
                              hazard_extraction_cache=None, asset_to_lu=None, asset_population_map=None,
                              capture=None, **kwargs):
    """EMA model function: every map of one experiment (scenario x policy), independent of each other.

    Per-type uncertainties arrive as in multi-map (``number_repair_crews__<type>``,
    ``fragility_param_k_divby2__<type>``); ``seed`` varies the common random numbers.
    Returns ``{outcome: array of length n_maps}`` (societal metrics plus asset counts and
    ``max_recovery_time_h``). If the maps are return periods of one flood (see
    :func:`return_periods`), each outcome also gets an ``annualised_`` counterpart.
    ``capture`` (a list) receives one dict per map (``map``, ``column``,
    ``hazard_value``, ``outcome``, ``roads``) for spatial-trace export.
    """
    combine_per_type_crew_kwargs(kwargs)
    combine_per_type_fragility_steepness_kwargs(kwargs)
    n_crews = kwargs.get("number_repair_crews", 1)
    steepness = kwargs.get("fragility_steepness_overrides")
    params = config["recovery_parameters"]
    state = _extent_state(gdf_assets, graph_path, config)
    asset_hash = single_map_asset_hash(gdf_assets)
    reduction = adaptation_depth_reduction(gdf_assets, l1_area_geojson, l2_asset_geojson, config)
    asset_ids = np.arange(len(gdf_assets))
    allocation_cache = societal_access_config.get("allocation_cache")
    impact_map = repair_impact_map(repair_crew_assignment_method, asset_to_lu, asset_population_map)
    reference = None
    if societal_access_config.get("capacity_groups"):
        exemptions = _road_exemption_kwargs(config)
        ref_ids, ref_roads = undisrupted_reference(state, graph_path, island_cache, **exemptions)
        reference = (ref_ids, ref_roads,
                     create_island_cache_key("__undisrupted__", np.inf, asset_hash, exemptions=exemptions)
                     + f"_graph{file_fingerprint(graph_path)}_capacity_reference_v1")
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
            dependency_edges=state["dependency_edges"], asset_impact_map=impact_map,
        )
        road_state_key = (create_island_cache_key(column, flood_threshold, asset_hash, l1_area_geojson)
                          + f"_graph{file_fingerprint(graph_path)}")
        metrics = single_map_societal_metrics(
            societal_access_config, gdf_assets, outcome, ids, roads, road_state_key, name,
            allocation_cache, reference=reference,
        )
        metrics = {k: v for k, v in metrics.items()
                   if k != "timestep" and isinstance(v, (int, float, np.number))}
        recovery = outcome["recovery_time"]
        metrics.update(
            flooded_assets=int(outcome["flooded"].sum()),
            non_operational_assets=int((~outcome["operational"]).sum()),
            max_recovery_time_h=float(recovery[np.isfinite(recovery)].max(initial=0.0)),
            unrepairable_assets=int(np.isinf(recovery).sum()),
            **single_map_monetary_rate(outcome["operational"], asset_ids, asset_to_lu),
        )
        names.append(name)
        per_map.append(metrics)
        if capture is not None:
            capture.append({"map": name, "column": column, "hazard_value": depths,
                            "outcome": outcome, "roads": roads})

    results = {key: np.array([m[key] for m in per_map], dtype=float) for key in per_map[0]}
    rps = return_periods(names)
    if rps is not None:
        results.update({f"annualised_{k}": np.array([annualised(v, rps)]) for k, v in list(results.items())})
    return results
