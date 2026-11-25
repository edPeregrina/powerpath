import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import get_config  # noqa: E402
from src import island_analysis, single_map  # noqa: E402
from src.damage_recovery import default_fragility_function  # noqa: E402
from tests.l1_fixtures import make_l1_layers  # noqa: E402

HAZARD_DIR = REPO / "data" / "test_samples" / "test_hazard_timesteps"
GRAPH = HAZARD_DIR.parent / "static" / "output_graph" / "base_graph_hazard_editted.p"


@pytest.fixture(scope="module")
def assets():
    G = island_analysis._load_and_project_graph(GRAPH)
    geoms = [d["geometry_m"] for _, _, d in G.edges(data=True) if "geometry_m" in d]
    pts = [g.interpolate(0.5, normalized=True) for g in geoms[::7][:60]]
    types = np.where(np.arange(len(pts)) % 2 == 0, "msls", "ls")
    return gpd.GeoDataFrame({"type": types}, geometry=pts, crs="EPSG:28992")


@pytest.mark.parametrize("column,threshold", [("EV0_ma", 0.2), ("EV2_ma", 0.2), ("EV5_ma", 0.5)])
def test_map_asset_islands_matches_time_explicit(tmp_path, assets, column, threshold):
    config = get_config(root_dir=tmp_path, hazard_dir_override=HAZARD_DIR)
    config["simulation_config"]["verbose"] = False
    config["interim_dir"].mkdir(parents=True, exist_ok=True)
    gdf = assets.copy()
    access, boundary_assets, boundary_rfids, _ = island_analysis.match_assets_access(
        gdf, hazard_threshold=0.2, hazard_column="EV0_ma", config=config
    )
    expected, _ = island_analysis.match_island_ids_assets(
        gdf.assign(access_rfid=access), boundary_assets, boundary_rfids,
        hazard_threshold=threshold, hazard_column=column, config=config,
    )

    undisrupted = single_map.undisrupted_state(assets, GRAPH)
    ids, _ = single_map.map_asset_islands(undisrupted, GRAPH, column, threshold)
    expected = np.asarray(expected)
    # Ids are canonical, so compare the grouping of assets, not the numbers.
    assert _partition(dict(enumerate(ids.tolist()))) == _partition(dict(enumerate(expected.tolist())))


def test_simulate_single_map(assets):
    undisrupted = single_map.undisrupted_state(assets, GRAPH)
    island_ids, roads = single_map.map_asset_islands(undisrupted, GRAPH, "EV2_ma", 0.2)
    hazard = np.linspace(0, 1.5, len(assets))
    params = get_config(root_dir=Path("."), hazard_dir_override=HAZARD_DIR)["recovery_parameters"]

    def run(u, n_crews=5):
        return single_map.simulate_single_map(
            hazard, assets["type"], island_ids, roads, n_crews, u, params,
            flood_threshold=0.2, flood_duration=24.0, fragility_k=6.0,
        )

    all_fail = run(np.zeros(len(assets)))
    assert (all_fail["intrinsic_operational"] == ~all_fail["flooded"]).all()
    assert run(np.ones(len(assets)))["intrinsic_operational"].all()

    assert not all_fail["repair_crews_assigned"][~all_fail["flooded"]].any()
    assert all_fail["repair_crews_assigned"].sum() <= 5
    assert (all_fail["repair_time"][all_fail["flooded"]] > 0).all()
    crewed = all_fail["repair_crews_assigned"]
    assert (all_fail["recovery_time"][crewed] == all_fail["repair_time"][crewed] + 24.0).all()
    waiting = all_fail["flooded"] & ~crewed
    assert waiting.any()
    assert (all_fail["recovery_time"][waiting] > all_fail["repair_time"][waiting] + 24.0).all()

    u = np.random.default_rng(1).random(len(assets))
    assert (run(u)["intrinsic_operational"] == run(u)["intrinsic_operational"]).all()


def test_queued_repair_finish():
    repair = np.array([10.0, 10.0, 10.0, 4.0, 5.0, 0.0, 7.0])
    islands = np.array([1, 1, 1, 1, 2, 1, -1])
    assigned = np.array([False, True, True, False, False, False, False])
    is_a = np.array([True, True, True, False, True, True, True])
    pools = [(is_a, {1: 2, 2: 0}), (~is_a, {1: 1})]
    finish = single_map.queued_repair_finish(repair, islands, assigned, pools, np.arange(7.0))
    # two crews on island 1: assets 1, 2 first, asset 0 waits 10 h; asset 3 has its own pool;
    # island 2 has no crews, island -1 is unreachable, asset 5 needs no repair
    np.testing.assert_array_equal(finish, [20.0, 10.0, 10.0, 4.0, np.inf, 0.0, np.inf])


def test_model_extent_rejects_maps_with_different_bounds(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from config import get_model_extent

    def write(name, x0):
        with rasterio.open(tmp_path / name, "w", driver="GTiff", height=2, width=2, count=1,
                           dtype="float32", crs="EPSG:28992",
                           transform=from_origin(x0, 450000, 10, 10)) as dst:
            dst.write(np.zeros((1, 2, 2), dtype="float32"))

    write("rp10.tif", 100000)
    write("rp100.tif", 100002)  # within half a pixel
    assert get_model_extent({"hazard_dir": tmp_path}).bounds == pytest.approx((100000, 449980, 100020, 450000))
    write("rp1000.tif", 100100)
    with pytest.raises(ValueError, match="differ in extent"):
        get_model_extent({"hazard_dir": tmp_path})


def test_single_map_asset_hash_sees_order_and_edits(assets):
    base = single_map.single_map_asset_hash(assets)
    assert single_map.single_map_asset_hash(assets.copy()) == base
    assert single_map.single_map_asset_hash(assets.iloc[::-1]) != base
    retyped = assets.copy()
    retyped.loc[retyped.index[0], "type"] = "other"
    assert single_map.single_map_asset_hash(retyped) != base


def test_fragility_common_random_numbers(assets):
    depth = np.linspace(0, 1.5, len(assets))
    types = assets["type"].to_numpy()
    u = np.random.default_rng(0).random(len(assets))
    a = default_fragility_function(depth, types, k=6.0, random_values=u)
    assert np.array_equal(a, default_fragility_function(depth, types, k=6.0, random_values=u))
    assert (a[depth == 0] == 1).all()
    assert default_fragility_function(depth, types, k=6.0, random_values=np.zeros(len(u))).tolist() == \
        np.where(depth > 0, 0, 1).tolist()


def test_matches_time_explicit_at_last_map(tmp_path, assets, monkeypatch):
    from src.simulation import (
        _fragility_exclusions_from_knowledge_graph,
        build_dependency_runtime,
        simulate_asset_damage_recovery_access_breakdown,
    )

    assets = assets.copy()
    assets.loc[assets.index[::3], "type"] = "hospital"
    config = get_config(root_dir=tmp_path, hazard_dir_override=HAZARD_DIR)
    config["simulation_config"]["verbose"] = False
    config["interim_dir"].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(np.random, "random", lambda size=None: np.full(size, 0.5))
    n, threshold, k = 3, 0.2, 6.0
    res = simulate_asset_damage_recovery_access_breakdown(
        gdf_assets=assets.copy(), hazard_maps=[HAZARD_DIR / f"Frag_timesteps_test{i}.tif" for i in range(n)],
        number_repair_crews=5, repair_crew_assignment_method="island", flood_threshold=threshold,
        root_dir=tmp_path, config=config, major_timestep=24, timestep_output=True,
        fragility_param_k=k, verbose=False,
    )
    steps = res[0][0][2]
    last = [s for s in steps if s["map"] == n - 1][-1]
    earlier_flooded = np.any([np.asarray(s["flooded"], bool) for s in steps if s["map"] < n - 1], axis=0)
    first_here = np.asarray(last["flooded"], bool) & ~earlier_flooded
    assert first_here.any()

    undisrupted = single_map.undisrupted_state(assets, GRAPH)
    ids, roads = single_map.map_asset_islands(undisrupted, GRAPH, f"EV{n - 1}_ma", threshold)
    assert ids.tolist() == np.asarray(last["island_id"]).tolist()

    params = config["recovery_parameters"]
    kg, edges = build_dependency_runtime(config, assets)
    depths = single_map.map_asset_depths(HAZARD_DIR / f"Frag_timesteps_test{n - 1}.tif", assets, n - 1, config)
    np.testing.assert_allclose(depths, np.asarray(last["hazard_value"], float))
    out = single_map.simulate_single_map(
        depths, assets["type"], ids, roads, 5, np.full(len(assets), 0.5), params,
        flood_threshold=threshold, fragility_k=k,
        fragility_exclusions=_fragility_exclusions_from_knowledge_graph(config, assets["type"].to_numpy()),
        config=config, knowledge_graph=kg, dependency_edges=edges,
    )
    hosp = (assets["type"].to_numpy() == "hospital") & np.asarray(last["flooded"], bool)
    assert hosp.any()
    assert not np.asarray(last["operational"], bool)[hosp].any()
    assert not out["operational"][hosp].any()
    assert out["intrinsic_operational"][hosp].all()
    for key, ref in [("damage_ratio", "damage_ratio"), ("repair_time", "repair_time"),
                     ("intrinsic_operational", "intrinsic_operational"), ("operational", "operational")]:
        np.testing.assert_allclose(
            np.asarray(out[key], float)[first_here], np.asarray(last[ref], float)[first_here], err_msg=key
        )


def test_common_random_numbers():
    ids = np.arange(2000)
    u = single_map.common_random_numbers(7, "mapA", ids)
    assert ((u >= 0) & (u < 1)).all() and abs(u.mean() - 0.5) < 0.03
    assert np.array_equal(u, single_map.common_random_numbers(7, "mapA", ids))
    assert np.array_equal(u[[5, 1, 9]], single_map.common_random_numbers(7, "mapA", ids[[5, 1, 9]]))
    assert not np.array_equal(u, single_map.common_random_numbers(8, "mapA", ids))
    assert not np.array_equal(u, single_map.common_random_numbers(7, "mapB", ids))
    assert not np.array_equal(u, single_map.common_random_numbers(7, "mapA", ids, stream="x"))
    assert 0 <= single_map.common_random_numbers(7, "mapA") < 1


def test_map_draws():
    ids = np.arange(50)
    u, d = single_map.map_draws(3, "m1", ids)
    u2, d2 = single_map.map_draws(3, "m1", ids)
    assert np.array_equal(u, u2) and d == d2 and 0 <= d <= 7 * 24
    assert single_map.map_draws(4, "m1", ids)[1] != d
    assert not np.array_equal(u, single_map.common_random_numbers(3, "m1", ids, stream="duration"))


def test_sample_flood_durations():
    u = np.linspace(0, 1, 101)
    h = single_map.sample_flood_durations(u)
    assert h[0] == 0.0 and 12.0 < h[50] < 24.0
    assert (np.diff(h) >= 0).all() and h[-1] == pytest.approx(7 * 24.0)
    h2 = single_map.sample_flood_durations(u, {"min": 0.5, "max": 3.0, "median": 2.0})
    assert h2[0] == pytest.approx(12.0) and h2[-1] == pytest.approx(72.0)


def test_single_map_fragility_config_as_multi_map(assets):
    from src.simulation import _fragility_exclusions_from_knowledge_graph

    config = get_config(root_dir=Path("."), hazard_dir_override=HAZARD_DIR)
    undisrupted = single_map.undisrupted_state(assets, GRAPH)
    ids, roads = single_map.map_asset_islands(undisrupted, GRAPH, "EV2_ma", 0.2)
    types = np.where(np.arange(len(assets)) % 2 == 0, "msls", "hospital")
    hazard = np.full(len(assets), 1.0)
    excl = _fragility_exclusions_from_knowledge_graph(config, types)

    def run(u, **kw):
        return single_map.simulate_single_map(
            hazard, types, ids, roads, 5, np.full(len(assets), u), config["recovery_parameters"],
            flood_threshold=0.2, fragility_exclusions=excl, **kw,
        )["intrinsic_operational"]

    out = run(0.5, fragility_steepness_overrides={"msls": 50.0})
    assert out[types == "hospital"].all()
    assert not out[types == "msls"].any()
    assert config["recovery_parameters"]["fragility_models"]["msls"].get("steepness") is None


def test_adaptation_depth_reduction_applied_to_map_depths(tmp_path, assets):
    config = get_config(root_dir=tmp_path, hazard_dir_override=HAZARD_DIR)
    config["interim_dir"].mkdir(parents=True, exist_ok=True)
    hazard_map = HAZARD_DIR / "Frag_timesteps_test2.tif"
    base = single_map.map_asset_depths(hazard_map, assets, 2, config)
    l2 = gpd.GeoDataFrame({"depth_red": [0.2]}, geometry=[assets.geometry.iloc[0].buffer(1)], crs=assets.crs)
    red = single_map.adaptation_depth_reduction(assets, l2_asset_geojson=l2)
    assert red[0] == pytest.approx(0.2) and (red[1:][~assets.geometry.iloc[1:].intersects(l2.geometry.iloc[0])] == 0).all()
    adapted = single_map.map_asset_depths(hazard_map, assets, 2, config, depth_reduction=red)
    np.testing.assert_allclose(adapted, np.maximum(0.0, base - red))
    assert not single_map.adaptation_depth_reduction(assets).any()


def test_single_map_societal_metrics_match_postprocessor():
    from shapely.geometry import Point

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_societal_access import _make_islands_gdf, _make_population_gdf

    pop = _make_population_gdf().iloc[:2].copy()
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital", "school", "fire_station"]},
        geometry=[Point(1, 1), Point(12, 1), Point(11, 1)], crs="EPSG:28992",
    )
    sa_cfg = {"pop_grid_gdf": pop, "cell_id_column": "cell_id", "nearest_max_distance": 2.0,
              "all_functions": ["education", "emergency_response", "hospital"]}
    outcome = {"intrinsic_operational": np.array([True, False, True])}
    row = single_map.single_map_societal_metrics(
        sa_cfg, gdf_assets, outcome, np.array([1, 2, 2]), _make_islands_gdf(), road_state_key="0",
    )
    assert row["societal_total_population__total"] == pytest.approx(150.0)
    assert row["societal_access_pct__hospital__total"] == pytest.approx(66.67, abs=0.01)
    assert row["societal_access_pct__education__total"] == pytest.approx(0.0)
    assert row["societal_access_pct__emergency_response__total"] == pytest.approx(33.33, abs=0.01)


L1_FILES = [
    REPO / "data" / "test_samples" / "adaptation" / "l1_adapt_polygon.geojson",
    *make_l1_layers(GRAPH).values(),
]


def _partition(labels):
    groups = {}
    for node, label in labels.items():
        groups.setdefault(label, set()).add(node)
    return {frozenset(g) for g in groups.values()}


def _incremental_vs_full(G, column, threshold, l1):
    from src.road_islands import filter_hazard_graph, label_road_islands, l1_node_islands

    baseline = label_road_islands(filter_hazard_graph(G.copy(), threshold, column))
    incremental = l1_node_islands(G, baseline, threshold, column, l1)
    full = label_road_islands(filter_hazard_graph(G.copy(), threshold, column, l1_area_geojson=l1))
    return baseline, incremental, full


@pytest.mark.parametrize("column,threshold", [("EV2_ma", 0.2), ("EV5_ma", 0.5), ("EV3_ma", 0.0)])
def test_incremental_l1_islands_match_full(column, threshold):
    G = island_analysis._load_and_project_graph(GRAPH)
    changed = 0
    for l1 in L1_FILES:
        baseline, incremental, full = _incremental_vs_full(G, column, threshold, l1)
        assert _partition(incremental) == _partition(full)
        changed += _partition(baseline) != _partition(full)
    if threshold == 0.0:  # L1 merges islands and adds new ones here, so the check bites
        assert changed


@pytest.mark.parametrize("column,threshold", [("EV2_ma", 0.2), ("EV3_ma", 0.0)])
def test_incremental_l1_islands_graph_order_ids_match_full(column, threshold):
    from src.road_islands import graph_order_island_ids

    G = island_analysis._load_and_project_graph(GRAPH)
    for l1 in L1_FILES:
        _, incremental, full = _incremental_vs_full(G, column, threshold, l1)
        assert graph_order_island_ids(G, incremental) == full

def test_incremental_l1_islands_directed():
    rng = np.random.default_rng(0)
    G = island_analysis._load_and_project_graph(GRAPH)
    import networkx as nx

    D = nx.MultiDiGraph()
    D.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        a, b = (u, v) if rng.random() < 0.5 else (v, u)
        D.add_edge(a, b, **d)
        if rng.random() < 0.7:  # most roads two-way, so cycles exist
            D.add_edge(b, a, **d)
    for l1 in L1_FILES:
        _, incremental, full = _incremental_vs_full(D, "EV3_ma", 0.0, l1)
        assert _partition(incremental) == _partition(full)


def test_incremental_l1_labels_cycle_nodes_off_restored_edge(monkeypatch):
    import networkx as nx
    from src import road_islands

    D = nx.MultiDiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "d")])
    edges = list(D.edges(keys=True, data=True))
    flooded = np.array([(u, v) == ("c", "a") for u, v, _, _ in edges])

    def fake_removal(G, threshold, hazard_column, l1_area_geojson=None, **kwargs):
        return edges, flooded & (l1_area_geojson is None), None

    monkeypatch.setattr(road_islands, "edge_removal", fake_removal)
    labels = road_islands.l1_node_islands(D, {"d": 0, "e": 0}, 0.0, "EV0_ma", "l1.geojson")
    assert _partition(labels) == _partition({"a": 0, "b": 0, "c": 0, "d": 1, "e": 1})


def _rfid_partition(table):
    return _partition(dict(zip(table["rfid"], table["island_id"])))


def test_road_island_table_matches_full_and_caches():
    l1 = L1_FILES[0]
    cache = {}
    for threshold in (0.0, 0.2):
        for l1_file in (None, l1):
            table = single_map.road_island_table(GRAPH, "EV3_ma", threshold, l1_file, cache)
            full = island_analysis.compute_island_geodataframe_from_graph(
                GRAPH, threshold, "EV3_ma", l1_area_geojson=l1_file
            )
            assert _rfid_partition(table) == _rfid_partition(full)
            assert sorted(table["rfid"]) == sorted(full["rfid"])
    n = len(cache)
    again = single_map.road_island_table(GRAPH, "EV3_ma", 0.0, l1, cache)
    assert len(cache) == n and again is cache[next(k for k in cache if "L1" in k and "0.0" in k)][0]
    single_map.road_island_table(GRAPH, "EV3_ma", 0.3, None, cache)  # new threshold: a miss
    assert len(cache) == n + 1


def test_equal_partitions_share_island_ids():
    from src.road_islands import canonical_island_ids

    _, incremental, full = _incremental_vs_full(
        island_analysis._load_and_project_graph(GRAPH), "EV3_ma", 0.0, L1_FILES[0]
    )
    assert canonical_island_ids(incremental) == canonical_island_ids(full)


def _experiment_inputs(tmp_path, assets):
    from config import get_societal_access_config

    config = get_config(root_dir=tmp_path, hazard_dir_override=HAZARD_DIR)
    config["simulation_config"]["verbose"] = False
    config["interim_dir"].mkdir(parents=True, exist_ok=True)
    groups = list(config["service_node_config"]["population_groups"].values())
    pop = gpd.GeoDataFrame(
        {g: np.full(len(assets), 10.0) for g in groups}, geometry=assets.geometry.buffer(150).values, crs=assets.crs
    )
    pop["cell_id"] = pop.index.astype(str)
    maps = [HAZARD_DIR / f"Frag_timesteps_test{i}.tif" for i in (0, 2, 5)]
    return dict(
        gdf_assets=assets, hazard_maps=maps, config=config, graph_path=GRAPH,
        societal_access_config=get_societal_access_config(config, pop, asset_types=["ls", "hospital"]),
    )


def test_run_single_map_experiment(tmp_path, assets):
    inputs = _experiment_inputs(tmp_path, assets)
    levers = {"number_repair_crews__msls": 2, "number_repair_crews__ls": 1, "fragility_param_k_divby2__msls": 12}
    out = single_map.run_single_map_experiment(
        **inputs, seed=3, island_cache={}, hazard_extraction_cache={}, **levers
    )
    assert out["flooded_assets"].shape == (3,) and out["flooded_assets"].max() > 0
    assert any(k.startswith("societal_access_pct") for k in out)
    assert "societal_capacity_retained_pct__hospital__elderly" in out
    assert out["societal_capacity_retained_pct__hospital__total"].shape == (3,)
    assert not any(k.startswith("annualised_") for k in out)
    again = single_map.run_single_map_experiment(
        **inputs, seed=3, island_cache={}, hazard_extraction_cache={}, **levers
    )
    for key in out:
        np.testing.assert_array_equal(out[key], again[key])


def test_single_map_builds_capacity_reference_from_unfiltered_graph(tmp_path):
    import pickle

    import networkx as nx
    from pyproj import Transformer
    from shapely.geometry import LineString, Point
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True).transform
    graph = nx.Graph()
    coords = {0: (100000, 450000), 1: (100100, 450000), 2: (100200, 450000),
              3: (101000, 450000), 4: (101100, 450000)}
    for node, xy in coords.items():
        x, y = project(*xy)
        graph.add_node(node, x=x, y=y)
    for rfid, (u, v) in enumerate([(0, 1), (1, 2), (3, 4)]):
        graph.add_edge(
            u, v, rfid=rfid, length=100,
            geometry=transform(project, LineString([coords[u], coords[v]])),
            EV0_ma=1.0 if rfid == 1 else 0.0,
        )
    graph_path = tmp_path / "reference.p"
    with graph_path.open("wb") as stream:
        pickle.dump(graph, stream)
    points = [Point(100050, 450000), Point(100150, 450000), Point(101050, 450000)]
    assets = gpd.GeoDataFrame({"type": ["hospital"] * 3}, geometry=points, crs="EPSG:28992")
    pop = gpd.GeoDataFrame(
        {"cell_id": ["a", "b", "c"], "population": [10, 10, 10]},
        geometry=[p.buffer(5) for p in points], crs=assets.crs,
    )
    attachments = single_map.undisrupted_state(assets, graph_path)
    ref_ids, ref_roads = single_map.undisrupted_reference(
        {"undisrupted": attachments}, graph_path
    )
    assert set(ref_roads["rfid"]) == {0, 1, 2}
    assert ref_ids[0] == ref_ids[1] != ref_ids[2]
    roads = single_map.island_road_sections(graph_path, "EV0_ma", 0.2)
    by_rfid = dict(zip(roads["rfid"], roads["island_id"]))
    ids = np.array([by_rfid.get(rfid, -1) for rfid in attachments[0]])
    cfg = {
        "pop_grid_gdf": pop, "cell_id_column": "cell_id",
        "pop_group_columns": {"total": "population"}, "capacity_groups": ["total"],
        "taxonomy": {"hospital": "hospital"}, "all_functions": ["hospital"],
        "service_area_function_provider_types": {},
    }
    outcome = {"intrinsic_operational": np.array([True, False, True])}
    explicit = single_map.single_map_societal_metrics(
        cfg, assets, outcome, ids, roads, "current",
        reference=(ref_ids, ref_roads, "reference"),
    )
    automatic = single_map.single_map_societal_metrics(
        cfg, assets, outcome, ids, roads, "current", graph_path=graph_path,
    )
    assert automatic["societal_mean_providers_reference__hospital__total"] == pytest.approx(5 / 3)
    assert automatic["societal_mean_providers__hospital__total"] == pytest.approx(1)
    assert automatic["societal_capacity_retained_pct__hospital__total"] == 60
    for key in ("societal_mean_providers_reference__hospital__total",
                "societal_mean_providers__hospital__total", "societal_capacity_retained_pct__hospital__total"):
        assert automatic[key] == explicit[key]
    with graph_path.open("rb") as stream:
        assert pickle.load(stream).number_of_edges() == 3


def _two_edge_graph(tmp_path, with_geometry=True):
    """Two 100 m roads; the second lacks ``geometry`` unless ``with_geometry``.
    Columns ``dry`` (0 m) and ``wet`` (1 m)."""
    import pickle

    import networkx as nx
    from pyproj import Transformer
    from shapely.geometry import LineString
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True).transform
    graph = nx.Graph()
    coords = {0: (100000, 450000), 1: (100100, 450000), 2: (100200, 450000)}
    for node, xy in coords.items():
        x, y = project(*xy)
        graph.add_node(node, x=x, y=y)
    for rfid, (u, v) in enumerate([(0, 1), (1, 2)]):
        attrs = dict(rfid=rfid, length=100, dry=0.0, wet=1.0)
        if with_geometry or rfid == 0:
            attrs["geometry"] = transform(project, LineString([coords[u], coords[v]]))
        graph.add_edge(u, v, **attrs)
    graph_path = tmp_path / "two_edges.p"
    with graph_path.open("wb") as stream:
        pickle.dump(graph, stream)
    return graph_path


def test_edge_without_geometry_gets_endpoint_line(tmp_path):
    roads = single_map.island_road_sections(_two_edge_graph(tmp_path, with_geometry=False), "dry", 0.2)
    assert roads.loc[roads["rfid"] == 1, "original_geometry"].iloc[0].length == pytest.approx(100, abs=1)
    assert roads["geometry"].notna().all()


def test_fully_flooded_map_leaves_assets_unreachable(tmp_path):
    from shapely.geometry import Point

    graph_path = _two_edge_graph(tmp_path)
    assets = gpd.GeoDataFrame({"type": ["hospital"] * 2},
                              geometry=[Point(100050, 450000), Point(100150, 450000)], crs="EPSG:28992")
    undisrupted = single_map.undisrupted_state(assets, graph_path)
    ids, roads = single_map.map_asset_islands(undisrupted, graph_path, "wet", 0.2)
    assert roads.empty
    assert (ids == -1).all()


def test_null_l1_depth_reduction_does_not_keep_flooded_roads():
    from shapely.geometry import LineString

    from src.road_islands import l1_edge_depth_reduction

    roads = [LineString([(0, 0), (1, 0)])]
    polygons = np.array([box(-1, -1, 2, 1), box(0, -1, 2, 1)], dtype=object)
    assert l1_edge_depth_reduction(roads, polygons, np.array([np.nan, 0.3])).tolist() == [0.3]
    assert l1_edge_depth_reduction(roads, polygons[:1], np.array([np.nan])).tolist() == [0.0]


def test_multi_map_islands_of_fully_flooded_graph_are_empty(tmp_path):
    from src.road_islands import compute_island_geodataframe_from_graph, road_state_asset_islands

    islands = compute_island_geodataframe_from_graph(_two_edge_graph(tmp_path), 0.2, "wet")
    assert islands.empty and {"rfid", "island_id", "island_size_km"} <= set(islands.columns)
    ids, _, _ = road_state_asset_islands(islands, pd.Series([0, 1]), [], [])
    assert (ids == -1).all()



def test_single_map_trace_and_island_report(tmp_path, assets):
    from src import island_report

    inputs = _experiment_inputs(tmp_path, assets)
    levers = {"number_repair_crews__msls": 2, "number_repair_crews__ls": 1}
    capture = []
    out = single_map.run_single_map_experiment(
        **inputs, seed=3, island_cache={}, hazard_extraction_cache={}, capture=capture, **levers
    )
    fields = ["island_id", "operational", "intrinsic_operational", "accessible", "unreachable",
              "flooded", "blocking_reason", "hazard_value"]
    columns = single_map.single_map_trace_columns(capture, fields)
    n = len(assets)
    assert len(columns["asset_id"]) == 3 * n and list(columns["map"][::n]) == ["Frag_timesteps_test0",
                                                                               "Frag_timesteps_test2",
                                                                               "Frag_timesteps_test5"]
    last = capture[-1]["outcome"]
    assert (columns["blocking_reason"][-n:] == last["blocking_reason"]).all()
    assert ((last["blocking_reason"] == 0) == last["operational"]).all()
    assert out["non_operational_assets"][-1] == (~last["operational"]).sum()

    c = capture[-1]
    paths = island_report.write_island_report(
        tmp_path, "m", c["roads"], c["outcome"]["island_id"], c["outcome"]["operational"],
        np.full(n, 2.0), single_map.road_peak_depths(GRAPH, c["column"]),
    )
    table = island_report.island_distribution(c["roads"], c["outcome"]["island_id"], c["outcome"]["operational"])
    assert table["n_assets"].sum() == n and all(p.exists() for p in paths)
    assert gpd.read_parquet(paths[0])["peak_depth_m"].notna().any()
    exported = gpd.read_parquet(paths[0])
    assert set(exported.geom_type) == {"LineString"} and "original_geometry" not in exported
    assert exported.crs == c["roads"].crs

    minx, miny, maxx, maxy = c["roads"].total_bounds
    half = box(minx, miny, (minx + maxx) / 2, maxy)
    clipped = island_report.write_island_report(
        tmp_path, "clip", c["roads"], c["outcome"]["island_id"], c["outcome"]["operational"],
        footprint=half, asset_geometry=assets.geometry,
    )
    assert 0 < len(gpd.read_parquet(clipped[0])) < len(exported)
    assert pd.read_csv(clipped[1])["n_assets"].sum() < n


def test_impact_based_crew_assignment(tmp_path, assets):
    inputs = _experiment_inputs(tmp_path, assets)
    out = single_map.run_single_map_experiment(
        **inputs, seed=1, island_cache={}, hazard_extraction_cache={}, number_repair_crews__msls=1,
        number_repair_crews__ls=1, repair_crew_assignment_method="population impacts islands",
        asset_population_map={i: float(i) for i in range(len(assets))},
    )
    assert out["flooded_assets"].max() > 0
    assert single_map.repair_impact_map("random", None, {1: 1.0}) is None


def test_single_map_monetary_rate():
    lu = {0: [("residential", 10.0, 2.0), ("commercial", 5.0, 1.0)], 1: [("residential", 1.0, 3.0)]}
    rate = single_map.single_map_monetary_rate(np.array([False, False, True]), np.arange(3), lu)
    assert rate["monetary_impact_residential"] == pytest.approx(23.0)
    assert rate["monetary_impact_commercial"] == pytest.approx(5.0)
    assert rate["monetary_impact_total"] == pytest.approx(28.0)
    assert single_map.single_map_monetary_rate(np.array([True]), np.arange(1), lu)["monetary_impact_total"] == 0.0
    assert single_map.single_map_monetary_rate(np.array([False]), np.arange(1), None)["monetary_impact_industrial"] == 0.0


def test_shift_overlay_columns():
    import networkx as nx
    from src.graph_manager import shift_overlay_columns

    G = nx.MultiGraph()
    G.add_edge(0, 1, EV1_ma=0.5, EV1_fr=1.0, EV2_ma=0.1, EV5_ma=9.0)
    G.add_edge(1, 2)
    shift_overlay_columns(G, 2)
    first, second = (d for *_, d in G.edges(data=True))
    assert first == {"EV0_ma": 0.5, "EV0_fr": 1.0, "EV1_ma": 0.1}  # EV5 dropped (beyond n_maps)
    assert second == {"EV0_ma": 0.0, "EV1_ma": 0.0}


def test_graph_manager_keys(tmp_path):
    from src.graph_manager import file_fingerprint, single_map_extent_key

    maps = [HAZARD_DIR / f"Frag_timesteps_test{i}.tif" for i in (1, 2)]
    key, polygon, _ = single_map_extent_key(maps)
    assert key == single_map_extent_key(maps)[0] != single_map_extent_key(maps, ("MOTORWAY",))[0]
    assert polygon.contains(polygon.centroid)
    copy = tmp_path / "m.tif"
    copy.write_bytes(maps[0].read_bytes())
    assert file_fingerprint(copy) != file_fingerprint(maps[0]) or copy.stat().st_mtime_ns == maps[0].stat().st_mtime_ns


def test_concurrent_graph_builds_overlay_once(tmp_path, monkeypatch):
    import pickle
    import threading
    import time

    import networkx as nx

    from src import graph_manager

    overlays = []

    def overlay_hazard_on_network(extent_path, maps, crs, extent_dir, static_path, *args, **kwargs):
        overlays.append(1)
        time.sleep(0.3)  # long enough for the other build to hit the lock
        out = static_path / "output_graph"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "base_graph_hazard.p", "wb") as f:
            pickle.dump(nx.Graph(), f)

    monkeypatch.setattr(graph_manager, "run_ra2ce", overlay_hazard_on_network)
    monkeypatch.setattr(graph_manager, "write_extent", lambda *a, **k: tmp_path / "extent.geojson")
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: real_sleep(min(s, 0.3)))
    maps = [HAZARD_DIR / "Frag_timesteps_test1.tif"]
    paths = []
    builds = [threading.Thread(target=lambda: paths.append(
        graph_manager.build_single_map_graph(maps, tmp_path))) for _ in range(2)]
    for t in builds:
        t.start()
    for t in builds:
        t.join()
    assert len(overlays) == 1
    assert paths[0] == paths[1] and paths[0].exists()
    assert not list(paths[0].parent.glob("build.lock"))


def test_graph_key_differs_for_same_named_maps_in_other_folders(tmp_path, monkeypatch):
    import os
    import pickle

    import networkx as nx

    from src import graph_manager

    def overlay_hazard_on_network(extent_path, maps, crs, extent_dir, static_path, *args, **kwargs):
        out = static_path / "output_graph"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "base_graph_hazard.p", "wb") as f:
            pickle.dump(nx.Graph(), f)

    monkeypatch.setattr(graph_manager, "run_ra2ce", overlay_hazard_on_network)
    monkeypatch.setattr(graph_manager, "write_extent", lambda *a, **k: tmp_path / "extent.geojson")
    source = HAZARD_DIR / "Frag_timesteps_test1.tif"
    copies = []
    for folder in ("event_a", "event_b"):
        (tmp_path / folder).mkdir()
        copy = tmp_path / folder / source.name
        copy.write_bytes(source.read_bytes())
        os.utime(copy, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns))
        copies.append(copy)
    assert graph_manager.file_fingerprint(copies[0]) == graph_manager.file_fingerprint(copies[1])
    paths = [graph_manager.build_single_map_graph([c], tmp_path / "root") for c in copies]
    assert paths[0] != paths[1]


def test_extent_state_keyed_on_exemptions_and_dependencies(tmp_path, assets):
    config = get_config(root_dir=tmp_path, hazard_dir_override=HAZARD_DIR)
    state = single_map._extent_state(assets, GRAPH, config)
    assert single_map._extent_state(assets, GRAPH, config) is state
    config["simulation_config"]["exempt_motorways"] = False
    no_exempt = single_map._extent_state(assets, GRAPH, config)
    assert no_exempt is not state
    config["dependency_parameters"] = {**config["dependency_parameters"], "dependency_edges": []}
    assert single_map._extent_state(assets, GRAPH, config) is not no_exempt


def test_depth_cache_keyed_by_file(tmp_path, assets):
    inputs = _experiment_inputs(tmp_path, assets)
    cache = {}
    first = single_map.map_asset_depths(inputs["hazard_maps"][1], assets, 1, inputs["config"], cache)
    assert len(cache) == 1 and next(iter(cache)).startswith("single_map_depths|")
    cache[next(iter(cache))] = np.full(len(assets), 7.0)  # a hit returns the cached entry
    assert (single_map.map_asset_depths(inputs["hazard_maps"][1], assets, 1, inputs["config"], cache) == 7.0).all()
    assert (first != 7.0).any()


def test_write_extent_is_wgs84(tmp_path):
    from shapely.geometry import box
    from src.graph_manager import write_extent

    path = write_extent(box(80000, 440000, 85000, 445000), "EPSG:28992", tmp_path)
    extent = gpd.read_file(path)
    assert extent.crs.to_epsg() == 4326 and 3 < extent.total_bounds[0] < 5


def test_return_periods_and_annualised():
    assert single_map.return_periods(["flood_RP10", "flood_RP100", "flood_RP1000"]) == [10, 100, 1000]
    assert single_map.return_periods(["floodA_RP10", "floodB_RP100"]) is None
    assert single_map.return_periods(["a", "b"]) is None
    assert single_map.annualised([0.0, 10.0], [100, 10]) == pytest.approx(0.5 * 10.0 * (0.1 - 0.01))
