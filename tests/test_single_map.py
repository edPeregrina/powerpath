import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import get_config  # noqa: E402
from src import island_analysis, single_map  # noqa: E402
from src.damage_recovery import default_fragility_function  # noqa: E402

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
    assert (all_fail["recovery_time"][all_fail["flooded"]]
            == all_fail["repair_time"][all_fail["flooded"]] + 24.0).all()

    u = np.random.default_rng(1).random(len(assets))
    assert (run(u)["intrinsic_operational"] == run(u)["intrinsic_operational"]).all()


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
    *sorted((REPO / "data" / "test_samples" / "adaptation" / "L1_adaptations").glob("*.geojson")),
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
        societal_access_config=get_societal_access_config(config, pop, asset_types=["ls"]),
    )


def test_run_single_map_experiment(tmp_path, assets):
    inputs = _experiment_inputs(tmp_path, assets)
    levers = {"repair_crews__msls": 2, "repair_crews__ls": 1, "fragility_param_k_divby2__msls": 12}
    out = single_map.run_single_map_experiment(
        **inputs, seed=3, island_cache={}, hazard_extraction_cache={}, **levers
    )
    assert out["flooded_assets"].shape == (3,) and out["flooded_assets"].max() > 0
    assert any(k.startswith("societal_access_pct") for k in out)
    assert not any(k.startswith("annualised_") for k in out)
    again = single_map.run_single_map_experiment(
        **inputs, seed=3, island_cache={}, hazard_extraction_cache={}, **levers
    )
    for key in out:
        np.testing.assert_array_equal(out[key], again[key])


def test_return_periods_and_annualised():
    assert single_map.return_periods(["flood_RP10", "flood_RP100", "flood_RP1000"]) == [10, 100, 1000]
    assert single_map.return_periods(["floodA_RP10", "floodB_RP100"]) is None
    assert single_map.return_periods(["a", "b"]) is None
    assert single_map.annualised([0.0, 10.0], [100, 10]) == pytest.approx(0.5 * 10.0 * (0.1 - 0.01))
