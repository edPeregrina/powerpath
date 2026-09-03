import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from config import get_config

import src.island_analysis as island_analysis
import src.grid_based_accessibility_hex as grid_hex_module
import src.impacts as impacts_module
import src.simulation as simulation_module
from src.adaptation import simulate_asset_damage_recovery_access_breakdown_ema
from src.caching import (
    create_island_cache_key,
    get_asset_centroid_hash,
    load_simulation_caches,
    save_simulation_caches,
)
from src.simulation import simulate_asset_damage_recovery_access_breakdown
from src.societal_access import (
    analyse_societal_access,
    apply_population_to_allocations,
    build_allocation_cache_key,
    build_destination_function_map,
    build_island_assignment,
    build_origin_island_allocations,
    compute_access_matrix,
    compute_access_matrix_from_origins,
    compute_equity_gaps,
    compute_origin_access,
    get_or_build_allocation,
    list_societal_metric_names,
    postprocess_societal_access_results,
)


def _make_islands_gdf():
    return gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:28992",
    )


def _make_population_gdf():
    return gpd.GeoDataFrame(
        {
            "cell_id": ["split", "inside_1", "near_2", "far"],
            "aantal_inwoners": [100, 50, 30, 20],
            "aantal_inwoners_65_jaar_en_ouder": [20, 10, 6, 4],
            "aantal_inwoners_0_tot_15_jaar": [25, 10, 5, 2],
            "aantal_inwoners_25_tot_45_jaar": [40, 20, 10, 5],
        },
        geometry=[
            box(8, 2, 12, 6),
            box(1, 1, 4, 4),
            box(20.5, 1, 21.5, 2),
            box(100, 100, 101, 101),
        ],
        crs="EPSG:28992",
    )


def test_graph_path_access_pipeline():
    nx = pytest.importorskip("networkx")
    graph = nx.Graph()
    graph.add_edges_from([
        ("origin_a", "hospital_1"),
        ("origin_b", "school_1"),
    ])
    graph.add_node("isolated")

    assignment = build_island_assignment(graph)
    assert assignment["origin_a"] == assignment["hospital_1"]
    assert assignment["origin_b"] == assignment["school_1"]
    assert assignment["isolated"] not in {assignment["origin_a"], assignment["origin_b"]}

    with pytest.warns(UserWarning):
        island_functions = build_destination_function_map(
            {"hospital_1": "hospital", "school_1": "school", "missing": "clinic"},
            assignment,
        )

    origin_access = compute_origin_access(
        {"origin_a": assignment["origin_a"], "origin_b": assignment["origin_b"]},
        island_functions,
        all_functions=["education", "health"],
    )
    assert bool(origin_access.loc["origin_a", "health"])
    assert not bool(origin_access.loc["origin_a", "education"])

    stakeholder_groups = {
        "total": {"origin_a": 60, "origin_b": 40},
        "elderly": {"origin_a": 30, "origin_b": 10},
    }
    access_matrix = compute_access_matrix_from_origins(origin_access, stakeholder_groups)
    assert access_matrix.loc["health", "total"] == 60.0
    assert access_matrix.loc["education", "total"] == 40.0

    equity = compute_equity_gaps(access_matrix, reference_group="total")
    assert equity.loc["health", "elderly_absolute_gap"] == -15.0


def test_spatial_access_pipeline_and_wrapper():
    islands = gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(200, 0, 210, 10)],
        crs="EPSG:28992",
    )
    population = gpd.GeoDataFrame(
        {
            "aantal_inwoners": [100, 50],
            "aantal_inwoners_65_jaar_en_ouder": [20, 10],
            "aantal_inwoners_0_tot_15_jaar": [25, 10],
            "aantal_inwoners_25_tot_45_jaar": [40, 20],
        },
        geometry=[box(1, 1, 3, 3), box(202, 1, 204, 3)],
        crs="EPSG:28992",
    )
    services = gpd.GeoDataFrame(
        {"type": ["hospital", "school"]},
        geometry=[Point(2, 2), Point(203, 2)],
        crs="EPSG:28992",
    )

    result = analyse_societal_access(
        islands_gdf=islands,
        population_gdf=population,
        service_nodes_gdf=services,
    )

    access_matrix = result["access_matrix"]
    assert access_matrix.loc["health", "total"] == pytest.approx(66.67, abs=0.01)
    assert access_matrix.loc["education", "total"] == pytest.approx(33.33, abs=0.01)
    assert result["origin_access"] is None


def test_default_config_uses_health_function_category():
    config = get_config()
    taxonomy = config["service_node_config"]["taxonomy"]
    assert taxonomy["hospital"] == "health"
    assert taxonomy["clinic"] == "health"


def test_compute_access_matrix_handles_explicit_functions():
    island_function_map = {1: frozenset(["hospital"]), 2: frozenset(["education"])}
    island_population_df = pd.DataFrame(
        {
            "island_id": [1, 2],
            "aantal_inwoners": [60, 40],
            "aantal_inwoners_65_jaar_en_ouder": [20, 10],
            "aantal_inwoners_0_tot_15_jaar": [15, 10],
            "aantal_inwoners_25_tot_45_jaar": [25, 20],
        }
    )

    matrix = compute_access_matrix(
        island_function_map,
        island_population_df,
        all_functions=["education", "hospital", "repair_logistics"],
    )
    assert matrix.loc["repair_logistics", "total"] == 0.0


def test_build_origin_island_allocations_covers_intersection_nearest_and_unassigned():
    islands = _make_islands_gdf()
    pop = _make_population_gdf()

    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=islands,
        nearest_max_distance=2.0,
        road_state_key="roads_a",
    )

    split_rows = allocation_df[allocation_df["cell_id"] == "split"].sort_values("island_id")
    assert len(split_rows) == 2
    assert split_rows["allocation_method"].tolist() == ["intersection", "intersection"]
    assert split_rows["allocation_fraction"].sum() == pytest.approx(1.0)
    assert split_rows["allocation_fraction"].tolist() == pytest.approx([0.5, 0.5])

    near_row = allocation_df[allocation_df["cell_id"] == "near_2"].iloc[0]
    assert near_row["island_id"] == 2
    assert near_row["allocation_method"] == "nearest_island"

    far_row = allocation_df[allocation_df["cell_id"] == "far"].iloc[0]
    assert far_row["island_id"] == -1
    assert far_row["allocation_method"] == "unassigned"


def test_apply_population_to_allocations_weights_and_sanitises_values():
    pop = _make_population_gdf().copy()
    pop.loc[0, "aantal_inwoners_65_jaar_en_ouder"] = -5
    pop["aantal_inwoners"] = pop["aantal_inwoners"].astype(object)
    pop.loc[1, "aantal_inwoners"] = "suppressed"

    allocation_df = pd.DataFrame(
        {
            "cell_id": ["split", "split", "inside_1"],
            "island_id": [1, 2, 1],
            "allocation_fraction": [0.25, 0.75, 1.0],
            "allocation_method": ["intersection", "intersection", "intersection"],
        }
    )

    merged = apply_population_to_allocations(allocation_df, pop, "cell_id")
    split_island_2 = merged[(merged["cell_id"] == "split") & (merged["island_id"] == 2)].iloc[0]
    assert split_island_2["total_weighted"] == pytest.approx(75.0)
    assert split_island_2["elderly_weighted"] == 0.0
    inside_1 = merged[merged["cell_id"] == "inside_1"].iloc[0]
    assert inside_1["aantal_inwoners"] == 0.0


def test_get_or_build_allocation_uses_cache_key():
    islands = _make_islands_gdf()
    pop = _make_population_gdf()
    cache = {}

    alloc_1, key_1, updated_1 = get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_a")
    alloc_2, key_2, updated_2 = get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_a")
    changed_key = build_allocation_cache_key(pop, "cell_id", islands, road_state_key="roads_b")

    assert updated_1 is True
    assert updated_2 is False
    assert key_1 == key_2
    assert key_1 in cache
    assert alloc_1.equals(alloc_2)
    assert changed_key != key_1


def test_island_cache_key_includes_l2_road_adaptation():
    l2 = gpd.GeoDataFrame(
        {"depth_red": [0.5]},
        geometry=[box(0, 0, 5, 5)],
        crs="EPSG:28992",
    )

    baseline_key = create_island_cache_key("EV0_ma", 0.2, "assets")
    l2_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        "assets",
        l2_asset_geojson=l2,
        l2_active_timesteps=[0, 24],
    )
    other_timesteps_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        "assets",
        l2_asset_geojson=l2,
        l2_active_timesteps=[48],
    )
    same_bounds_l2 = gpd.GeoDataFrame(
        {"depth_red": [0.5]},
        geometry=[Polygon([(0, 0), (5, 0), (5, 5), (0, 0)])],
        crs="EPSG:28992",
    )
    other_geometry_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        "assets",
        l2_asset_geojson=same_bounds_l2,
        l2_active_timesteps=[0, 24],
    )

    assert l2_key != baseline_key
    assert other_timesteps_key != l2_key
    assert other_geometry_key != l2_key


def test_l2_road_adaptation_changes_computed_topology():
    nx = pytest.importorskip("networkx")
    from src.utils import filter_hazard_graph

    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_edge(
        1,
        2,
        geometry=LineString([(0, 0), (1, 0)]),
        EV0_ma=0.3,
    )
    l2 = gpd.GeoDataFrame(
        {"depth_red": [0.2]},
        geometry=[box(-0.1, -0.1, 1.1, 0.1)],
        crs="EPSG:4326",
    )

    disrupted = filter_hazard_graph(
        graph.copy(), 0.2, "EV0_ma"
    )
    adapted = filter_hazard_graph(
        graph.copy(), 0.2, "EV0_ma", l2_asset_geojson=l2
    )

    assert not disrupted.has_edge(1, 2)
    assert adapted.has_edge(1, 2)

    parallel_graph = nx.MultiGraph()
    parallel_graph.add_nodes_from(graph.nodes(data=True))
    for edge_key in ("a", "b"):
        parallel_graph.add_edge(
            1,
            2,
            key=edge_key,
            geometry=LineString([(0, 0), (1, 0)]),
            EV0_ma=0.5,
        )
    adapted_parallel = filter_hazard_graph(
        parallel_graph, 0.2, "EV0_ma", l2_asset_geojson=l2
    )
    assert adapted_parallel.number_of_edges() == 0


def test_adaptation_geodataframe_is_not_mutated():
    from src.adaptation import build_l1_l2_reduction_array

    assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(0, 0)],
        crs="EPSG:28992",
    )
    adaptation = gpd.GeoDataFrame(
        geometry=[box(-1, -1, 1, 1)],
        crs="EPSG:28992",
    )

    build_l1_l2_reduction_array(
        assets,
        l1_area_geojson=adaptation,
        l1_active_timesteps=[0],
        hazard_maps=["unused"],
        major_timestep=1,
    )

    assert "depth_red" not in adaptation.columns


def test_societal_allocation_cache_uses_simulation_cache_pickle(tmp_path):
    allocation_df = build_origin_island_allocations(
        _make_population_gdf(),
        cell_id_column="cell_id",
        islands_gdf=_make_islands_gdf(),
        road_state_key="roads_a",
    )
    cache = {"allocation_key": allocation_df}

    saved = save_simulation_caches(
        {"societal_allocation_cache": cache},
        tmp_path,
        hazard_dir="hazard_a",
    )
    loaded = load_simulation_caches(tmp_path, hazard_dir="hazard_a")

    assert (
        tmp_path / "societal_allocation_cache_hazard_a.pkl"
    ).exists()
    assert saved["societal_allocation_cache"]["count"] == 1
    assert loaded["societal_allocation_cache"]["allocation_key"].equals(
        allocation_df
    )
    assert (
        loaded["societal_allocation_cache"]["allocation_key"].attrs
        == allocation_df.attrs
    )


def test_island_cache_remains_lightweight_without_islands_gdf(tmp_path, monkeypatch):
    config = get_config(
        root_dir=tmp_path,
        hazard_dir_override=tmp_path / "hazard_case",
    )
    config["simulation_config"]["verbose"] = False

    temp_gdf = gpd.GeoDataFrame(
        {
            "type": ["msls", "hospital"],
            "access_rfid": [10, 10],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )

    monkeypatch.setattr(
        island_analysis,
        "compute_island_geodataframe_from_graph",
        lambda *args, **kwargs: gpd.GeoDataFrame(
            {
                "rfid": [10],
                "island_id": [1],
                "length_m": [100.0],
            },
            geometry=[LineString([(0, 0), (10, 0)])],
            crs="EPSG:28992",
        ),
    )

    island_cache = {}
    asset_island_ids, rfids_islands = island_analysis.match_island_ids_assets(
        temp_gdf,
        boundary_asset_indices=[],
        boundary_islands_rfids=[],
        hazard_threshold=0.2,
        hazard_column="EV0_ma",
        config=config,
        island_cache=island_cache,
        cache_dir=config["interim_dir"],
        hazard_dir=config["hazard_dir"],
    )

    cache_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        get_asset_centroid_hash(temp_gdf),
    )
    assert asset_island_ids.tolist() == [1, 1]
    assert rfids_islands == {10: 1}
    assert set(island_cache[cache_key].keys()) == {"island_ids", "rfids_islands"}
    assert "islands_gdf" not in island_cache[cache_key]


def test_match_island_ids_assets_cache_hit_with_cached_allocation_skips_graph(
    tmp_path, monkeypatch, capsys
):
    config = get_config(
        root_dir=tmp_path,
        hazard_dir_override=tmp_path / "hazard_case",
    )
    config["simulation_config"]["verbose"] = True

    temp_gdf = gpd.GeoDataFrame(
        {
            "type": ["msls", "hospital"],
            "access_rfid": [10, 10],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    cache_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        get_asset_centroid_hash(temp_gdf),
    )
    island_cache = {
        cache_key: {
            "island_ids": np.array([1, 1], dtype=int),
            "rfids_islands": {10: 1},
        }
    }

    pop = _make_population_gdf().iloc[:1].copy()
    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=_make_islands_gdf(),
        road_state_key=cache_key,
    )
    allocation_cache = {"allocation_key": allocation_df}

    compute_calls = {"count": 0}
    save_calls = {"count": 0}

    def _unexpected_compute(*args, **kwargs):
        compute_calls["count"] += 1
        raise AssertionError("Graph reconstruction should not run on allocation cache hit")

    def _unexpected_save(*args, **kwargs):
        save_calls["count"] += 1
        raise AssertionError("Island cache should not be rewritten on cache hit")

    monkeypatch.setattr(
        island_analysis,
        "compute_island_geodataframe_from_graph",
        _unexpected_compute,
    )
    monkeypatch.setattr(island_analysis, "save_island_cache", _unexpected_save)

    asset_island_ids, rfids_islands = island_analysis.match_island_ids_assets(
        temp_gdf,
        boundary_asset_indices=[],
        boundary_islands_rfids=[],
        hazard_threshold=0.2,
        hazard_column="EV0_ma",
        config=config,
        island_cache=island_cache,
        cache_dir=config["interim_dir"],
        hazard_dir=config["hazard_dir"],
        societal_allocation_cache=allocation_cache,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
    )

    output = capsys.readouterr().out
    assert compute_calls["count"] == 0
    assert save_calls["count"] == 0
    assert "Island cache hit" in output
    assert "Societal allocation cache hit" in output
    assert "Skipping hazard graph reconstruction" in output
    assert asset_island_ids.tolist() == [1, 1]
    assert rfids_islands == {10: 1}


def test_match_island_ids_assets_cache_hit_missing_allocation_builds_once(
    tmp_path, monkeypatch
):
    config = get_config(
        root_dir=tmp_path,
        hazard_dir_override=tmp_path / "hazard_case",
    )
    config["simulation_config"]["verbose"] = False

    temp_gdf = gpd.GeoDataFrame(
        {
            "type": ["msls", "hospital"],
            "access_rfid": [10, 10],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    cache_key = create_island_cache_key(
        "EV0_ma",
        0.2,
        get_asset_centroid_hash(temp_gdf),
    )
    island_cache = {
        cache_key: {
            "island_ids": np.array([1, 1], dtype=int),
            "rfids_islands": {10: 1},
        }
    }
    pop = _make_population_gdf().iloc[:1].copy()
    allocation_cache = {}

    compute_calls = {"count": 0}
    save_calls = {"count": 0}

    def _mock_compute(*args, **kwargs):
        compute_calls["count"] += 1
        return gpd.GeoDataFrame(
            {"rfid": [10], "island_id": [1], "length_m": [100.0]},
            geometry=[LineString([(0, 0), (10, 0)])],
            crs="EPSG:28992",
        )

    monkeypatch.setattr(
        island_analysis,
        "compute_island_geodataframe_from_graph",
        _mock_compute,
    )
    monkeypatch.setattr(
        island_analysis,
        "save_island_cache",
        lambda *args, **kwargs: save_calls.__setitem__("count", save_calls["count"] + 1),
    )

    for _ in range(2):
        asset_island_ids, rfids_islands = island_analysis.match_island_ids_assets(
            temp_gdf,
            boundary_asset_indices=[],
            boundary_islands_rfids=[],
            hazard_threshold=0.2,
            hazard_column="EV0_ma",
            config=config,
            island_cache=island_cache,
            cache_dir=config["interim_dir"],
            hazard_dir=config["hazard_dir"],
            societal_allocation_cache=allocation_cache,
            pop_grid_gdf=pop,
            cell_id_column="cell_id",
        )
        assert asset_island_ids.tolist() == [1, 1]
        assert rfids_islands == {10: 1}

    assert compute_calls["count"] == 1
    assert save_calls["count"] == 0
    assert len(allocation_cache) == 1


def test_match_island_ids_assets_repeated_run_reuses_island_and_allocation_caches(
    tmp_path, monkeypatch
):
    config = get_config(
        root_dir=tmp_path,
        hazard_dir_override=tmp_path / "hazard_case",
    )
    config["simulation_config"]["verbose"] = False

    temp_gdf = gpd.GeoDataFrame(
        {
            "type": ["msls", "hospital"],
            "access_rfid": [10, 10],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:28992",
    )
    island_cache = {}
    pop = _make_population_gdf().iloc[:1].copy()
    allocation_cache = {}

    compute_calls = {"count": 0}
    save_calls = {"count": 0}

    def _mock_compute(*args, **kwargs):
        compute_calls["count"] += 1
        return gpd.GeoDataFrame(
            {"rfid": [10], "island_id": [1], "length_m": [100.0]},
            geometry=[LineString([(0, 0), (10, 0)])],
            crs="EPSG:28992",
        )

    monkeypatch.setattr(
        island_analysis,
        "compute_island_geodataframe_from_graph",
        _mock_compute,
    )
    monkeypatch.setattr(
        island_analysis,
        "save_island_cache",
        lambda *args, **kwargs: save_calls.__setitem__("count", save_calls["count"] + 1),
    )

    for _ in range(2):
        asset_island_ids, rfids_islands = island_analysis.match_island_ids_assets(
            temp_gdf,
            boundary_asset_indices=[],
            boundary_islands_rfids=[],
            hazard_threshold=0.2,
            hazard_column="EV0_ma",
            config=config,
            island_cache=island_cache,
            cache_dir=config["interim_dir"],
            hazard_dir=config["hazard_dir"],
            societal_allocation_cache=allocation_cache,
            pop_grid_gdf=pop,
            cell_id_column="cell_id",
        )
        assert asset_island_ids.tolist() == [1, 1]
        assert rfids_islands == {10: 1}

    assert compute_calls["count"] == 1
    assert save_calls["count"] == 1
    assert len(island_cache) == 1
    assert len(allocation_cache) == 1


def test_postprocess_societal_access_results_uses_cached_allocations():
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=islands,
        nearest_max_distance=2.0,
        road_state_key="0",
    )
    cache = {"k": allocation_df}

    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital", "school", "fire_station"]},
        geometry=[Point(1, 1), Point(12, 1), Point(11, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "0",
        "operational": np.array([True, False, True]),
        "island_id": np.array([1, 2, 2]),
    }]

    updated_summary, updated_cache = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=cache,
        all_functions=["education", "emergency_response", "hospital"],
        nearest_max_distance=2.0,
    )

    row = updated_summary[0]
    assert row["societal_total_population__total"] == pytest.approx(150.0)
    assert row["societal_access_pct__hospital__total"] == pytest.approx(66.67, abs=0.01)
    assert row["societal_access_pct__education__total"] == pytest.approx(0.0)
    assert row["societal_access_pct__emergency_response__total"] == pytest.approx(33.33, abs=0.01)
    assert updated_cache is cache


def test_postprocess_without_matching_allocation_emits_nan():
    pop = _make_population_gdf().iloc[:1].copy()
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(1, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 99}]
    detailed_results = [{"timestep": 0, "map": 99, "operational": [True], "island_id": [1]}]

    updated_summary, _ = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache={},
        all_functions=["hospital"],
    )

    assert math.isnan(updated_summary[0]["societal_access_pct__hospital__total"])


def test_postprocess_does_not_reuse_another_road_state_allocation():
    pop = _make_population_gdf().iloc[:1].copy()
    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=_make_islands_gdf(),
        road_state_key="roads_a",
    )
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(1, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_b",
        "operational": [True],
        "island_id": [1],
    }]

    updated_summary, _ = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache={"allocation_key": allocation_df},
        all_functions=["hospital"],
    )

    assert math.isnan(
        updated_summary[0]["societal_access_pct__hospital__total"]
    )


def test_postprocess_does_not_reuse_another_population_grid_allocation():
    source_pop = _make_population_gdf().iloc[:1].copy()
    allocation_df = build_origin_island_allocations(
        source_pop,
        cell_id_column="cell_id",
        islands_gdf=_make_islands_gdf(),
        road_state_key="roads_a",
    )
    changed_pop = source_pop.copy()
    changed_pop.geometry = [box(2, 2, 4, 4)]
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(1, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_a",
        "operational": [True],
        "island_id": [1],
    }]

    updated_summary, _ = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=changed_pop,
        cell_id_column="cell_id",
        allocation_cache={"allocation_key": allocation_df},
        all_functions=["hospital"],
    )

    assert math.isnan(
        updated_summary[0]["societal_access_pct__hospital__total"]
    )


def test_list_societal_metric_names_contains_expected_patterns():
    names = list_societal_metric_names(
        ["hospital"],
        {"total": "aantal_inwoners", "elderly": "aantal_inwoners_65_jaar_en_ouder"},
    )
    assert "societal_access_pct__hospital__total" in names
    assert "societal_total_population__elderly" in names
    assert "societal_equity_absolute_gap__hospital__elderly" in names


# ---------------------------------------------------------------------------
# MSLS-only electricity access tests
# ---------------------------------------------------------------------------

def _make_mixed_assets_gdf():
    """Mixed LS, MS, and MSLS assets for taxonomy tests."""
    return gpd.GeoDataFrame(
        {"type": ["ls", "ms", "msls", "msls"]},
        geometry=[Point(1, 1), Point(5, 5), Point(9, 9), Point(12, 5)],
        crs="EPSG:28992",
    )


def test_postprocess_verbose_logs_asset_and_provider_scope(capsys):
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        islands,
        road_state_key="roads_scope",
    )

    assets = gpd.GeoDataFrame(
        {"type": ["ls", "ms", "msls", "hospital"]},
        geometry=[Point(1, 1), Point(5, 5), Point(9, 9), Point(12, 5)],
        crs="EPSG:28992",
    )
    postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads_scope",
            "operational": np.array([True, True, True, True]),
            "island_id": np.array([1, 1, 1, 2]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"msls": "electricity", "hospital": "hospital"},
        all_functions=["electricity", "hospital"],
        verbose=True,
    )
    output = capsys.readouterr().out
    assert "Simulation assets: 4" in output
    assert "Asset counts by type:" in output
    assert "Electricity providers: 1 MSLS" in output
    assert "Hospital providers: 1" in output


def test_ls_and_ms_assets_do_not_provide_electricity_access():
    """LS and MS assets must not contribute electricity access."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    cache = {}
    get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_ls_ms")

    ls_ms_assets = _make_mixed_assets_gdf()

    # Filter to msls only (empty)
    msls_services = ls_ms_assets.loc[ls_ms_assets["type"].eq("msls"), ["type", "geometry"]].copy()
    assert msls_services.empty is False  # we have msls assets in the full gdf

    # Use only ls/ms assets — no electricity should appear
    only_ls_ms = ls_ms_assets.loc[ls_ms_assets["type"].isin(["ls", "ms"])].copy()
    summary = [{"timestep": 0, "map": 0}]
    detailed = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_ls_ms",
        "operational": np.array([True, True]),
        "island_id": np.array([1, 2]),
    }]

    updated, _ = postprocess_societal_access_results(
        summary_results=summary,
        detailed_results=detailed,
        gdf_assets=only_ls_ms,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=cache,
        taxonomy={"msls": "electricity"},
        asset_type_column="type",
        all_functions=["electricity"],
    )

    # island_function_map is empty because no msls assets exist
    # → access is 0 %
    assert updated[0]["societal_access_pct__electricity__total"] == pytest.approx(0.0)


def test_operational_msls_provides_electricity_access():
    """Operational MSLS assets grant electricity access to their road island."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    cache = {}
    get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_msls_op")

    assets = _make_mixed_assets_gdf()  # has ls, ms, msls, msls
    summary = [{"timestep": 0, "map": 0}]
    detailed = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_msls_op",
        "operational": np.array([False, False, True, True]),  # only msls operational
        "island_id": np.array([1, 2, 1, 2]),
    }]

    updated, _ = postprocess_societal_access_results(
        summary_results=summary,
        detailed_results=detailed,
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=cache,
        taxonomy={"msls": "electricity"},
        asset_type_column="type",
        all_functions=["electricity"],
    )

    # Both islands have an operational MSLS asset, so all allocated population has access
    assert updated[0]["societal_access_pct__electricity__total"] == pytest.approx(100.0, abs=0.01)


def test_failed_msls_assets_remove_electricity_access():
    """When MSLS assets are non-operational, electricity access drops to zero."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    cache = {}
    get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_fail")

    assets = _make_mixed_assets_gdf()
    summary = [{"timestep": 0, "map": 0}]
    detailed = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_fail",
        "operational": np.array([False, False, False, False]),  # all failed
        "island_id": np.array([1, 2, 1, 2]),
    }]

    updated, _ = postprocess_societal_access_results(
        summary_results=summary,
        detailed_results=detailed,
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=cache,
        taxonomy={"msls": "electricity"},
        asset_type_column="type",
        all_functions=["electricity"],
    )

    assert updated[0]["societal_access_pct__electricity__total"] == pytest.approx(0.0)


def test_electricity_access_uses_voronoi_service_areas_not_road_islands():
    islands = gpd.GeoDataFrame(
        {"island_id": [1]},
        geometry=[box(-10, -10, 210, 10)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["west", "east"],
            "aantal_inwoners": [100, 100],
            "aantal_inwoners_65_jaar_en_ouder": [20, 20],
            "aantal_inwoners_0_tot_15_jaar": [15, 15],
            "aantal_inwoners_25_tot_45_jaar": [40, 40],
        },
        geometry=[box(-5, -5, 5, 5), box(195, -5, 205, 5)],
        crs="EPSG:28992",
    )
    assets = gpd.GeoDataFrame(
        {"type": ["msls", "msls"]},
        geometry=[Point(0, 0), Point(200, 0)],
        crs="EPSG:28992",
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        islands,
        road_state_key="roads_connected",
    )

    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads_connected",
            "operational": np.array([True, False]),
            "island_id": np.array([1, 1]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"msls": "electricity"},
        all_functions=["electricity"],
    )

    assert updated[0]["societal_access_pct__electricity__total"] == pytest.approx(50.0)


def test_service_area_function_provider_types_can_override_default_functions():
    islands = gpd.GeoDataFrame(
        {"island_id": [1]},
        geometry=[box(-10, -10, 210, 10)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["west", "east"],
            "aantal_inwoners": [100, 100],
            "aantal_inwoners_65_jaar_en_ouder": [20, 20],
            "aantal_inwoners_0_tot_15_jaar": [15, 15],
            "aantal_inwoners_25_tot_45_jaar": [40, 40],
        },
        geometry=[box(-5, -5, 5, 5), box(195, -5, 205, 5)],
        crs="EPSG:28992",
    )
    assets = gpd.GeoDataFrame(
        {"type": ["hospital", "hospital"]},
        geometry=[Point(0, 0), Point(200, 0)],
        crs="EPSG:28992",
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        islands,
        road_state_key="roads_connected_hospital",
    )
    common_kwargs = dict(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads_connected_hospital",
            "operational": np.array([True, False]),
            "island_id": np.array([1, 1]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"hospital": "hospital"},
        all_functions=["hospital"],
    )

    updated_default, _ = postprocess_societal_access_results(**common_kwargs)
    assert updated_default[0]["societal_access_pct__hospital__total"] == pytest.approx(100.0)

    updated_override, _ = postprocess_societal_access_results(
        **{
            **common_kwargs,
            "summary_results": [{"timestep": 0, "map": 0}],
            "detailed_results": [{
                "timestep": 0,
                "map": 0,
                "road_state_key": "roads_connected_hospital",
                "operational": np.array([True, False]),
                "island_id": np.array([1, 1]),
            }],
        },
        service_area_function_provider_types={"hospital": frozenset({"hospital"})},
    )

    assert updated_override[0]["societal_access_pct__hospital__total"] == pytest.approx(50.0)


def test_postprocess_reuses_cached_voronoi_across_calls(monkeypatch, capsys):
    impacts_module._VORONOI_CACHE.clear()

    islands = gpd.GeoDataFrame(
        {"island_id": [1]},
        geometry=[box(-10, -10, 210, 210)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["center_a", "center_b"],
            "aantal_inwoners": [100, 100],
            "aantal_inwoners_65_jaar_en_ouder": [20, 20],
            "aantal_inwoners_0_tot_15_jaar": [15, 15],
            "aantal_inwoners_25_tot_45_jaar": [40, 40],
        },
        geometry=[
            box(95, 95, 100, 100),
            box(100, 100, 105, 105),
        ],
        crs="EPSG:28992",
    )
    assets = gpd.GeoDataFrame(
        {"type": ["msls", "msls", "msls", "msls", "msls"]},
        geometry=[
            Point(0, 0),
            Point(200, 0),
            Point(0, 200),
            Point(200, 200),
            Point(100, 100),
        ],
        crs="EPSG:28992",
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        islands,
        road_state_key="roads_cached_voronoi",
    )

    original_voronoi = impacts_module.Voronoi
    build_calls = {"count": 0}

    def counting_voronoi(*args, **kwargs):
        build_calls["count"] += 1
        return original_voronoi(*args, **kwargs)

    monkeypatch.setattr(impacts_module, "Voronoi", counting_voronoi)

    kwargs = dict(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads_cached_voronoi",
            "operational": np.array([False, False, False, False, True]),
            "island_id": np.array([1, 1, 1, 1, 1]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"msls": "electricity"},
        all_functions=["electricity"],
    )

    updated_first, _ = postprocess_societal_access_results(**kwargs)
    first_output = capsys.readouterr().out
    updated_second, _ = postprocess_societal_access_results(**kwargs)
    second_output = capsys.readouterr().out

    assert build_calls["count"] == 1
    assert len(impacts_module._VORONOI_CACHE) == 1
    assert updated_first[0]["societal_access_pct__electricity__total"] == pytest.approx(100.0)
    assert updated_second[0]["societal_access_pct__electricity__total"] == pytest.approx(100.0)
    assert "Sample points" not in first_output
    assert "Sample points" not in second_output


def test_electricity_metric_present_when_all_msls_providers_fail():
    """The 'electricity' key must be present even when every MSLS provider is down."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    cache = {}
    get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_all_fail")

    assets = gpd.GeoDataFrame(
        {"type": ["msls"]},
        geometry=[Point(5, 5)],
        crs="EPSG:28992",
    )
    summary = [{"timestep": 0, "map": 0}]
    detailed = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_all_fail",
        "operational": np.array([False]),
        "island_id": np.array([1]),
    }]

    updated, _ = postprocess_societal_access_results(
        summary_results=summary,
        detailed_results=detailed,
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=cache,
        taxonomy={"msls": "electricity"},
        asset_type_column="type",
        all_functions=["electricity"],
    )

    row = updated[0]
    assert "societal_access_pct__electricity__total" in row
    assert row["societal_access_pct__electricity__total"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Strict road_state_key, islands_gdf_cache, distinct allocations
# ---------------------------------------------------------------------------

def test_postprocess_missing_road_state_key_emits_nan_and_warns():
    """Absent road_state_key must emit NaN (not reuse any cached allocation)."""
    pop = _make_population_gdf().iloc[:2].copy()
    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=_make_islands_gdf(),
        road_state_key="roads_a",
    )
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(1, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        # road_state_key deliberately absent
        "operational": [True],
        "island_id": [1],
    }]

    with pytest.warns(RuntimeWarning, match="road_state_key is absent"):
        updated_summary, _ = postprocess_societal_access_results(
            summary_results=summary_results,
            detailed_results=detailed_results,
            gdf_assets=gdf_assets,
            pop_grid_gdf=pop,
            cell_id_column="cell_id",
            allocation_cache={"k": allocation_df},
            all_functions=["hospital"],
        )

    assert math.isnan(updated_summary[0]["societal_access_pct__hospital__total"])


def test_postprocess_builds_allocation_from_islands_gdf_cache():
    """When islands_gdf_cache is provided, allocation is built on the fly."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital", "school"]},
        geometry=[Point(1, 1), Point(12, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_fresh",
        "operational": np.array([True, True]),
        "island_id": np.array([1, 2]),
    }]

    # Start with an empty allocation cache — the allocation must be built
    empty_cache: dict = {}
    updated_summary, updated_cache = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=empty_cache,
        all_functions=["hospital"],
        islands_gdf_cache={"roads_fresh": islands},
        nearest_max_distance=200.0,
    )

    row = updated_summary[0]
    # A numeric (non-NaN) result means the allocation was built successfully.
    assert not math.isnan(row["societal_access_pct__hospital__total"])
    # The new allocation must have been stored in the cache.
    assert len(updated_cache) == 1


def test_postprocess_distinct_allocations_for_different_road_states():
    """Baseline and adapted road states must produce separate cache entries."""
    islands_baseline = _make_islands_gdf()
    # Adapted state: second island geometry slightly shifted
    islands_adapted = gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(11, 0, 21, 10)],
        crs="EPSG:28992",
    )
    pop = _make_population_gdf().iloc[:2].copy()
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital", "hospital"]},
        geometry=[Point(1, 1), Point(12, 1)],
        crs="EPSG:28992",
    )
    summary_results = [
        {"timestep": 0, "map": 0},
        {"timestep": 1, "map": 0},
    ]
    detailed_results = [
        {
            "timestep": 0,
            "map": 0,
            "road_state_key": "baseline_key",
            "operational": np.array([True, True]),
            "island_id": np.array([1, 2]),
        },
        {
            "timestep": 1,
            "map": 0,
            "road_state_key": "adapted_key",
            "operational": np.array([True, True]),
            "island_id": np.array([1, 2]),
        },
    ]
    islands_gdf_cache = {
        "baseline_key": islands_baseline,
        "adapted_key": islands_adapted,
    }

    _, updated_cache = postprocess_societal_access_results(
        summary_results=summary_results,
        detailed_results=detailed_results,
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache={},
        all_functions=["hospital"],
        islands_gdf_cache=islands_gdf_cache,
        nearest_max_distance=200.0,
    )

    # Two distinct road states must produce two distinct allocation entries.
    assert len(updated_cache) == 2
    keys = list(updated_cache.keys())
    assert keys[0] != keys[1]
    road_state_keys = [df.attrs.get("road_state_key") for df in updated_cache.values()]
    assert "baseline_key" in road_state_keys
    assert "adapted_key" in road_state_keys


def test_postprocess_missing_islands_gdf_in_cache_emits_nan_and_warns():
    """If islands_gdf_cache exists but lacks the road_state_key, emit NaN."""
    pop = _make_population_gdf().iloc[:1].copy()
    gdf_assets = gpd.GeoDataFrame(
        {"type": ["hospital"]},
        geometry=[Point(1, 1)],
        crs="EPSG:28992",
    )
    summary_results = [{"timestep": 0, "map": 0}]
    detailed_results = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "unknown_state",
        "operational": [True],
        "island_id": [1],
    }]

    with pytest.warns(RuntimeWarning, match="no islands_gdf found"):
        updated_summary, _ = postprocess_societal_access_results(
            summary_results=summary_results,
            detailed_results=detailed_results,
            gdf_assets=gdf_assets,
            pop_grid_gdf=pop,
            cell_id_column="cell_id",
            allocation_cache={},
            all_functions=["hospital"],
            islands_gdf_cache={"some_other_key": _make_islands_gdf()},
        )

    assert math.isnan(updated_summary[0]["societal_access_pct__hospital__total"])


# ---------------------------------------------------------------------------
# Hospital failure-mode test (first required regression test)
# ---------------------------------------------------------------------------

def test_hospital_not_flooded_disrupted_voronoi():
    """First failure-mode regression test: fake hospital Hospital_not_flooded_disrupted_voronoi.

    Scenario
    --------
    - Hospital centroid: (82526.21, 455532.75)
    - Hospital is NOT flooded (hazard value = 0).
    - Road network remains CONNECTED (single island).
    - All population starts on the same road island as the hospital.
    - Supporting MSLS station is initially operational.

    Before MSLS failure
      hospital flooded: False
      roads disconnected: False
      MSLS operational: True
      hospital operational: True
      hospital access: 100 %

    After MSLS failure (pairwise dependency blocks the hospital)
      hospital flooded: False
      roads disconnected: False   (island_id unchanged)
      MSLS operational: False
      hospital operational: False (blocked by dependency)
      hospital access: 0 %
    """
    from src.dependency_evaluator import evaluate_dependencies

    # Assets: index 0 = supporting MSLS station, index 1 = fake hospital
    asset_types = np.array(["msls", "hospital"])
    hazard_values = np.array([0.0, 0.0])       # neither asset is flooded
    island_ids = np.array([1, 1])              # both on the same road island

    # Synthetic road island covering the fake hospital centroid
    islands = gpd.GeoDataFrame(
        {"island_id": [1]},
        geometry=[box(80000, 453000, 86000, 458000)],
        crs="EPSG:28992",
    )

    # Population on the same island
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["pop_cell"],
            "aantal_inwoners": [1000],
            "aantal_inwoners_65_jaar_en_ouder": [200],
            "aantal_inwoners_0_tot_15_jaar": [150],
            "aantal_inwoners_25_tot_45_jaar": [300],
        },
        geometry=[box(80500, 454500, 82000, 455500)],
        crs="EPSG:28992",
    )

    gdf_assets = gpd.GeoDataFrame(
        {"type": ["msls", "hospital"]},
        geometry=[
            Point(82000.0, 455000.0),      # MSLS station
            Point(82526.21, 455532.75),    # fake hospital centroid
        ],
        crs="EPSG:28992",
    )

    # Pre-build the population->island allocation (road state key stays constant
    # throughout the test because roads do not disconnect)
    allocation_cache: dict = {}
    get_or_build_allocation(
        allocation_cache, pop, "cell_id", islands, road_state_key="roads_connected"
    )

    # MSLS -> hospital pairwise dependency: when MSLS (index 0) is non-operational,
    # the hospital (index 1) is blocked.
    pairwise_deps = [(0, 1)]

    # --- BEFORE MSLS failure ---
    op_initial = np.array([True, True])
    op_before, _ = evaluate_dependencies(
        op_initial.copy(),
        asset_types,
        hazard_values=hazard_values,
        pairwise_dependencies=pairwise_deps,
        return_report=True,
    )
    hospital_operational_before = bool(op_before[1])

    roads_remain_connected = bool(island_ids[0] == island_ids[1])

    updated_before, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads_connected",
            "operational": op_before.astype(int),
            "island_id": island_ids.copy(),
        }],
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"hospital": "hospital"},
        all_functions=["hospital"],
    )
    hospital_access_before = updated_before[0]["societal_access_pct__hospital__total"]

    # --- AFTER MSLS failure (hospital physically intact, roads intact) ---
    op_msls_failed = np.array([False, True])   # only MSLS fails; hospital not flooded
    op_after, _ = evaluate_dependencies(
        op_msls_failed.copy(),
        asset_types,
        hazard_values=hazard_values,
        pairwise_dependencies=pairwise_deps,
        return_report=True,
    )
    hospital_operational_after = bool(op_after[1])

    updated_after, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 1, "map": 0}],
        detailed_results=[{
            "timestep": 1,
            "map": 0,
            "road_state_key": "roads_connected",  # roads still connected
            "operational": op_after.astype(int),
            "island_id": island_ids.copy(),        # island unchanged (no road disruption)
        }],
        gdf_assets=gdf_assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"hospital": "hospital"},
        all_functions=["hospital"],
    )
    hospital_access_after = updated_after[0]["societal_access_pct__hospital__total"]

    # Verify scenario preconditions
    assert hospital_operational_before is True, "Hospital must be operational before MSLS failure"
    assert hospital_operational_after is False, "Hospital must be non-operational after MSLS failure"
    assert roads_remain_connected is True, "Roads must stay connected throughout"

    # Core assertions: access drops from 100 % to 0 % due to dependency alone
    assert hospital_access_before == pytest.approx(100.0), (
        "All population should have hospital access before MSLS failure"
    )
    assert hospital_access_after == pytest.approx(0.0), (
        "No population should have hospital access after MSLS failure"
    )


def test_simulation_smoke_builds_allocation_cache_and_finite_hospital_ema(monkeypatch, tmp_path, capsys):
    hazard_dir = tmp_path / "hazard_case"
    hazard_dir.mkdir(parents=True, exist_ok=True)
    config = get_config(root_dir=tmp_path, hazard_dir_override=hazard_dir)
    config["simulation_config"]["verbose"] = False
    config["simulation_config"]["accessibility_model"] = None
    config["dependency_parameters"]["knowledge_graph"] = [
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": None,
            "relationship": "direct",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "repair_complete"},
            },
        },
        {
            "hazard_type": "flooding",
            "asset_type_a": "hospital",
            "asset_type_b": None,
            "relationship": "direct",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "repair_complete"},
            },
        },
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "hospital",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        },
    ]
    config["dependency_parameters"]["service_area_map"] = None

    gdf_assets = gpd.GeoDataFrame(
        {"type": ["msls", "hospital"]},
        geometry=[Point(0, 0), Point(5, 0)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["pop_cell"],
            "aantal_inwoners": [100],
            "aantal_inwoners_65_jaar_en_ouder": [20],
            "aantal_inwoners_0_tot_15_jaar": [15],
            "aantal_inwoners_25_tot_45_jaar": [40],
        },
        geometry=[box(-2, -2, 8, 2)],
        crs="EPSG:28992",
    )

    monkeypatch.setattr(
        simulation_module,
        "find_hazard_value_at_points_optimized",
        lambda hazard_map, temp_gdf, map_counter, **kwargs: temp_gdf.assign(
            **{f"EV{map_counter}_ma": 0.0}
        ),
    )
    monkeypatch.setattr(
        simulation_module,
        "match_assets_access",
        lambda *args, **kwargs: (
            np.array([10, 10]),
            [],
            [],
            {10: 100.0},
        ),
    )
    monkeypatch.setattr(
        island_analysis,
        "compute_island_geodataframe_from_graph",
        lambda *args, **kwargs: gpd.GeoDataFrame(
            {
                "rfid": [10],
                "island_id": [1],
                "length_m": [100.0],
            },
            geometry=[LineString([(0, 0), (10, 0)])],
            crs="EPSG:28992",
        ),
    )
    accessibility_calls = {"count": 0}
    monkeypatch.setattr(
        grid_hex_module,
        "accessibility_model",
        lambda *args, **kwargs: accessibility_calls.__setitem__(
            "count", accessibility_calls["count"] + 1
        ) or [True] * len(args[0]),
    )

    societal_access_config = {
        "pop_grid_gdf": pop,
        "cell_id_column": "cell_id",
        "taxonomy": {"hospital": "hospital", "msls": "electricity"},
        "all_functions": ["hospital"],
        "allocation_cache": {},
        "fail_on_missing_allocation": True,
    }
    hazard_maps = [hazard_dir / "hazard_0.tif"]

    all_results, _, cache_updated = simulate_asset_damage_recovery_access_breakdown(
        gdf_assets=gdf_assets.copy(),
        hazard_maps=hazard_maps,
        number_repair_crews=1,
        repair_crew_assignment_method="island",
        flood_threshold=0.2,
        root_dir=tmp_path,
        config=config,
        major_timestep=1,
        timestep_output=True,
        societal_access_config=societal_access_config,
        verbose=False,
    )
    first_output = capsys.readouterr().out

    summary_row = all_results[0][1][0]
    detail_row = all_results[0][2][0]
    assert summary_row["allocation_road_state_key"] == detail_row["road_state_key"]
    assert summary_row["allocation_cache_key"] in cache_updated["societal_allocation_cache"]
    assert math.isfinite(summary_row["societal_access_pct__hospital__total"])
    assert summary_row["societal_access_pct__hospital__total"] == pytest.approx(100.0)
    assert all("islands_gdf" not in entry for entry in cache_updated["island_cache"].values())
    assert accessibility_calls["count"] == 0

    ema_result = simulate_asset_damage_recovery_access_breakdown_ema(
        gdf_assets=gdf_assets.copy(),
        hazard_maps=hazard_maps,
        number_repair_crews=1,
        repair_crew_assignment_method="island",
        flood_threshold=0.2,
        root_dir=tmp_path,
        config=config,
        major_timestep=1,
        timestep_output=True,
        societal_access_config={
            **societal_access_config,
            "allocation_cache": cache_updated["societal_allocation_cache"],
        },
        verbose=False,
    )
    second_output = capsys.readouterr().out

    assert math.isfinite(float(ema_result["societal_access_pct__hospital__total"][0]))
    assert float(ema_result["societal_access_pct__hospital__total"][0]) == pytest.approx(100.0)
    combined_output = first_output + second_output
    assert "Successfully resolved islands" not in combined_output
    assert "Loading hazard graph from" not in combined_output
    assert "Saved island cache:" not in combined_output
