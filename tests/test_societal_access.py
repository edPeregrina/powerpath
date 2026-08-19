import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from src.caching import (
    create_island_cache_key,
    load_simulation_caches,
    save_simulation_caches,
)
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


def test_compute_access_matrix_handles_explicit_functions():
    island_function_map = {1: frozenset(["health"]), 2: frozenset(["education"])}
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
        all_functions=["education", "health", "repair_logistics"],
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
        all_functions=["education", "emergency_response", "health"],
        nearest_max_distance=2.0,
    )

    row = updated_summary[0]
    assert row["societal_total_population__total"] == pytest.approx(150.0)
    assert row["societal_access_pct__health__total"] == pytest.approx(66.67, abs=0.01)
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
        all_functions=["health"],
    )

    assert math.isnan(updated_summary[0]["societal_access_pct__health__total"])


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
        all_functions=["health"],
    )

    assert math.isnan(
        updated_summary[0]["societal_access_pct__health__total"]
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
        all_functions=["health"],
    )

    assert math.isnan(
        updated_summary[0]["societal_access_pct__health__total"]
    )


def test_list_societal_metric_names_contains_expected_patterns():
    names = list_societal_metric_names(
        ["health"],
        {"total": "aantal_inwoners", "elderly": "aantal_inwoners_65_jaar_en_ouder"},
    )
    assert "societal_access_pct__health__total" in names
    assert "societal_total_population__elderly" in names
    assert "societal_equity_absolute_gap__health__elderly" in names


# ---------------------------------------------------------------------------
# MSLS-only electricity access tests
# ---------------------------------------------------------------------------

def _make_mixed_assets_gdf():
    """Assets with LS, MS, and MSLS types."""
    return gpd.GeoDataFrame(
        {"type": ["ls", "ms", "msls", "msls"]},
        geometry=[Point(1, 1), Point(12, 1), Point(3, 3), Point(14, 3)],
        crs="EPSG:28992",
    )


def test_ls_and_ms_assets_do_not_provide_electricity_access():
    """LS and MS assets must not contribute electricity access."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()

    # Only LS and MS assets — no MSLS
    ls_ms_assets = gpd.GeoDataFrame(
        {"type": ["ls", "ms"]},
        geometry=[Point(1, 1), Point(12, 1)],
        crs="EPSG:28992",
    )

    # Filter to msls only (empty)
    msls_services = ls_ms_assets.loc[ls_ms_assets["type"].eq("msls"), ["type", "geometry"]].copy()
    assert msls_services.empty

    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=islands,
        road_state_key="roads_ls_ms",
    )
    allocation_with_pop = apply_population_to_allocations(allocation_df, pop, "cell_id")

    # island_function_map is empty because no msls assets exist
    island_function_map = {}

    access = compute_access_matrix(
        island_function_map=island_function_map,
        island_population_df=allocation_with_pop,
        pop_columns={"total": "aantal_inwoners"},
        island_id_column="island_id",
        all_functions=["electricity"],
    )

    assert access.loc["electricity", "total"] == 0.0


def test_operational_msls_provides_electricity_access():
    """Operational MSLS assets grant electricity access to their road island."""
    islands = _make_islands_gdf()
    pop = _make_population_gdf().iloc[:2].copy()
    allocation_df = build_origin_island_allocations(
        pop,
        cell_id_column="cell_id",
        islands_gdf=islands,
        road_state_key="roads_ok",
    )
    cache = {}
    get_or_build_allocation(cache, pop, "cell_id", islands, road_state_key="roads_ok")

    assets = _make_mixed_assets_gdf()
    summary = [{"timestep": 0, "map": 0}]
    detailed = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_ok",
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
# New tests: strict road_state_key, islands_gdf_cache, distinct allocations
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
            all_functions=["health"],
        )

    assert math.isnan(updated_summary[0]["societal_access_pct__health__total"])


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
        all_functions=["health"],
        islands_gdf_cache={"roads_fresh": islands},
        nearest_max_distance=200.0,
    )

    row = updated_summary[0]
    # A numeric (non-NaN) result means the allocation was built successfully.
    assert not math.isnan(row["societal_access_pct__health__total"])
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
        all_functions=["health"],
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
            all_functions=["health"],
            islands_gdf_cache={"some_other_key": _make_islands_gdf()},
        )

    assert math.isnan(updated_summary[0]["societal_access_pct__health__total"])
