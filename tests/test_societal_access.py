import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

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


def test_list_societal_metric_names_contains_expected_patterns():
    names = list_societal_metric_names(
        ["health"],
        {"total": "aantal_inwoners", "elderly": "aantal_inwoners_65_jaar_en_ouder"},
    )
    assert "societal_access_pct__health__total" in names
    assert "societal_total_population__elderly" in names
    assert "societal_equity_absolute_gap__health__elderly" in names
