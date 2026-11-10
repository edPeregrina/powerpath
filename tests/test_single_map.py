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
    assert ids.tolist() == np.asarray(expected).tolist()


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
    from src.simulation import simulate_asset_damage_recovery_access_breakdown

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
    out = single_map.simulate_single_map(
        last["hazard_value"], assets["type"], ids, roads, 5, np.full(len(assets), 0.5), params,
        flood_threshold=threshold, fragility_k=k,
    )
    for key, ref in [("damage_ratio", "damage_ratio"), ("repair_time", "repair_time"),
                     ("intrinsic_operational", "intrinsic_operational")]:
        np.testing.assert_allclose(
            np.asarray(out[key], float)[first_here], np.asarray(last[ref], float)[first_here], err_msg=key
        )


def test_sample_flood_durations():
    u = np.linspace(0, 1, 101)
    h = single_map.sample_flood_durations(u)
    assert h[0] == 0.0 and 12.0 < h[50] < 24.0
    assert (np.diff(h) >= 0).all() and h[-1] == pytest.approx(7 * 24.0)
    h2 = single_map.sample_flood_durations(u, {"min": 0.5, "max": 3.0, "median": 2.0})
    assert h2[0] == pytest.approx(12.0) and h2[-1] == pytest.approx(72.0)


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
