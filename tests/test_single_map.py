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


def test_failure_status_matches_default_fragility(assets, monkeypatch):
    depth = np.linspace(0, 1.5, len(assets))
    types = assets["type"].to_numpy()
    prob = single_map.failure_probability(depth, types, k=6.0)

    monkeypatch.setattr(np.random, "random", lambda size: np.full(size, 0.5))
    reference = default_fragility_function(depth, types, k=6.0)
    assert np.array_equal(single_map.operational_status(prob, np.full(len(depth), 0.5)), reference)
    assert (prob[depth == 0] == 0).all()
