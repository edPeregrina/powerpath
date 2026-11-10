import pickle
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterstats import zonal_stats
from shapely.geometry import LineString

from src.ra2ce_fast_overlay import max_all_touched

REPO = Path(__file__).resolve().parent.parent
FIXTURE_GRAPH = REPO / "data" / "test_samples" / "static" / "output_graph" / "base_graph_hazard_editted.p"
FIXTURE_TIF = REPO / "data" / "test_samples" / "test_hazard_timesteps" / "Frag_timesteps_test2.tif"


def _stock_max(geom, arr, transform, nodata):
    out = zonal_stats(geom, arr, affine=transform, all_touched=True, stats="max", nodata=nodata)
    return out[0]["max"]


@pytest.mark.parametrize("nodata", [None, 0.0, -9999.0])
def test_max_all_touched_matches_rasterstats(nodata):
    rng = np.random.default_rng(7)
    arr = np.where(rng.random((60, 80)) < 0.15, rng.random((60, 80)) * 3, 0.0)
    arr[rng.random(arr.shape) < 0.02] = np.nan
    if nodata not in (None, 0.0):
        arr[rng.random(arr.shape) < 0.03] = nodata
    transform = Affine(0.001, 0, 4.0, 0, -0.001, 52.0)

    lines = []
    for _ in range(150):
        # include lines partly/fully outside the raster and tiny sub-pixel ones
        x0, y0 = 3.99 + rng.random() * 0.1, 51.93 + rng.random() * 0.08
        dx, dy = (rng.random(2) - 0.5) * rng.choice([0.0004, 0.01, 0.03])
        lines.append(LineString([(x0, y0), (x0 + dx, y0 + dy)]))

    for line in lines:
        expected = _stock_max(line, arr, transform, nodata)
        got = max_all_touched(line, arr, transform, nodata)
        # ra2ce collapses None and 0.0 to 0; compare in that same space
        assert (got or 0) == pytest.approx(expected or 0, abs=0)


def test_prefilter_does_not_change_result():
    arr = np.zeros((30, 30))
    arr[10:12, 10:12] = 0.8
    transform = Affine(1, 0, 0, 0, -1, 30)
    line = LineString([(5, 20), (25, 20)])
    assert max_all_touched(line, arr, transform, prefilter=True) == max_all_touched(
        line, arr, transform, prefilter=False
    )


def test_patched_ra2ce_overlay_reproduces_fixture_graph_values():
    pytest.importorskip("ra2ce")
    from ra2ce.network.hazard.hazard_intersect.hazard_intersect_builder_for_tif import (
        HazardIntersectBuilderForTif,
    )
    from src.ra2ce_fast_overlay import patch_ra2ce_hazard_overlay, unpatch

    with open(FIXTURE_GRAPH, "rb") as f:
        graph = pickle.load(f)
    reference = {(u, v, k): d["EV2_ma"] for u, v, k, d in graph.edges(keys=True, data=True)}
    for _, _, d in graph.edges(data=True):
        for col in [c for c in d if c.startswith("EV")]:
            del d[col]

    patch_ra2ce_hazard_overlay(fraction_flooded="skip", fast_max=True)
    try:
        builder = HazardIntersectBuilderForTif(
            hazard_aggregate_wl="max",
            hazard_names=["t2"],
            ra2ce_names=["EV2"],
            hazard_tif_files=[FIXTURE_TIF],
        )
        result = builder.get_intersection(graph)
    finally:
        unpatch()

    for u, v, k, d in result.edges(keys=True, data=True):
        assert d["EV2_ma"] == reference[(u, v, k)]
        assert d["EV2_fr"] == 0.0
