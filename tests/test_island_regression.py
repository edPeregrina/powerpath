"""Frozen-behaviour regression test for road-island computation.

The golden digests in ``tests/data/island_golden.json`` were generated from the
implementation *before* ``filter_hazard_graph`` and
``compute_island_geodataframe_from_graph`` were refactored onto shared array
code. They pin the full output (columns, dtypes, row order, every attribute
value, buffered and original geometries) so multi-map runs stay reproducible.

Regenerate ONLY when an intentional behaviour change is made (and say so in
the commit message):  ``set REGEN_GOLDEN=1`` then run this test module.
Geometry digests are exact WKB hashes, so a GEOS upgrade that changes
``buffer`` output would require a regeneration as well.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("geopandas")
pytest.importorskip("networkx")

import geopandas as gpd  # noqa: E402
import networkx as nx  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import LineString, box  # noqa: E402

from src.island_analysis import (  # noqa: E402
    _load_and_project_graph,
    compute_island_geodataframe_from_graph,
)
from src.utils import filter_hazard_graph  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIXTURE_GRAPH = REPO / "data" / "test_samples" / "static" / "output_graph" / "base_graph_hazard_editted.p"
ADAPT = REPO / "data" / "test_samples" / "adaptation"
L1_LAYERS = {
    "none": None,
    "polygon": ADAPT / "l1_adapt_polygon.geojson",
    "monetary30_0.4": ADAPT / "L1_adaptations" / "policy_L1_area_monetary_top30_red_0.4.geojson",
    "population30_0.4": ADAPT / "L1_adaptations" / "policy_L1_area_population_top30_red_0.4.geojson",
}
GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "island_golden.json"
GEOM_COLS = ("geometry", "original_geometry")


def _sha(parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
        h.update(b"\x1f")
    return h.hexdigest()


def digest_islands_gdf(gdf):
    plain = gdf.drop(columns=[c for c in GEOM_COLS if c in gdf.columns])
    parts = []
    for col in plain.columns:
        parts.append(col)
        parts.extend(plain[col].astype(str).tolist())
    return {
        "columns": list(gdf.columns),
        "dtypes": [str(t) for t in gdf.dtypes],
        "n_rows": len(gdf),
        "index": _sha(gdf.index.astype(str).tolist()),
        "values": _sha(parts),
        "geometry": _sha(shapely.to_wkb(gdf.geometry.values)),
        "original_geometry": _sha(shapely.to_wkb(gdf["original_geometry"].values)),
        "n_islands": int(gdf["island_id"].nunique()),
    }


def digest_filtered_graph(G):
    return {
        "nodes": _sha(sorted(map(str, G.nodes))),
        "edges": _sha(sorted(str(e) for e in G.edges(keys=True) if True) if G.is_multigraph()
                      else sorted(str(e) for e in G.edges)),
        "n_edges": G.number_of_edges(),
    }


def _fixture_footprint():
    G = _load_and_project_graph(FIXTURE_GRAPH)
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = x1 - x0, y1 - y0
    return box(x0 + 0.2 * dx, y0 + 0.2 * dy, x1 - 0.2 * dx, y1 - 0.2 * dy)


def island_cases():
    cases = {}
    for col in ("EV2_ma", "EV5_ma"):
        for thr in (0.0, 0.2, 0.6):
            for l1_name in L1_LAYERS:
                cases[f"islands|{col}|{thr}|{l1_name}"] = dict(column=col, threshold=thr, l1=l1_name)
    for policy in ("strict", "permissive"):
        for l1_name in ("none", "monetary30_0.4"):
            cases[f"islands|EV2_ma|0.2|{l1_name}|footprint-{policy}"] = dict(
                column="EV2_ma", threshold=0.2, l1=l1_name, policy=policy
            )
    return cases


def run_island_case(case):
    kwargs = {}
    if case.get("policy"):
        kwargs.update(event_footprint=_fixture_footprint(), outside_footprint_policy=case["policy"])
    gdf = compute_island_geodataframe_from_graph(
        FIXTURE_GRAPH,
        hazard_threshold=case["threshold"],
        hazard_column=case["column"],
        buffer_distance=20,
        l1_area_geojson=L1_LAYERS[case["l1"]],
        **kwargs,
    )
    return digest_islands_gdf(gdf)


def filter_cases():
    cases = {}
    for thr in (0.0, 0.2, 0.6):
        for l1_name in ("none", "monetary30_0.4"):
            cases[f"filter|EV2_ma|{thr}|{l1_name}"] = dict(column="EV2_ma", threshold=thr, l1=l1_name)
    cases["filter|EV2_ma|0.2|strict"] = dict(column="EV2_ma", threshold=0.2, l1="none", policy="strict")
    cases["filter|small-simple-graph"] = dict(simple=True)
    return cases


def run_filter_case(case):
    if case.get("simple"):
        g = nx.Graph()
        for i, x in enumerate([0.0, 1.0, 2.0, 3.0]):
            g.add_node(i, x=x, y=0.0)
        g.add_edge(0, 1, geometry=LineString([(0, 0), (1, 0)]), EV0_ma=0.1, highway="residential")
        g.add_edge(1, 2, geometry=LineString([(1, 0), (2, 0)]), EV0_ma=0.5, highway="residential")
        g.add_edge(2, 3, geometry=LineString([(2, 0), (3, 0)]), EV0_ma=0.9, highway="motorway")
        l1 = gpd.GeoDataFrame({"depth_red": [0.4]}, geometry=[box(0.9, -0.1, 2.1, 0.1)], crs="EPSG:4326")
        return digest_filtered_graph(filter_hazard_graph(g, 0.2, "EV0_ma", l1_area_geojson=l1))
    kwargs = {}
    if case.get("policy"):
        kwargs.update(event_footprint=_fixture_footprint(), outside_footprint_policy=case["policy"])
    G = _load_and_project_graph(FIXTURE_GRAPH)
    G = filter_hazard_graph(
        G, case["threshold"], case["column"], l1_area_geojson=L1_LAYERS[case["l1"]], **kwargs
    )
    return digest_filtered_graph(G)


def _all_cases():
    return {**{k: (run_island_case, v) for k, v in island_cases().items()},
            **{k: (run_filter_case, v) for k, v in filter_cases().items()}}


def _compute_all():
    return {name: runner(case) for name, (runner, case) in _all_cases().items()}


@pytest.fixture(scope="module")
def current():
    return _compute_all()


@pytest.fixture(scope="module")
def golden(current):
    if os.environ.get("REGEN_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(current, indent=1, sort_keys=True))
    if not GOLDEN_PATH.exists():
        pytest.fail(f"Missing {GOLDEN_PATH}; generate it with REGEN_GOLDEN=1 on known-good code")
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("name", sorted(_all_cases()))
def test_matches_golden(name, current, golden):
    assert name in golden, f"{name} missing from golden file"
    assert current[name] == golden[name]


def test_golden_covers_every_case(golden):
    assert set(golden) == set(_all_cases())


def test_l1_and_footprint_cases_are_not_vacuous(golden):
    """The snapshot only protects behaviour the fixture actually exercises."""
    assert golden["islands|EV2_ma|0.2|monetary30_0.4"] != golden["islands|EV2_ma|0.2|none"] or \
        golden["islands|EV2_ma|0.2|polygon"] != golden["islands|EV2_ma|0.2|none"]
    assert golden["islands|EV2_ma|0.2|none|footprint-strict"] != golden["islands|EV2_ma|0.2|none|footprint-permissive"]
    assert golden["islands|EV2_ma|0.2|none"]["n_islands"] > 1
