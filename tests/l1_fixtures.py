"""Synthetic L1 adaptation layers so tests never read the live adaptation folders."""

import atexit
import shutil
import tempfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from src.island_analysis import _load_and_project_graph

# Fractions of the graph extent (x0, y0, x1, y1); the two layers hit different areas.
_LAYER_BOXES = {
    "monetary30_0.4": [(0.1, 0.1, 0.35, 0.35), (0.4, 0.4, 0.65, 0.65), (0.7, 0.7, 0.95, 0.95)],
    "population30_0.4": [(0.1, 0.65, 0.35, 0.9), (0.4, 0.1, 0.65, 0.35), (0.7, 0.4, 0.95, 0.65)],
}


def make_l1_layers(graph_path, depth_red=0.4):
    """Write the synthetic L1 geojsons to a temp dir; return ``{name: Path}``."""
    G = _load_and_project_graph(graph_path)
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    x0, y0 = min(xs), min(ys)
    dx, dy = max(xs) - x0, max(ys) - y0

    out_dir = Path(tempfile.mkdtemp(prefix="l1_fixtures_"))
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    layers = {}
    for name, fractions in _LAYER_BOXES.items():
        geoms = [box(x0 + a * dx, y0 + b * dy, x0 + c * dx, y0 + d * dy) for a, b, c, d in fractions]
        gdf = gpd.GeoDataFrame(
            {"id": range(len(geoms)), "depth_red": depth_red}, geometry=geoms, crs="EPSG:28992"
        )
        path = out_dir / f"policy_L1_area_{name}.geojson"
        gdf.to_file(path, driver="GeoJSON")
        layers[name] = path
    return layers
