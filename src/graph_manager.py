"""Road-network graph management shared by the multi-map and single-map pipelines.

Multi-map (``book/preprocessing/prep_data_from_extent.py``) keeps one graph at a fixed path
(``<hazard_dir>/../static/output_graph``); the functions it uses were moved here unchanged.
Single-map (:func:`build_single_map_graph`) builds one graph per extent and map set, stored under
a hash of both, so a different extent or a changed map can never be served a stale graph.
"""

import hashlib
import pickle
import re
from pathlib import Path

import geopandas as gpd
from rasterio import open as rio_open
from shapely.geometry import box

DEFAULT_ROAD_TYPES = (
    "MOTORWAY", "MOTORWAY_LINK", "PRIMARY", "PRIMARY_LINK", "TRUNK", "SECONDARY",
    "SECONDARY_LINK", "TERTIARY", "RESIDENTIAL", "LIVING_STREET", "UNCLASSIFIED",
)
GRAPH_FORMAT_VERSION = "1"  # bump to invalidate every single-map graph


def read_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def build_buffered_extent(hazard_map_path: Path, network_path: Path, buffer_frac: float = 0.05):
    """Buffer the hazard raster's bounding box and save it as the RA2CE network extent."""
    with rio_open(hazard_map_path) as src:
        hazard_crs = src.crs
        bounding_box_haz = box(*src.bounds)

    xmin, ymin, xmax, ymax = bounding_box_haz.bounds
    xmin -= (xmax - xmin) * buffer_frac
    xmax += (xmax - xmin) * buffer_frac
    ymin -= (ymax - ymin) * buffer_frac
    ymax += (ymax - ymin) * buffer_frac
    buffered_box = box(xmin, ymin, xmax, ymax)

    return write_extent(buffered_box, hazard_crs, network_path), hazard_crs


def write_extent(polygon, crs, network_path: Path) -> Path:
    """Save ``polygon`` as the RA2CE network extent shapefile and return its path."""
    extent_path = network_path / "extent.shp"
    network_path.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[polygon], crs=crs).to_file(extent_path, driver="ESRI Shapefile")
    return extent_path


def _patch_ra2ce_fraction_flooded():
    """Replace ra2ce's per-edge rasterize+polygonize fraction-flooded overlay
    with a precomputed, STRtree-indexed flood-polygon lookup (built once per
    hazard file instead of once per edge). Exact same result (verified
    bit-identical on the test fixture), ~150-300x faster than stock: the
    per-edge cost drops from re-cropping+vectorizing the raster to an
    `predicate="intersects"` index query plus summed per-candidate
    intersection lengths. No `unary_union` is needed: polygons from
    rasterio.features.shapes() are disjoint by construction, so summing each
    candidate's intersection length independently equals the union's length.
    """
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely import STRtree
    from ra2ce.network.hazard.hazard_intersect.hazard_intersect_builder_for_tif import (
        HazardIntersectBuilderForTif,
    )

    def _fast_fraction_flooded_array(self, line, src):
        cache = getattr(self, "_flood_index_cache", None)
        if cache is None:
            cache = self._flood_index_cache = {}
        entry = cache.get(src.name)
        if entry is None:
            arr = src.read(1)
            flood_bool = arr > 0
            polys = [
                shape(geom)
                for geom, _ in shapes(
                    flood_bool.astype("uint8"), mask=flood_bool, transform=src.transform
                )
            ]
            tree = STRtree(polys) if polys else None
            entry = cache[src.name] = (polys, tree)
        polys, tree = entry
        if tree is None:
            return 0.0
        try:
            idxs = tree.query(line, predicate="intersects")
            if len(idxs) == 0:
                return 0.0
            total = sum(polys[i].intersection(line).length for i in idxs)
            return total / line.length
        except Exception as e:
            print(f"_fast_fraction_flooded_array() {e} for line {line}")
            return 0

    HazardIntersectBuilderForTif._fraction_flooded_array = _fast_fraction_flooded_array


def run_ra2ce(extent_path: Path, hazard_files: list[Path], hazard_crs, root_dir: Path, static_path: Path,
              road_types=DEFAULT_ROAD_TYPES):
    """Build the RA2CE road network and overlay it with the hazard timesteps.

    RA2CE numbers the overlay columns by list position: the i-th file becomes ``EV{i+1}_ma``.
    """
    from ra2ce.network.network_config_data.enums.aggregate_wl_enum import AggregateWlEnum
    from ra2ce.network.network_config_data.enums.source_enum import SourceEnum
    from ra2ce.network.network_config_data.enums.network_type_enum import NetworkTypeEnum
    from ra2ce.network.network_config_data.enums.road_type_enum import RoadTypeEnum
    from ra2ce.network.network_config_data.network_config_data import (
        HazardSection,
        NetworkConfigData,
        NetworkSection,
    )
    from ra2ce.ra2ce_handler import Ra2ceHandler

    _patch_ra2ce_fraction_flooded()

    network_section = NetworkSection(
        network_type=NetworkTypeEnum.DRIVE,
        source=SourceEnum.OSM_DOWNLOAD,
        polygon=extent_path,
        save_gpkg=True,
        # Reuse an existing static/output_graph build instead of ra2ce wiping
        # it via shutil.rmtree (network_config_wrapper.py L69) on every run.
        reuse_network_output=True,
        road_types=[getattr(RoadTypeEnum, name) for name in road_types],
    )

    hazard_section = HazardSection(
        hazard_map=hazard_files,
        hazard_id=None,
        hazard_field_name="waterdepth",
        aggregate_wl=AggregateWlEnum.MAX,
        hazard_crs=hazard_crs,
        overlay_segmented_network=False,
    )

    network_config_data = NetworkConfigData(
        root_path=root_dir,
        static_path=static_path,
        network=network_section,
        hazard=hazard_section,
    )

    handler = Ra2ceHandler.from_config(network_config_data, analysis=None)
    handler.configure()


def zero_pad_hazard_graph(output_path: Path, number_of_timesteps: int):
    """
    Produce ``base_graph_hazard_editted.p`` from RA2CE's ``base_graph_hazard.p``.

    RA2CE numbers hazard overlay columns by *list position*: the ``i``-th
    file passed to ``HazardSection.hazard_map`` becomes ``EV{i+1}_ma``. Since
    ``real_hazard_files`` is fed in filename order starting at
    ``Frag_timesteps_test1.tif``, RA2CE's own numbering already lines up
    with the fixture's timestep indices (``EV1_ma`` <-> ``test1``, ``EV2_ma``
    <-> ``test2``, ...) with **no renumbering needed**. (This differs from
    the original production notebook, which fed a differently-indexed file
    series into RA2CE and needed an ``EV{n}`` -> ``EV{n-1}`` shift; that
    shift must NOT be applied here or every column would be off by one.)

    ``EV0_ma`` (the pre-event baseline, matching the blank
    ``Frag_timesteps_test0.tif``) and every timestep beyond what RA2CE
    produced (the "no more flooding" / recovery period, up to
    ``number_of_timesteps - 1``) are added explicitly as 0.0 so the graph
    carries a value for every step, even though ``filter_hazard_graph``
    would already default a missing hazard column to 0 via
    ``d.get(hazard_column, 0)``.
    """
    input_graph = output_path / "base_graph_hazard.p"
    output_graph = output_path / "base_graph_hazard_editted.p"

    graph_data = read_pickle(input_graph)

    for i in range(number_of_timesteps):
        col_name = f"EV{i}_ma"
        for u, v, key in graph_data.edges(keys=True):
            if col_name not in graph_data[u][v][key]:
                graph_data[u][v][key][col_name] = 0.0

    pickle.dump(graph_data, open(output_graph, "wb"))
    return output_graph


# --- single-map ---------------------------------------------------------------------------

_OVERLAY_COLUMN = re.compile(r"EV(\d+)_(ma|fr)")


def file_fingerprint(path) -> str:
    """Cheap identity of a file (size and mtime); a copy or edit changes it, so caches miss."""
    stat = Path(path).stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"


def _digest(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def single_map_extent_key(hazard_maps, road_types=DEFAULT_ROAD_TYPES):
    """``(key, polygon, crs)``: the union bbox of the maps, so one extent per map set.

    The key covers the bbox, CRS and road selection; maps of differing CRS are rejected.
    """
    bounds, crs = [], None
    for path in hazard_maps:
        with rio_open(path) as src:
            if crs is not None and src.crs != crs:
                raise ValueError(f"Hazard maps have different CRS ({crs} vs {src.crs} in {path})")
            crs = src.crs
            bounds.append(src.bounds)
    xmin, ymin, xmax, ymax = (
        min(b[0] for b in bounds), min(b[1] for b in bounds),
        max(b[2] for b in bounds), max(b[3] for b in bounds),
    )
    key = _digest(GRAPH_FORMAT_VERSION, *(round(v, 6) for v in (xmin, ymin, xmax, ymax)),
                  crs.to_string(), *sorted(road_types))
    return key, box(xmin, ymin, xmax, ymax), crs


def shift_overlay_columns(graph, n_maps):
    """Renumber RA2CE overlay columns ``EV{i+1}_*`` to ``EV{i}_*`` so map ``i`` is ``EV{i}_ma``.

    Columns beyond ``n_maps`` (left over from an earlier, longer map list) are dropped and a
    missing ``EV{i}_ma`` is set to 0.0. Edge attributes are modified in place.
    """
    for _, _, data in graph.edges(data=True):
        old = {k: data.pop(k) for k in [k for k in data if _OVERLAY_COLUMN.fullmatch(k)]}
        for column, value in old.items():
            index, kind = _OVERLAY_COLUMN.fullmatch(column).groups()
            if 1 <= int(index) <= n_maps:
                data[f"EV{int(index) - 1}_{kind}"] = value
        for i in range(n_maps):
            data.setdefault(f"EV{i}_ma", 0.0)
    return graph


def build_single_map_graph(hazard_maps, root_dir, road_types=DEFAULT_ROAD_TYPES):
    """Path of the hazard-overlaid graph for these maps (map ``i`` is column ``EV{i}_ma``).

    Stored as ``<root_dir>/static_single/extent_<extent key>/graph_<maps key>.p``. The network
    is shared by every map set with the same extent; the overlay file is keyed by the ordered
    maps and their fingerprints, so a changed map gives a new file. Existing files are reused.
    """
    maps = [Path(m) for m in hazard_maps]
    extent_key, polygon, crs = single_map_extent_key(maps, road_types)
    extent_dir = Path(root_dir) / "static_single" / f"extent_{extent_key}"
    graph_path = extent_dir / f"graph_{_digest(extent_key, *(f'{m.name}:{file_fingerprint(m)}' for m in maps))}.p"
    if graph_path.exists():
        return graph_path

    static_path = extent_dir / "static"
    extent_path = write_extent(polygon, crs, static_path / "network")
    run_ra2ce(extent_path, maps, crs, extent_dir, static_path, road_types)
    graph = shift_overlay_columns(read_pickle(static_path / "output_graph" / "base_graph_hazard.p"), len(maps))
    tmp = graph_path.with_suffix(".tmp")
    with open(tmp, "wb") as file:
        pickle.dump(graph, file)
    tmp.replace(graph_path)
    return graph_path
