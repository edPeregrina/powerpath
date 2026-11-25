"""Road-network graph management shared by the multi-map and single-map pipelines.

Multi-map (``book/preprocessing/prep_data_from_extent.py``) keeps one graph at a fixed path
(``<hazard_dir>/../static/output_graph``); the functions it uses were moved here unchanged.
Single-map (:func:`build_single_map_graph`) builds one graph per extent and map set, stored under
a hash of both, so a different extent or a changed map can never be served a stale graph.
"""

import hashlib
import logging
import os
import pickle
import re
import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
from rasterio import open as rio_open
from shapely.geometry import box

LOGGER = logging.getLogger(__name__)

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
    """Save ``polygon`` as the RA2CE network extent shapefile (RA2CE needs EPSG:4326) and return its path."""
    extent_path = network_path / "extent.shp"
    network_path.mkdir(parents=True, exist_ok=True)
    extent = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    if crs is not None and crs != "EPSG:4326":
        extent = extent.to_crs("EPSG:4326")
    extent.to_file(extent_path, driver="ESRI Shapefile")
    return extent_path


def run_ra2ce(extent_path: Path, hazard_files: list[Path], hazard_crs, root_dir: Path, static_path: Path,
              road_types=DEFAULT_ROAD_TYPES, skip_fraction_flooded=False):
    """Build the RA2CE road network and overlay it with the hazard timesteps.

    RA2CE numbers the overlay columns by list position: the i-th file becomes ``EV{i+1}_ma``.
    The overlay runs through :mod:`src.ra2ce_fast_overlay` (exact for the max depth,
    ``EV*_ma``). The flooded fraction ``EV*_fr`` is kept unless ``skip_fraction_flooded=True``
    (then it is all 0.0; nothing in the model reads it, but it is useful for artefact checks).
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
    from src.ra2ce_fast_overlay import patch_ra2ce_hazard_overlay, unpatch

    unsupported = [name for name in road_types if not hasattr(RoadTypeEnum, name)]
    if unsupported:
        LOGGER.warning("Installed ra2ce has no road type(s) %s; they are left out of the network.", unsupported)
    road_types = [name for name in road_types if name not in unsupported]

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

    patch_ra2ce_hazard_overlay(fraction_flooded="skip" if skip_fraction_flooded else "fast")
    try:
        handler = Ra2ceHandler.from_config(network_config_data, analysis=None)
        handler.configure()
    finally:
        unpatch()


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


def build_single_map_graph(hazard_maps, root_dir, road_types=DEFAULT_ROAD_TYPES, include_living_streets=True,
                           skip_fraction_flooded=False):
    """Path of the hazard-overlaid graph for these maps (map ``i`` is column ``EV{i}_ma``).

    Stored as ``<root_dir>/static_single/extent_<extent key>/graph_<maps key>.p``. The network
    is shared by every map set with the same extent; the overlay file is keyed by the ordered
    maps, their fingerprints and ``skip_fraction_flooded``, so a changed map gives a new file.
    Existing files are reused. Living streets are included unless ``include_living_streets=False``
    (they make large graphs much bigger and slower, hence the warning). The flooded fraction
    ``EV{i}_fr`` is kept unless ``skip_fraction_flooded=True`` (then it is all 0.0).
    """
    road_types = tuple(t for t in road_types if include_living_streets or t != "LIVING_STREET")
    if "LIVING_STREET" in road_types:
        LOGGER.warning("Road graph includes living streets: large extents get many more edges "
                       "(slower graph build and island computation).")
    maps = [Path(m) for m in hazard_maps]
    extent_key, polygon, crs = single_map_extent_key(maps, road_types)
    extent_dir = Path(root_dir) / "static_single" / f"extent_{extent_key}"
    graph_key = _digest(extent_key, skip_fraction_flooded,
                        *(f"{m.name}:{file_fingerprint(m)}" for m in maps))
    graph_path = extent_dir / f"graph_{graph_key}.p"
    if graph_path.exists():
        return graph_path

    # Concurrent runs on one extent share RA2CE's output folder: build under a lock, then recheck.
    with _extent_lock(extent_dir):
        if graph_path.exists():
            return graph_path
        static_path = extent_dir / "static"
        extent_path = write_extent(polygon, crs, static_path / "network")
        run_ra2ce(extent_path, maps, crs, extent_dir, static_path, road_types,
                  skip_fraction_flooded=skip_fraction_flooded)
        graph = shift_overlay_columns(read_pickle(static_path / "output_graph" / "base_graph_hazard.p"), len(maps))
        tmp = graph_path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as file:
            pickle.dump(graph, file)
        tmp.replace(graph_path)
    return graph_path


@contextmanager
def _extent_lock(extent_dir, timeout_s=6 * 3600, poll_s=5.0):
    """Exclusive lock file ``<extent_dir>/build.lock`` (atomic create, polled).

    A run killed mid-build leaves the file behind; delete it by hand (the timeout error names it).
    """
    extent_dir.mkdir(parents=True, exist_ok=True)
    lock = extent_dir / "build.lock"
    start = time.monotonic()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"Graph build lock held for over {timeout_s} s; delete {lock} if stale")
            LOGGER.info("Waiting for another graph build on this extent (%s)", lock)
            time.sleep(poll_s)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)
