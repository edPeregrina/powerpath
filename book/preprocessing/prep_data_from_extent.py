"""
Road-network preprocessing chain for a hazard-timestep raster series.

This script:

1. Builds a buffered extent polygon from the first (non-blank) hazard raster.
2. Runs RA2CE (network build + hazard overlay) over the *real* hazard
   timesteps to obtain ``base_graph_hazard.p``.
3. Zero-pads the RA2CE hazard columns for the timesteps it wasn't given
   (``EV0_ma``, the pre-event baseline, and the post-event recovery period)
   up to ``number_of_timesteps`` so every timestep has an explicit value.
   RA2CE numbers overlays by list position (``EV1_ma`` for the first file
   given to it), so feeding it the real hazard files in filename order
   already makes ``EV{n}_ma`` line up with ``Frag_timesteps_test{n}.tif``.
4. Creates matching blank raster copies on disk for timestep indices that
   don't already exist, up to ``number_of_timesteps``.

It is configured, by default, for the repository's small test fixture
(``data/test_samples/test_hazard_timesteps``), which only has hazard values
in ``Frag_timesteps_test1.tif`` .. ``Frag_timesteps_test5.tif`` (indices 0
and 6-9 are blank/all-zero) and only needs 10 timesteps (0-9), unlike the
full production dataset which goes up to 33. Adjust ``number_of_timesteps``
and the ``config``/``get_development_config`` call below to point at the
production dataset if this script is reused for a full run.
"""

from pathlib import Path
import pickle
import re
import sys

import geopandas as gpd
from rasterio import open as rio_open
from shapely.geometry import box

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

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))

from config import get_development_config  # noqa: E402


def get_all_files(directory) -> list[Path]:
    p = Path(directory)
    return [file for file in p.iterdir() if file.is_file()]


def read_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def _timestep_index(path: Path) -> int:
    """Extract the trailing integer timestep index from a hazard filename."""
    match = re.search(r"(\d+)(?=\.tif$)", path.name)
    if match is None:
        raise ValueError(f"Could not find a timestep index in {path.name}")
    return int(match.group(1))


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

    simulation_bounding_box_gdf = gpd.GeoDataFrame(geometry=[buffered_box], crs=hazard_crs)
    extent_path = network_path / "extent.shp"
    network_path.mkdir(parents=True, exist_ok=True)
    simulation_bounding_box_gdf.to_file(extent_path, driver="ESRI Shapefile")
    return extent_path, hazard_crs


def run_ra2ce(extent_path: Path, hazard_files: list[Path], hazard_crs, root_dir: Path, static_path: Path):
    """Build the RA2CE road network and overlay it with the hazard timesteps."""
    network_section = NetworkSection(
        network_type=NetworkTypeEnum.DRIVE,
        source=SourceEnum.OSM_DOWNLOAD,
        polygon=extent_path,
        save_gpkg=True,
        road_types=[
            RoadTypeEnum.MOTORWAY, RoadTypeEnum.MOTORWAY_LINK, RoadTypeEnum.PRIMARY,
            RoadTypeEnum.PRIMARY_LINK, RoadTypeEnum.TRUNK, RoadTypeEnum.SECONDARY,
            RoadTypeEnum.SECONDARY_LINK, RoadTypeEnum.TERTIARY, RoadTypeEnum.RESIDENTIAL,
            RoadTypeEnum.LIVING_STREET, RoadTypeEnum.UNCLASSIFIED,
        ],
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
        output_path=static_path.joinpath("output_graph"),
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


def fill_blank_hazard_rasters(hazard_files_dir: Path, blank_template_path: Path, number_of_timesteps: int):
    """
    Ensure a raster file exists on disk for every timestep index up to
    ``number_of_timesteps - 1``, so downstream steps that iterate hazard
    maps by timestep (e.g. animations) have a file for the recovery period
    too. Missing indices are filled with a zero-valued copy of the blank
    template (timestep 0).
    """
    with rio_open(blank_template_path) as blank_src:
        blank_array = blank_src.read(1) * 0
        blank_meta = blank_src.meta.copy()

    existing_indices = {_timestep_index(p) for p in hazard_files_dir.glob("*.tif")}
    created = []
    for i in range(number_of_timesteps):
        if i in existing_indices:
            continue
        out_path = hazard_files_dir / f"{blank_template_path.stem[:-1]}{i}{blank_template_path.suffix}"
        with rio_open(out_path, "w", **blank_meta) as dst:
            dst.write(blank_array, 1)
        created.append(out_path)
    return created


def main():
    # For the small repository test fixture. Swap for a production config /
    # pass hazard_dir_override to point at the full dataset instead.
    config = get_development_config()

    root_dir = config["data_dir"]
    static_path = root_dir / "static"
    network_path = static_path / "network"
    output_path = static_path / "output_graph"
    hazard_files_dir = config["hazard_dir"]

    # Test dataset only has meaningful hazard values in indices 1-5; indices
    # 0 and 6-9 are blank. The full production dataset instead runs to 33
    # timesteps - adjust this when reusing the script for that dataset.
    number_of_timesteps = 10

    all_hazard_files = sorted(hazard_files_dir.glob("*.tif"), key=_timestep_index)
    blank_template_path = next(p for p in all_hazard_files if _timestep_index(p) == 0)

    real_hazard_files = []
    with rio_open(blank_template_path) as blank_src:
        for p in all_hazard_files:
            idx = _timestep_index(p)
            if idx == 0:
                continue
            with rio_open(p) as src:
                if src.read(1).any():
                    real_hazard_files.append(p)

    print(f"Using {len(real_hazard_files)} non-blank hazard timesteps for RA2CE:")
    for p in real_hazard_files:
        print(f"  {p.name}")

    extent_path, hazard_crs = build_buffered_extent(real_hazard_files[0], network_path)
    run_ra2ce(extent_path, real_hazard_files, hazard_crs, root_dir, static_path)

    output_graph = zero_pad_hazard_graph(output_path, number_of_timesteps)
    print(f"Wrote renumbered/zero-padded hazard graph to {output_graph}")

    created = fill_blank_hazard_rasters(hazard_files_dir, blank_template_path, number_of_timesteps)
    if created:
        print(f"Created {len(created)} blank raster(s) for missing timesteps: {[p.name for p in created]}")


if __name__ == "__main__":
    main()
