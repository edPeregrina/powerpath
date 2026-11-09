"""Flood overlay: which assets are directly inundated at a given simulation hour.

Companion to ``produce_figures_group_equity.py`` -- a first, purely
geometric step towards decomposing "loss of access" into its three possible
causes (see ``CLAUDE.md``, "Open task: decompose loss of access..."):

  1. **Direct flooding of the provider asset itself** -- what this script
     answers: overlay the raw hazard raster at one simulation hour against
     every asset's location (electricity substations + every OSM-derived
     service amenity) and flag assets whose extracted hazard value exceeds
     ``flood_threshold``.
  2. Loss of a *dependency* (e.g. a hospital's governing substation losing
     power) -- NOT answered here; needs per-timestep ``operational``/
     ``intrinsic_operational`` state from a real simulation run.
  3. *Road-network severance* (population cell's island has no operational
     provider) -- NOT answered here either; a graph-connectivity fact,
     orthogonal to any single asset's own state.

This script only needs the raw hazard rasters + asset geometries (the same
inputs ``book/interdependent_ci_societal_access.py`` loads before building
the EMA model) -- no EMA run or simulation needed. It reuses
``find_hazard_value_at_points_optimized`` so the extraction method/column
exactly matches what the simulation itself uses internally
(``src.simulation._update_hazard_map_states``), rather than re-deriving a
parallel implementation.

Hour-to-hazard-map-index mapping mirrors the simulation exactly:
``map_counter = floor(hour / major_timestep)``. Default ``major_timestep=6``
(the full-extent run's CLI default), so hour 48 -> hazard map index 8.

Usage:
    python book/postprocessing/flood_overlay_hour48.py
    python book/postprocessing/flood_overlay_hour48.py --hour 48 --major-timestep 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from config import get_config  # noqa: E402
from src.data_loader import load_hazard_maps  # noqa: E402
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized  # noqa: E402
from book.interdependent_ci_societal_access import (  # noqa: E402
    AMENITY_CATEGORY_LABELS,
    load_combined_assets,
)
from book.postprocessing.produce_figures_group_equity import (  # noqa: E402
    FUNCTION_LABELS,
    PLOT_STYLE,
)

FIG_DIR = REPO_ROOT / "data" / "output" / "paper_figures_group_equity"

# Distinct marker per asset type (electricity substations + each OSM amenity
# category); shape carries type identity, colour carries flooded/not.
ASSET_MARKERS = {
    "msls": "s",
    "ems": "^",
    "hospital": "P",
    "primary_care": "o",
    "pharmacy": "D",
    "supermarket": "v",
}
ASSET_LABELS = {"msls": "Electricity substation", **AMENITY_CATEGORY_LABELS}
# Edge/outline colour per asset type -- used both for point markers'
# edgecolor and polygon footprints' edgecolor, so type identity is visible
# whether an asset's native geometry is a point or a polygon footprint.
ASSET_TYPE_EDGE_COLORS = {
    "msls": "#4C72B0",
    "ems": "#DD8452",
    "hospital": "#C44E52",
    "primary_care": "#8172B2",
    "pharmacy": "#937860",
    "supermarket": "#55A868",
}

FLOODED_COLOR = "#B22222"
NOT_FLOODED_COLOR = "#D9D9D9"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hour", type=int, default=48,
                         help="Simulation hour to inspect (hourly timestep index). Default: 48.")
    parser.add_argument("--major-timestep", type=int, default=6,
                         help="Hours per hazard map update, matching the EMA run's "
                              "--major-timestep. Default: 6.")
    parser.add_argument("--flood-threshold", type=float, default=None,
                         help="Hazard depth (m) above which an asset is considered flooded. "
                              "Default: config['simulation_config']['flood_threshold'].")
    parser.add_argument("--extraction-method", type=str, default=None,
                         choices=["max", "mean", "median", "point", "centroid"],
                         help="Hazard value extraction method. Default: config's "
                              "analysis_config['hazard_extraction_method'] (normally 'max').")
    parser.add_argument("--dev-config", action="store_true",
                         help="Use get_development_config() (small sample dataset) instead of "
                              "the full-extent get_config().")
    parser.add_argument("--output-dir", type=Path, default=FIG_DIR,
                         help="Directory to write the overlay PNG + per-asset CSV into.")
    parser.add_argument("--savefig-dpi", type=int, default=200,
                         help="PNG resolution; kept lower than the paper figures' 600 dpi "
                              "since this is a diagnostic overlay with a large raster background. "
                              "Default: 200.")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.rcParams.update(PLOT_STYLE)
    plt.rcParams["savefig.dpi"] = args.savefig_dpi

    if args.dev_config:
        from config import get_development_config
        config = get_development_config()
    else:
        config = get_config()

    flood_threshold = args.flood_threshold
    if flood_threshold is None:
        flood_threshold = config["simulation_config"]["flood_threshold"]
    extraction_method = args.extraction_method
    if extraction_method is None:
        extraction_method = config["analysis_config"]["hazard_extraction_method"]

    map_counter = args.hour // args.major_timestep
    print(f"Hour {args.hour} (major_timestep={args.major_timestep}) -> hazard map index {map_counter}")

    print("Loading combined assets (electricity substations + OSM service amenities)...")
    gdf_assets, substation_idx, osm_asset_idx = load_combined_assets(config)

    hazard_maps = load_hazard_maps(config["hazard_dir"])
    if map_counter >= len(hazard_maps):
        raise ValueError(
            f"Hazard map index {map_counter} is out of range; only {len(hazard_maps)} maps available."
        )
    hazard_map_path = hazard_maps[map_counter]
    print(f"Using hazard map: {hazard_map_path}")

    temp_gdf = gdf_assets[["type", "geometry"]].copy()
    result_gdf = find_hazard_value_at_points_optimized(
        hazard_map_path, temp_gdf, map_counter, extraction_method=extraction_method,
    )
    haz_val_str = f"hazard_value_{str(map_counter).zfill(2)}"
    haz_col_str = f"EV{map_counter}_ma"
    hazard_col = haz_val_str if haz_val_str in result_gdf.columns else haz_col_str
    gdf_assets = gdf_assets.copy()
    gdf_assets["hazard_value"] = result_gdf[hazard_col].fillna(0.0).values
    gdf_assets["flooded"] = gdf_assets["hazard_value"] > flood_threshold

    # --- Summary table (per asset type, counts + fraction directly flooded) ---
    summary = (
        gdf_assets.groupby("type")
        .agg(n_assets=("flooded", "size"), n_flooded=("flooded", "sum"),
             max_hazard_value=("hazard_value", "max"))
        .reset_index()
    )
    summary["pct_flooded"] = 100.0 * summary["n_flooded"] / summary["n_assets"]
    summary["label"] = summary["type"].map(ASSET_LABELS).fillna(summary["type"])
    summary = summary[["type", "label", "n_assets", "n_flooded", "pct_flooded", "max_hazard_value"]]
    summary = summary.sort_values("pct_flooded", ascending=False)
    print(summary.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_asset_csv = args.output_dir / f"flood_overlay_hour{args.hour}_per_asset.csv"
    summary_csv = args.output_dir / f"flood_overlay_hour{args.hour}_summary.csv"
    # Keep the native geometry (point or polygon) as WKT rather than
    # collapsing polygons to a centroid -- the hazard extraction itself
    # already used the native polygon extent (zonal max over the footprint),
    # so the record of "what was actually tested" should match that.
    (
        gdf_assets.assign(geometry_wkt=gdf_assets.geometry.to_wkt())
        .drop(columns="geometry")
        .to_csv(per_asset_csv, index=False)
    )
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {per_asset_csv}")
    print(f"Wrote {summary_csv}")

    # --- Overlay figure: hazard raster + asset locations (shape=type, colour=flooded) ---
    fig, ax = plt.subplots(figsize=(11, 10))
    with rasterio.open(hazard_map_path) as src:
        # Hazard value extraction (flood determination above) was already
        # done at full raster resolution against the true asset geometries;
        # this downsampled read is only for the background image render, to
        # keep the output PNG a reasonable size. Block-reduce with max (not
        # mean/"average") so the displayed flood extent isn't visually
        # diluted -- rasterio's direct `read()` only supports warp-style
        # resamplers (nearest/bilinear/average/...) for `out_shape`, not
        # `max`, so block-max pooling is done manually in numpy instead.
        downsample = 4
        full_band = src.read(1, masked=True)
        full_band = np.ma.masked_less(full_band, 0.0).filled(-1.0)
        n_rows = (full_band.shape[0] // downsample) * downsample
        n_cols = (full_band.shape[1] // downsample) * downsample
        trimmed = full_band[:n_rows, :n_cols]
        band = trimmed.reshape(n_rows // downsample, downsample, n_cols // downsample, downsample).max(axis=(1, 3))
        band = np.ma.masked_less(band, 0.0)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    im = ax.imshow(
        band, extent=extent, origin="upper", cmap="Blues",
        vmin=0.0, vmax=max(0.5, float(np.nanpercentile(band.filled(0.0), 99))),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Hazard depth (m)")

    # Draw assets using their native geometry: polygons are drawn as actual
    # footprints (filled by flood status, outlined by asset type), points
    # are drawn as point markers -- no polygon-to-centroid conversion.
    for asset_type, type_color in ASSET_TYPE_EDGE_COLORS.items():
        sub = gdf_assets[gdf_assets["type"] == asset_type]
        if sub.empty:
            continue
        is_poly = sub.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        poly_sub, point_sub = sub[is_poly], sub[~is_poly]

        if not poly_sub.empty:
            fill_colors = np.where(poly_sub["flooded"].values, FLOODED_COLOR, NOT_FLOODED_COLOR)
            poly_sub.plot(ax=ax, facecolor=fill_colors, edgecolor=type_color,
                          linewidth=1.0, alpha=0.9, zorder=3)
        if not point_sub.empty:
            marker = ASSET_MARKERS[asset_type]
            colors = np.where(point_sub["flooded"].values, FLOODED_COLOR, NOT_FLOODED_COLOR)
            xs = [g.x for g in point_sub.geometry]
            ys = [g.y for g in point_sub.geometry]
            ax.scatter(
                xs, ys, marker=marker, c=colors, s=70, linewidths=1.0,
                edgecolors=type_color, zorder=4,
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Directly flooded assets at simulation hour {args.hour} "
        f"(hazard map {map_counter}, threshold {flood_threshold:.1f} m)"
    )

    type_handles = [
        Line2D([0], [0], marker=ASSET_MARKERS[t], color="none", markerfacecolor="none",
               markeredgecolor=color, markeredgewidth=1.6, markersize=10,
               label=FUNCTION_LABELS.get(t, ASSET_LABELS.get(t, t)))
        for t, color in ASSET_TYPE_EDGE_COLORS.items() if t in gdf_assets["type"].unique()
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FLOODED_COLOR,
               markeredgecolor="black", markersize=10, label="Directly flooded"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NOT_FLOODED_COLOR,
               markeredgecolor="black", markersize=10, label="Not directly flooded"),
    ]
    legend1 = ax.legend(handles=type_handles, title="Asset type", loc="upper left",
                         bbox_to_anchor=(1.12, 1.0), fontsize=11, title_fontsize=12)
    ax.add_artist(legend1)
    ax.legend(handles=status_handles, title="Flood status", loc="upper left",
              bbox_to_anchor=(1.12, 0.55), fontsize=11, title_fontsize=12)

    fig_path = args.output_dir / f"flood_overlay_hour{args.hour}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"Wrote {fig_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
