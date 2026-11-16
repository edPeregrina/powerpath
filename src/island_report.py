"""Spatial-trace outputs shared by single-map and multi-map: long-format trace files and per-map/day island reports."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def asset_population_array(asset_population_map, n_assets):
    """Per-asset served population from an ``asset_id -> population`` dict (0 where missing)."""
    return np.array([(asset_population_map or {}).get(i, 0.0) for i in range(n_assets)], dtype=float)


def write_trace_file(trace_path, columns, asset_geometry, crs, uncertainties):
    """Write one long-format trace (``columns`` incl. ``asset_id``) joined to asset geometry as GeoParquet,
    plus the experiment's sampled uncertainties as ``<stem>_uncertainties.json``."""
    trace = pd.DataFrame(columns).merge(asset_geometry, on="asset_id", how="left")
    trace = gpd.GeoDataFrame(trace, geometry="geometry", crs=crs)
    trace.to_parquet(trace_path)
    trace_path = Path(trace_path)
    with open(trace_path.with_name(trace_path.stem + "_uncertainties.json"), "w", encoding="utf-8") as f:
        json.dump(uncertainties, f, indent=2)
    return trace


def island_distribution(roads, island_ids, operational, asset_population=None):
    """One row per island: road sections, length, assets, non-operational assets, served population.

    ``island_ids``/``operational``/``asset_population`` are per asset (same order); island -1
    collects assets without a road island.
    """
    island_ids = np.asarray(island_ids)
    operational = np.asarray(operational).astype(bool)
    sections = roads.groupby("island_id").agg(n_road_sections=("rfid", "size"), length_km=("length_m", "sum"))
    sections["length_km"] /= 1000.0
    assets = pd.DataFrame({
        "island_id": island_ids,
        "n_assets": 1,
        "n_non_operational": ~operational,
        "population": 0.0 if asset_population is None else np.asarray(asset_population, dtype=float),
    }).groupby("island_id").sum()
    table = sections.join(assets, how="outer").fillna(0).reset_index()
    return table.astype({"n_road_sections": int, "n_assets": int, "n_non_operational": int})


def write_island_report(out_dir, stem, roads, island_ids, operational, asset_population=None, depth_by_rfid=None):
    """Write ``<stem>_road_islands.parquet`` (road sections coloured by island, GIS-ready) and
    ``<stem>_island_distribution.csv``. Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = roads.copy()
    if depth_by_rfid is not None:
        sections["peak_depth_m"] = sections["rfid"].map(depth_by_rfid)
    roads_path = out_dir / f"{stem}_road_islands.parquet"
    sections.to_parquet(roads_path)
    table_path = out_dir / f"{stem}_island_distribution.csv"
    island_distribution(roads, island_ids, operational, asset_population).to_csv(table_path, index=False)
    return roads_path, table_path
