"""Societal-access postprocessing: population access to critical functions.

Conceptual model
-----------------
Access is evaluated as **shared island membership** in the disrupted road
network, following the set-theoretic framework of Paper 4:

- ``V``  — the complete set of graph nodes (entities).
- ``G = (V, E)``  — the infrastructure graph before disruption.
- After disruption, unavailable nodes/edges are removed, yielding a
  disrupted subgraph ``G' = (V', E')``.
- Connected components of ``G'`` form disjoint **islands** ``I_1, …, I_k``.
- ``D ⊆ V`` — the set of destination/service nodes (hospitals, fire
  stations, …).
- ``O ⊆ V`` — the set of origin entities (population zones, grid cells).

Access of origin ``o ∈ O`` to function ``f``:

    ``∃ d ∈ D_f : island_id[o] == island_id[d]``

where ``D_f ⊆ D`` is the subset of destinations providing function ``f``.
Functions listed in ``SERVICE_AREA_FUNCTION_PROVIDER_TYPES`` (electricity)
instead use Voronoi service areas built from provider centroids, because
shared-island membership is too coarse a proxy for who a substation actually
serves; see ``_build_service_area_population_maps`` and
``_apply_service_area_societal_scalars``.

Production use
---------------
This module is called exactly once per experiment, after
``simulate_asset_damage_recovery_access_breakdown``'s timestep loop
completes: ``simulation.py`` hands the per-timestep ``detailed_results``
(``island_id`` + ``operational`` arrays) to
``postprocess_societal_access_results``, which returns ``summary_results``
merged with ``societal_*`` fields. Everything else in this module —
allocation caching, Voronoi service-area maps, scalar caches — exists to
make that one call fast and repeatable across EMA experiments. 
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.timing_profiler import NULL_PROFILER


# ---------------------------------------------------------------------------
# Service node taxonomy
# ---------------------------------------------------------------------------

#: Default mapping: node *type* string → *function category* label.
SERVICE_NODE_TAXONOMY: Dict[str, str] = {
    # Healthcare infrastructure
    "hospital": "health",
    "clinic": "health",
    "huisartsenpraktijk": "health",
    "apotheek": "health",
    # Emergency response
    "fire_station": "emergency_response",
    # Education infrastructure
    "school": "education",
    # Electricity infrastructure
    "msls": "electricity",
}

#: Default demographic columns in the CBS population grid.
POPULATION_GROUP_COLUMNS: Dict[str, str] = {
    "total": "aantal_inwoners",
    "elderly": "aantal_inwoners_65_jaar_en_ouder",
    "children": "aantal_inwoners_0_tot_15_jaar",
}

ALLOCATION_ALGORITHM_VERSION: str = "1.0.0"
SERVICE_AREA_FUNCTION_PROVIDER_TYPES: Dict[str, FrozenSet[str]] = {
    "electricity": frozenset({"msls"}),
}
_FUNCTION_CATEGORY_EQUIVALENTS: Dict[str, FrozenSet[str]] = {
    "health": frozenset({"health", "hospital"}),
    "hospital": frozenset({"health", "hospital"}),
}
_REALIZED_STATE_CACHE_KEY_VERSION = "2.1.0"


class SharedRealizedStateCacheError(RuntimeError):
    """Raised when shared realized-state cache operations fail in fail-hard mode."""

def _expand_function_category_equivalents(function_name: str) -> FrozenSet[str]:
    """Return equivalent function-category labels for compatibility outputs."""
    return _FUNCTION_CATEGORY_EQUIVALENTS.get(function_name, frozenset({function_name}))


def _stable_value_token(value: Any) -> str:
    """Stable token for heterogeneous hash-key values."""
    return f"{type(value).__name__}:{value!r}"


def _canonical_operational_signature(
    operational_asset_ids_by_function: Dict[str, Set[Any]],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Canonical function→provider signature independent of insertion order."""
    return tuple(
        (
            str(func),
            tuple(sorted(_stable_value_token(provider_id) for provider_id in ids)),
        )
        for func, ids in sorted(operational_asset_ids_by_function.items(), key=lambda kv: str(kv[0]))
    )


def _build_label_invariant_island_profiles(
    frozen_island_function_map: Dict[int, FrozenSet[str]],
    island_pop: pd.DataFrame,
    available_group_cols: Dict[str, str],
) -> Tuple[Tuple[Tuple[str, ...], Tuple[float, ...]], ...]:
    """Canonical island profiles that are invariant to island ID renumbering."""
    if island_pop is None or island_pop.empty or not available_group_cols:
        return tuple()

    sorted_groups = sorted(available_group_cols.items(), key=lambda kv: str(kv[0]))
    profiles: List[Tuple[Tuple[str, ...], Tuple[float, ...]]] = []
    for island_id in island_pop.index:
        try:
            island_int = int(island_id)
        except Exception:
            continue
        funcs = tuple(sorted(str(f) for f in frozen_island_function_map.get(island_int, frozenset())))
        pop_vector = tuple(float(island_pop.loc[island_id, col]) for _, col in sorted_groups)
        profiles.append((funcs, pop_vector))
    return tuple(sorted(profiles))


def _build_realized_state_cache_key(
    *,
    frozen_island_function_map: Dict[int, FrozenSet[str]],
    operational_asset_ids_by_function: Dict[str, Set[Any]],
    island_pop: Optional[pd.DataFrame],
    total_pop: Optional[Dict[str, float]],
    available_group_cols: Optional[Dict[str, str]],
    all_functions: List[str],
    pop_group_columns: Dict[str, str],
    reference_group: str,
    service_area_function_provider_types: Dict[str, FrozenSet[str]],
    allocation_df: Optional[pd.DataFrame],
    service_area_population_maps_digest: str = "",
) -> str:
    """Build a label-invariant realized-state cache key."""
    if island_pop is None or total_pop is None or available_group_cols is None:
        return ""

    allocation_attrs = allocation_df.attrs if allocation_df is not None else {}
    key_dict: Dict[str, Any] = {
        "version": _REALIZED_STATE_CACHE_KEY_VERSION,
        "allocation_algorithm_version": str(
            allocation_attrs.get("allocation_algorithm_version", ALLOCATION_ALGORITHM_VERSION)
        ),
        "population_grid_hash": str(allocation_attrs.get("population_grid_hash")),
        "cell_id_hash": str(allocation_attrs.get("cell_id_hash")),
        "nearest_max_distance": float(allocation_attrs.get("nearest_max_distance", 200.0)),
        "all_functions": tuple(sorted(str(f) for f in all_functions)),
        "pop_group_labels": tuple(sorted(str(label) for label in pop_group_columns.keys())),
        "reference_group": str(reference_group),
        "service_area_function_provider_types": _canonicalize_service_area_provider_types(
            service_area_function_provider_types
        ),
        "service_area_population_maps_digest": str(service_area_population_maps_digest),
        "operational_signature": _canonical_operational_signature(
            operational_asset_ids_by_function
        ),
        "total_population": tuple(
            (str(label), float(total_pop.get(label, 0.0)))
            for label in sorted(total_pop.keys(), key=str)
        ),
        "island_profiles": _build_label_invariant_island_profiles(
            frozen_island_function_map=frozen_island_function_map,
            island_pop=island_pop,
            available_group_cols=available_group_cols,
        ),
    }
    key_str = json.dumps(key_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_str.encode()).hexdigest()


def _geometry_hash(gdf: gpd.GeoDataFrame) -> str:
    """Deterministic hash of a GeoDataFrame geometry column."""
    digest = hashlib.sha256()
    digest.update(str(gdf.crs).encode())
    for geom in gdf.geometry:
        digest.update(geom.wkb if geom is not None else b"null")
    return digest.hexdigest()[:16]


def _column_hash(df: pd.DataFrame, column: str) -> str:
    """Deterministic hash of a cache-relevant identifier column."""
    digest = hashlib.sha256()
    for value in df[column]:
        digest.update(f"{type(value).__name__}:{value!r}\0".encode())
    return digest.hexdigest()[:16]


def build_allocation_cache_key(
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    islands_gdf: gpd.GeoDataFrame,
    island_id_column: str = "island_id",
    nearest_max_distance: float = 200.0,
    road_state_key: str = "default",
    algorithm_version: str = ALLOCATION_ALGORITHM_VERSION,
    extra: Optional[Dict[str, Any]] = None,
    pop_grid_geom_hash: Optional[str] = None,
    pop_cell_id_hash: Optional[str] = None,
    islands_geom_hash: Optional[str] = None,
    islands_id_hash: Optional[str] = None,
) -> str:
    """Build a deterministic cache key for the geometry-only allocation.

    The key covers all deterministic inputs that affect the road partition or
    grid geometry.

    Optional pre-computed hash strings (``pop_grid_geom_hash``,
    ``pop_cell_id_hash``, ``islands_geom_hash``, ``islands_id_hash``) may be
    supplied to avoid redundant hash computation inside tight loops.  When
    ``None``, the corresponding hash is computed internally.
    """
    key_dict: Dict[str, Any] = {
        "pop_grid_hash": pop_grid_geom_hash if pop_grid_geom_hash is not None else _geometry_hash(pop_grid_gdf),
        "cell_id_column": cell_id_column,
        "cell_id_hash": pop_cell_id_hash if pop_cell_id_hash is not None else _column_hash(pop_grid_gdf, cell_id_column),
        "islands_hash": islands_geom_hash if islands_geom_hash is not None else _geometry_hash(islands_gdf),
        "island_id_column": island_id_column,
        "island_id_hash": islands_id_hash if islands_id_hash is not None else _column_hash(islands_gdf, island_id_column),
        "nearest_max_distance": nearest_max_distance,
        "road_state_key": str(road_state_key),
        "algorithm_version": algorithm_version,
    }
    if extra:
        key_dict.update(extra)
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:24]


def _validate_allocation_df(allocation_df: pd.DataFrame, cell_id_column: str) -> None:
    """Validate that allocation_df satisfies the allocation invariants."""
    required = {cell_id_column, "island_id", "allocation_fraction", "allocation_method"}
    missing = required - set(allocation_df.columns)
    if missing:
        raise ValueError(f"Allocation DataFrame missing columns: {missing}")

    # Every fraction in [0, 1]
    if not ((allocation_df["allocation_fraction"] >= 0) &
            (allocation_df["allocation_fraction"] <= 1)).all():
        raise ValueError("allocation_fraction values must be in [0, 1]")

    # Fractions sum to 1 per cell
    sums = allocation_df.groupby(cell_id_column)["allocation_fraction"].sum()
    if not np.allclose(sums.values, 1.0, atol=1e-6):
        bad = sums[~np.isclose(sums, 1.0, atol=1e-6)]
        raise ValueError(
            f"allocation_fraction does not sum to 1 for {len(bad)} cell(s). "
            f"First few: {bad.head().to_dict()}"
        )

    # Every cell has at least one row
    if len(allocation_df) == 0:
        raise ValueError("Allocation DataFrame is empty.")


def build_origin_island_allocations(
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    islands_gdf: gpd.GeoDataFrame,
    island_id_column: str = "island_id",
    nearest_max_distance: float = 200.0,
    road_state_key: str = "default",
    pop_grid_geom_hash: Optional[str] = None,
    pop_cell_id_hash: Optional[str] = None,
    islands_geom_hash: Optional[str] = None,
    islands_id_hash: Optional[str] = None,
) -> pd.DataFrame:
    """Build geometry-only cell→island allocation fractions.

    For each population grid cell:

    1. Intersect it with all dissolved island geometries.
    2. Sum positive overlap area by island.
    3. If overlaps exist, normalise across intersected islands (fractions sum to 1).
    4. If no overlap, assign to nearest island within *nearest_max_distance*
       with fraction 1.0 and method ``nearest_island``.
    5. If no island within distance, assign ``island_id = -1`` with fraction 1.0
       and method ``unassigned``.

    The output contains **no population or demographic values** — only geometry-
    derived fractions.  Population is applied separately via
    :func:`apply_population_to_allocations`.

    Parameters
    ----------
    pop_grid_gdf:
        Population grid GeoDataFrame.  Must have *cell_id_column* and geometry.
    cell_id_column:
        Stable identifier column for grid cells.  Must be unique per cell.
    islands_gdf:
        Dissolved road-island polygons with *island_id_column* and geometry.
    island_id_column:
        Column in *islands_gdf* holding island identifiers.
    nearest_max_distance:
        Maximum distance (CRS units, typically metres) for nearest-island
        fallback assignment.
    road_state_key:
        Island cache key identifying the road disruption and adaptation state
        that produced these islands.
    pop_grid_geom_hash, pop_cell_id_hash, islands_geom_hash, islands_id_hash:
        Pre-computed hash strings.  When supplied, the corresponding
        ``_geometry_hash`` / ``_column_hash`` calls are skipped.

    Returns
    -------
    pd.DataFrame with columns:
        ``cell_id_column``, ``island_id``, ``allocation_fraction``,
        ``allocation_method``, ``road_state_key``,
        ``allocation_algorithm_version``.
    """
    road_state_key = str(road_state_key)
    target_crs = islands_gdf.crs or "EPSG:28992"
    pop = pop_grid_gdf[[cell_id_column, "geometry"]].copy().to_crs(target_crs)
    islands = islands_gdf[[island_id_column, "geometry"]].copy().to_crs(target_crs)

    islands_dissolved = islands.dissolve(by=island_id_column).reset_index()

    rows: List[Dict[str, Any]] = []

    # --- Vectorized intersection-based allocation (Problem 2) ---
    pop_renamed = pop.rename(columns={cell_id_column: "__cell_id"})
    islands_renamed = islands_dissolved.rename(columns={island_id_column: "__island_id"})

    overlaps = gpd.overlay(
        pop_renamed,
        islands_renamed,
        how="intersection",
        keep_geom_type=False,
    )
    overlaps["__area"] = overlaps.geometry.area
    overlaps = overlaps[overlaps["__area"] > 0]

    if not overlaps.empty:
        area_by_cell_island = overlaps.groupby(["__cell_id", "__island_id"])["__area"].sum()
        total_by_cell = area_by_cell_island.groupby(level=0).transform("sum")
        fractions = (area_by_cell_island / total_by_cell).reset_index()
        fractions.columns = ["__cell_id", "__island_id", "allocation_fraction"]

        for _, frow in fractions.iterrows():
            rows.append({
                cell_id_column: frow["__cell_id"],
                "island_id": int(frow["__island_id"]),
                "allocation_fraction": float(frow["allocation_fraction"]),
                "allocation_method": "intersection",
                "road_state_key": road_state_key,
                "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
            })

    # Identify cells with no intersection
    intersected_cell_ids: set = set(overlaps["__cell_id"].unique()) if not overlaps.empty else set()
    no_overlap_pop = pop[~pop[cell_id_column].isin(intersected_cell_ids)].copy()

    if not no_overlap_pop.empty:
        # Step 4: vectorized nearest-island fallback
        no_overlap_centroids = no_overlap_pop.copy()
        no_overlap_centroids["geometry"] = no_overlap_centroids.geometry.centroid
        no_overlap_centroids_gdf = gpd.GeoDataFrame(
            no_overlap_centroids[[cell_id_column, "geometry"]],
            geometry="geometry",
            crs=target_crs,
        )

        nearest_join = gpd.sjoin_nearest(
            no_overlap_centroids_gdf,
            islands_renamed[["__island_id", "geometry"]],
            how="left",
            max_distance=nearest_max_distance,
            distance_col="__dist",
        )

        # sjoin_nearest may produce duplicates; keep the first (closest)
        nearest_join = nearest_join.drop_duplicates(subset=[cell_id_column], keep="first")

        for _, nrow in nearest_join.iterrows():
            cell_id = nrow[cell_id_column]
            island_id_val = nrow.get("__island_id")
            if pd.notna(island_id_val):
                rows.append({
                    cell_id_column: cell_id,
                    "island_id": int(island_id_val),
                    "allocation_fraction": 1.0,
                    "allocation_method": "nearest_island",
                    "road_state_key": road_state_key,
                    "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
                })
            else:
                # Step 5: unassigned
                rows.append({
                    cell_id_column: cell_id,
                    "island_id": -1,
                    "allocation_fraction": 1.0,
                    "allocation_method": "unassigned",
                    "road_state_key": road_state_key,
                    "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
                })

    allocation_df = pd.DataFrame(rows)

    # Resolve hashes — use pre-computed values when available
    _pop_geom_hash = pop_grid_geom_hash if pop_grid_geom_hash is not None else _geometry_hash(pop_grid_gdf)
    _pop_id_hash = pop_cell_id_hash if pop_cell_id_hash is not None else _column_hash(pop_grid_gdf, cell_id_column)
    _isl_geom_hash = islands_geom_hash if islands_geom_hash is not None else _geometry_hash(islands_gdf)
    _isl_id_hash = islands_id_hash if islands_id_hash is not None else _column_hash(islands_gdf, island_id_column)

    allocation_df.attrs.update({
        "population_grid_hash": _pop_geom_hash,
        "cell_id_column": cell_id_column,
        "cell_id_hash": _pop_id_hash,
        "islands_hash": _isl_geom_hash,
        "island_id_column": island_id_column,
        "island_id_hash": _isl_id_hash,
        "nearest_max_distance": nearest_max_distance,
        "road_state_key": road_state_key,
        "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
    })
    _validate_allocation_df(allocation_df, cell_id_column)
    return allocation_df


def apply_population_to_allocations(
    allocation_df: pd.DataFrame,
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    pop_group_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Join population attributes to allocations and apply allocation fractions.

    Parameters
    ----------
    allocation_df:
        Output of :func:`build_origin_island_allocations`.  Not mutated.
    pop_grid_gdf:
        Population grid GeoDataFrame with *cell_id_column* and demographic
        columns.
    cell_id_column:
        Stable identifier column shared between *allocation_df* and
        *pop_grid_gdf*.
    pop_group_columns:
        ``{display_label: dataframe_column}`` demographic mapping.
        Defaults to :data:`POPULATION_GROUP_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        One row per (cell, island) allocation; original allocation columns
        retained; additional columns ``{display_label}_weighted`` added
        (= population_count × allocation_fraction).
    """
    if pop_group_columns is None:
        pop_group_columns = POPULATION_GROUP_COLUMNS

    pop_cols = list(pop_group_columns.values())
    available_cols = [c for c in pop_cols if c in pop_grid_gdf.columns]

    pop_attrs = pop_grid_gdf[[cell_id_column] + available_cols].copy()
    # Sanitise CBS suppressed values
    for col in available_cols:
        pop_attrs[col] = pd.to_numeric(pop_attrs[col], errors="coerce").fillna(0)
        pop_attrs[col] = pop_attrs[col].where(pop_attrs[col] >= 0, 0)

    # Validate many-to-one: each cell_id appears once in pop_attrs
    if pop_attrs[cell_id_column].duplicated().any():
        raise ValueError(
            f"Population grid has duplicate values in '{cell_id_column}'. "
            "The join must be many-to-one (allocation rows → one population row)."
        )

    merged = allocation_df.merge(pop_attrs, on=cell_id_column, how="left")

    for label, col in pop_group_columns.items():
        if col in merged.columns:
            merged[f"{label}_weighted"] = (
                merged[col].fillna(0) * merged["allocation_fraction"]
            )

    return merged


def _build_population_value_arrays(
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    pop_group_columns: Dict[str, str],
) -> Tuple[pd.Index, Dict[str, np.ndarray]]:
    """Sanitise population demographic columns once, as plain numpy arrays.

    Work item A: population values are invariant across road/allocation
    states -- only the cell → island assignment changes per state.  This
    extracts and sanitises them exactly once per
    :func:`postprocess_societal_access_results` call (mirroring the
    CBS-suppressed-value coercion and negative-value clipping performed
    inside :func:`apply_population_to_allocations`) instead of re-doing it
    for every distinct state.  Returned arrays are aligned to a stable
    ``cell_id`` position ordering (``cell_id_index``), which
    :func:`_aggregate_population_by_island` uses to align them against each
    state's allocation rows.

    Returns
    -------
    (cell_id_index, value_arrays)
        ``cell_id_index`` is a :class:`pandas.Index` of cell IDs in a fixed
        position order; ``value_arrays`` maps ``{label: numpy_array}`` for
        every demographic label whose column is present in *pop_grid_gdf*
        (labels with a missing column are simply absent, mirroring the
        ``if column_name not in ... .columns: continue`` guard elsewhere).

    Raises
    ------
    ValueError
        If *cell_id_column* has duplicate values (same invariant enforced by
        :func:`apply_population_to_allocations`).
    """
    pop_cols = list(pop_group_columns.values())
    available_cols = [c for c in pop_cols if c in pop_grid_gdf.columns]

    pop_attrs = pop_grid_gdf[[cell_id_column] + available_cols].copy()
    for col in available_cols:
        pop_attrs[col] = pd.to_numeric(pop_attrs[col], errors="coerce").fillna(0)
        pop_attrs[col] = pop_attrs[col].where(pop_attrs[col] >= 0, 0)

    if pop_attrs[cell_id_column].duplicated().any():
        raise ValueError(
            f"Population grid has duplicate values in '{cell_id_column}'. "
            "The join must be many-to-one (allocation rows → one population row)."
        )

    cell_id_index = pd.Index(pop_attrs[cell_id_column].to_numpy())
    value_arrays = {
        label: pop_attrs[col].to_numpy(dtype=float)
        for label, col in pop_group_columns.items()
        if col in available_cols
    }
    return cell_id_index, value_arrays


def _aggregate_population_by_island(
    allocation_df: pd.DataFrame,
    cell_id_column: str,
    cell_id_index: pd.Index,
    value_arrays: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, str]]:
    """Aggregate one allocation state's per-cell population to per-island sums.

    Work item A: replaces the previous ``apply_population_to_allocations``
    (full merge) + ``groupby("island_id").sum()`` pattern used on every
    distinct road/allocation state, which re-joined and re-sanitised the
    *entire* population grid on every cache miss.  Given the once-per-call
    sanitised arrays from :func:`_build_population_value_arrays`, this only
    positions those values against the current state's allocation rows
    (``cell_id_index.get_indexer``, O(n)) and aggregates by island via dense
    integer codes + ``np.bincount`` (O(n), no hash-join, no per-state
    re-sanitisation).

    Numerically equivalent to the previous merge/groupby path -- but *not*
    bit-identical to the ULP.  ``np.bincount``'s plain sequential
    accumulation differs from pandas' groupby-sum kernel (which uses a
    compensated summation internally) by ~1e-9 absolute on representative
    data.  This has been confirmed negligible: population magnitudes here
    are always non-negative small integers/floats, and downstream values are
    rounded to 2 decimal places, so this reduction-order difference cannot
    change any emitted metric.  See the equivalence tests in
    ``tests/test_societal_access.py``.

    Returns
    -------
    (island_pop, total_pop, available_group_cols)
        Same shapes as the corresponding entries previously stored in
        ``_pop_alloc_cache`` (Work item C dropped the redundant full
        per-cell ``pop_alloc`` frame from that cache once nothing downstream
        needed it beyond these three).
    """
    available_group_cols = {label: f"{label}_weighted" for label in value_arrays}
    if not available_group_cols or allocation_df.empty:
        return pd.DataFrame(), {}, available_group_cols

    positions = cell_id_index.get_indexer(allocation_df[cell_id_column].to_numpy())
    valid_mask = positions >= 0
    safe_positions = np.where(valid_mask, positions, 0)
    fractions = allocation_df["allocation_fraction"].to_numpy(dtype=float)
    island_codes, island_uniques = pd.factorize(allocation_df["island_id"].to_numpy(), sort=True)
    n_islands = len(island_uniques)

    island_pop_data: Dict[str, np.ndarray] = {}
    total_pop: Dict[str, float] = {}
    for label, col in available_group_cols.items():
        raw_vals = value_arrays[label][safe_positions]
        # get_indexer returns -1 for an unmatched cell_id (should not occur
        # in practice -- allocation_df is always derived from this same
        # population grid); treat any such gap as population 0, mirroring
        # `merged[col].fillna(0)` in the original merge-based path.
        pop_vals = np.where(valid_mask, raw_vals, 0.0)
        weighted = pop_vals * fractions
        island_sums = np.bincount(island_codes, weights=weighted, minlength=n_islands)
        island_pop_data[col] = island_sums
        total_pop[label] = float(island_sums.sum())

    island_pop = pd.DataFrame(island_pop_data, index=pd.Index(island_uniques, name="island_id"))
    return island_pop, total_pop, available_group_cols


def clip_population_to_service_area(
    pop_grid_gdf: gpd.GeoDataFrame,
    gdf_assets: gpd.GeoDataFrame,
    buffer_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """Clip a population grid to a buffered hull of asset/provider locations.

    Work item B: a lightweight, opt-in guard against feeding
    :func:`postprocess_societal_access_results` a population grid far larger
    than the area assets can actually serve.  This mirrors the pre-clip the
    sample notebook already performs manually (loading the population layer
    with ``bbox=tuple(voronoi_gdf.total_bounds)``), packaged as a reusable
    helper so other callers (e.g. a country-scale deployment loading the
    full national grid) can pre-clip *before* the grid ever reaches
    ``postprocess_societal_access_results`` -- every per-state population
    aggregation (Work item A) and the one-off service-area population map
    build both scale with population-grid size, so clipping upstream is the
    cheapest possible win.

    This is *not* a mandatory pipeline stage: it changes nothing unless a
    caller opts in by calling it, and :func:`postprocess_societal_access_results`
    only ever *warns* (never raises) if it detects an unclipped grid --
    some callers may have legitimate reasons for a larger grid than the
    convex hull of current asset locations (e.g. anticipated future asset
    placement, or a shared grid reused across multiple studies).

    Parameters
    ----------
    pop_grid_gdf:
        Population grid GeoDataFrame to clip.
    gdf_assets:
        Asset/provider GeoDataFrame; the convex hull of its geometries
        (buffered by *buffer_m*) defines the service-area boundary.
    buffer_m:
        Buffer distance in *pop_grid_gdf*'s CRS units (typically metres).

    Returns
    -------
    gpd.GeoDataFrame
        Subset of *pop_grid_gdf* intersecting the buffered hull, with the
        index reset.  Returns *pop_grid_gdf* unchanged if either input is
        empty.
    """
    if gdf_assets.empty or pop_grid_gdf.empty:
        return pop_grid_gdf

    assets_in_pop_crs = (
        gdf_assets.to_crs(pop_grid_gdf.crs) if gdf_assets.crs != pop_grid_gdf.crs else gdf_assets
    )
    _assets_geom = assets_in_pop_crs.geometry
    _assets_union = _assets_geom.union_all() if hasattr(_assets_geom, "union_all") else _assets_geom.unary_union
    hull = _assets_union.convex_hull.buffer(buffer_m)
    hull_gdf = gpd.GeoDataFrame(geometry=[hull], crs=pop_grid_gdf.crs)
    clipped = gpd.clip(pop_grid_gdf, hull_gdf)
    return clipped.reset_index(drop=True)


def _service_area_hull_area(gdf_assets: gpd.GeoDataFrame, crs, buffer_m: float) -> Optional[float]:
    """Area of the buffered convex hull of *gdf_assets* in *crs*, or ``None``."""
    if gdf_assets.empty:
        return None
    try:
        assets_in_crs = gdf_assets.to_crs(crs) if gdf_assets.crs != crs else gdf_assets
        _geom = assets_in_crs.geometry
        _union = _geom.union_all() if hasattr(_geom, "union_all") else _geom.unary_union
        hull = _union.convex_hull.buffer(buffer_m)
        area = float(hull.area)
        return area if area > 0 else None
    except Exception:
        return None


def get_or_build_allocation(
    allocation_cache: Dict[str, pd.DataFrame],
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    islands_gdf: gpd.GeoDataFrame,
    island_id_column: str = "island_id",
    nearest_max_distance: float = 200.0,
    road_state_key: str = "default",
    pop_grid_geom_hash: Optional[str] = None,
    pop_cell_id_hash: Optional[str] = None,
    islands_geom_hash: Optional[str] = None,
    islands_id_hash: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, bool]:
    """Return a cached geometry allocation or build and cache a new one.

    Parameters
    ----------
    allocation_cache:
        In-memory dict keyed by allocation cache key → allocation DataFrame.
    pop_grid_gdf, cell_id_column, islands_gdf, island_id_column,
    nearest_max_distance, road_state_key:
        Forwarded to :func:`build_origin_island_allocations`.
    pop_grid_geom_hash, pop_cell_id_hash, islands_geom_hash, islands_id_hash:
        Pre-computed hash strings forwarded to avoid redundant computation.

    Returns
    -------
    (allocation_df, cache_key, cache_was_updated)
    """
    cache_key = build_allocation_cache_key(
        pop_grid_gdf=pop_grid_gdf,
        cell_id_column=cell_id_column,
        islands_gdf=islands_gdf,
        island_id_column=island_id_column,
        nearest_max_distance=nearest_max_distance,
        road_state_key=road_state_key,
        pop_grid_geom_hash=pop_grid_geom_hash,
        pop_cell_id_hash=pop_cell_id_hash,
        islands_geom_hash=islands_geom_hash,
        islands_id_hash=islands_id_hash,
    )

    if cache_key in allocation_cache:
        return allocation_cache[cache_key], cache_key, False

    allocation_df = build_origin_island_allocations(
        pop_grid_gdf=pop_grid_gdf,
        cell_id_column=cell_id_column,
        islands_gdf=islands_gdf,
        island_id_column=island_id_column,
        nearest_max_distance=nearest_max_distance,
        road_state_key=road_state_key,
        pop_grid_geom_hash=pop_grid_geom_hash,
        pop_cell_id_hash=pop_cell_id_hash,
        islands_geom_hash=islands_geom_hash,
        islands_id_hash=islands_id_hash,
    )
    allocation_cache[cache_key] = allocation_df
    return allocation_df, cache_key, True


def _find_allocation_cache_entry(
    allocation_cache: Dict[str, pd.DataFrame],
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    road_state_key: str,
    nearest_max_distance: float,
    pop_grid_geom_hash: Optional[str] = None,
    pop_cell_id_hash: Optional[str] = None,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """Return the unique cache entry matching the expected allocation metadata.

    Parameters
    ----------
    pop_grid_geom_hash, pop_cell_id_hash:
        Pre-computed hash strings for the population grid.  When supplied,
        the corresponding ``_geometry_hash`` / ``_column_hash`` calls are
        skipped (Problem 5 optimisation).
    """
    expected_metadata = {
        "population_grid_hash": pop_grid_geom_hash if pop_grid_geom_hash is not None else _geometry_hash(pop_grid_gdf),
        "cell_id_column": cell_id_column,
        "cell_id_hash": pop_cell_id_hash if pop_cell_id_hash is not None else _column_hash(pop_grid_gdf, cell_id_column),
        "nearest_max_distance": nearest_max_distance,
        "road_state_key": road_state_key,
        "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
    }
    matching = [
        (cache_key, df)
        for cache_key, df in allocation_cache.items()
        if all(df.attrs.get(k) == v for k, v in expected_metadata.items())
    ]
    if len(matching) != 1:
        return None, None
    return matching[0]


# ---------------------------------------------------------------------------
# Module-level memo cache for `_build_service_area_population_maps`.
#
# The function is a pure function of (asset centroids, population grid
# geometry + group columns, provider-type spec, asset-type column), yet
# ``gdf_assets`` and the population grid are EMA ``Constant``s across an
# experiment set — so recomputing it once per experiment (rather than once
# per process) wastes the entire Voronoi/nearest-provider construction and
# accumulation cost.  Mirrors the ``_VORONOI_CACHE`` pattern in
# ``impacts.py``: cache-key-then-check-then-copy-on-hit-then-store-copy.
# ---------------------------------------------------------------------------
_SERVICE_AREA_POP_MAP_CACHE: Dict[Any, Dict[str, Dict[str, Dict[Any, float]]]] = {}
_SERVICE_AREA_POP_MAP_CACHE_VERSION = "1.0.0"


def _service_area_asset_state_digest(working_assets: gpd.GeoDataFrame) -> str:
    """Deterministic digest of service-area-relevant asset state."""
    digest = hashlib.sha256()
    digest.update(str(working_assets.crs).encode())
    for asset_id in sorted(working_assets.index, key=_stable_value_token):
        row = working_assets.loc[asset_id]
        digest.update(_stable_value_token(asset_id).encode())
        digest.update(_stable_value_token(row.get("type")).encode())
        geom = row.get("geometry")
        digest.update(geom.wkb if geom is not None else b"null")
    return digest.hexdigest()[:24]


def _service_area_population_values_digest(
    pop_grid_gdf: gpd.GeoDataFrame,
    pop_group_columns: Dict[str, str],
) -> str:
    """Deterministic digest of demographic values used in service-area maps."""
    digest = hashlib.sha256()
    digest.update(str(pop_grid_gdf.crs).encode())
    for label, column_name in sorted(pop_group_columns.items(), key=lambda kv: str(kv[0])):
        digest.update(str(label).encode())
        digest.update(str(column_name).encode())
        if column_name not in pop_grid_gdf.columns:
            digest.update(b"__missing__")
            continue
        col = pd.to_numeric(pop_grid_gdf[column_name], errors="coerce").fillna(0)
        col = col.where(col >= 0, 0)
        for value in col.to_numpy(dtype=float):
            digest.update(repr(float(value)).encode())
            digest.update(b"\0")
    return digest.hexdigest()[:24]


def _service_area_population_maps_digest(
    service_area_population_maps: Dict[str, Dict[str, Dict[Any, float]]],
) -> str:
    """Deterministic digest of provider→population mappings used for overrides."""
    canonical: List[Any] = []
    for function_name in sorted(service_area_population_maps.keys(), key=str):
        group_maps = service_area_population_maps[function_name]
        group_entries: List[Any] = []
        for label in sorted(group_maps.keys(), key=str):
            provider_population = group_maps.get(label, {})
            provider_entries = tuple(
                (
                    _stable_value_token(provider_id),
                    float(provider_population[provider_id]),
                )
                for provider_id in sorted(provider_population.keys(), key=_stable_value_token)
            )
            group_entries.append((str(label), provider_entries))
        canonical.append((str(function_name), tuple(group_entries)))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _canonicalize_service_area_provider_types(
    spec: Mapping[str, Iterable[str]],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Return a hashable, order-independent form of a provider-types spec."""
    return tuple(
        (str(function_name), tuple(sorted(str(provider_type) for provider_type in provider_types)))
        for function_name, provider_types in sorted(spec.items(), key=lambda kv: str(kv[0]))
    )


def _service_area_pop_map_cache_key(
    asset_state_digest: str,
    pop_grid_gdf: gpd.GeoDataFrame,
    pop_group_columns: Dict[str, str],
    service_area_function_provider_types: Dict[str, FrozenSet[str]],
    asset_type_column: str,
) -> Tuple[Any, ...]:
    """Build the memo key for :func:`_build_service_area_population_maps`."""
    return (
        _SERVICE_AREA_POP_MAP_CACHE_VERSION,
        asset_state_digest,
        _geometry_hash(pop_grid_gdf),
        _service_area_population_values_digest(pop_grid_gdf, pop_group_columns),
        tuple(sorted(pop_group_columns.items())),
        _canonicalize_service_area_provider_types(service_area_function_provider_types),
        asset_type_column,
    )


def _copy_service_area_population_maps(
    function_maps: Dict[str, Dict[str, Dict[Any, float]]],
) -> Dict[str, Dict[str, Dict[Any, float]]]:
    """Return an independent copy so callers can never mutate the cached value."""
    return {
        function_alias: {label: dict(pop_map) for label, pop_map in group_maps.items()}
        for function_alias, group_maps in function_maps.items()
    }


def _build_service_area_population_maps(
    gdf_assets: gpd.GeoDataFrame,
    pop_grid_gdf: gpd.GeoDataFrame,
    pop_group_columns: Dict[str, str],
    asset_type_column: str = "type",
    service_area_function_provider_types: Optional[Dict[str, FrozenSet[str]]] = None,
) -> Dict[str, Dict[str, Dict[Any, float]]]:
    """Pre-compute provider→population service-area assignments for special functions.

    Memoised at module scope (``_SERVICE_AREA_POP_MAP_CACHE``): this is a pure
    function of (asset centroids, population grid geometry + group columns,
    provider-type spec, asset-type column), and both ``gdf_assets`` and the
    population grid are EMA ``Constant``s across an experiment set.  The
    returned value is always a fresh copy — a cache hit never hands out the
    same mutable dict twice.
    """
    from src.caching import get_asset_centroid_hash
    from src.impacts import create_voronoi_for_asset_type
    from src.utils import build_voronoi_service_area_map

    if gdf_assets.empty or pop_grid_gdf.empty:
        return {}

    working_assets = gdf_assets[[asset_type_column, "geometry"]].copy()
    if asset_type_column != "type":
        working_assets = working_assets.rename(columns={asset_type_column: "type"})
    working_assets = gpd.GeoDataFrame(
        working_assets,
        geometry="geometry",
        crs=gdf_assets.crs,
    )
    asset_cache_key = get_asset_centroid_hash(working_assets[["geometry"]].copy())
    asset_state_digest = _service_area_asset_state_digest(working_assets)

    if service_area_function_provider_types is None:
        service_area_function_provider_types = SERVICE_AREA_FUNCTION_PROVIDER_TYPES

    cache_key = _service_area_pop_map_cache_key(
        asset_state_digest,
        pop_grid_gdf,
        pop_group_columns,
        service_area_function_provider_types,
        asset_type_column,
    )
    cached_maps = _SERVICE_AREA_POP_MAP_CACHE.get(cache_key)
    if cached_maps is not None:
        return _copy_service_area_population_maps(cached_maps)

    available_cols = [col for col in pop_group_columns.values() if col in pop_grid_gdf.columns]
    pop_values = pop_grid_gdf[available_cols].copy()
    for col in available_cols:
        pop_values[col] = pd.to_numeric(pop_values[col], errors="coerce").fillna(0)
        pop_values[col] = pop_values[col].where(pop_values[col] >= 0, 0)
    pop_assets = gpd.GeoDataFrame(pop_values, geometry=pop_grid_gdf.geometry, crs=pop_grid_gdf.crs)
    if (
        pop_assets.crs is not None
        and working_assets.crs is not None
        and pop_assets.crs != working_assets.crs
    ):
        pop_assets = pop_assets.to_crs(working_assets.crs)

    function_maps: Dict[str, Dict[str, Dict[Any, float]]] = {}
    for function_name, provider_types in service_area_function_provider_types.items():
        group_maps: Dict[str, Dict[Any, float]] = {label: {} for label in pop_group_columns}
        has_provider = False

        for provider_type in sorted(provider_types):
            providers = working_assets[working_assets["type"].astype(str) == str(provider_type)]
            if providers.empty:
                continue
            has_provider = True

            if len(providers) == 1:
                provider_map = {providers.index[0]: list(pop_assets.index)}
            else:
                if len(providers) < 4:
                    provider_centroids = providers.geometry.centroid
                    pop_centroids = pop_assets.geometry.centroid

                    provider_ids = providers.index.to_numpy()
                    pop_ids = pop_assets.index.to_numpy()
                    provider_coords = np.column_stack(
                        (
                            provider_centroids.x.to_numpy(dtype=float),
                            provider_centroids.y.to_numpy(dtype=float),
                        )
                    )
                    pop_coords = np.column_stack(
                        (
                            pop_centroids.x.to_numpy(dtype=float),
                            pop_centroids.y.to_numpy(dtype=float),
                        )
                    )

                    sq_dist = (
                        (pop_coords[:, None, :] - provider_coords[None, :, :]) ** 2
                    ).sum(axis=2)
                    nearest_idx = np.argmin(sq_dist, axis=1)
                    assigned_provider_ids = provider_ids[nearest_idx]
                    sort_order = np.argsort(assigned_provider_ids, kind="stable")
                    sorted_provider_ids = assigned_provider_ids[sort_order]
                    sorted_pop_ids = pop_ids[sort_order]
                    unique_provider_ids, start_idx = np.unique(
                        sorted_provider_ids, return_index=True
                    )
                    provider_map = {}
                    for i, provider_id in enumerate(unique_provider_ids):
                        end = start_idx[i + 1] if i + 1 < len(start_idx) else len(sorted_pop_ids)
                        provider_map[provider_id] = sorted_pop_ids[start_idx[i]:end].tolist()
                else:
                    voronoi_gdf = create_voronoi_for_asset_type(
                        working_assets,
                        provider_type,
                        asset_cache_key=asset_cache_key,
                    )
                    provider_map = build_voronoi_service_area_map(
                        voronoi_gdf,
                        pop_assets[["geometry"]].copy(),
                    )

            # --- Problem 2: invert provider_map into a provider-label series
            # aligned with pop_assets.index, then do a single groupby(...).sum()
            # over all available columns at once instead of O(providers × groups)
            # per-provider `.loc[...]` fancy-indexing sums. Providers with no
            # assigned pop cells contribute nothing; pop cells not covered by
            # any provider in *this* provider_map (e.g. clipped Voronoi edge
            # effects) are excluded, matching the original behaviour exactly.
            valid_label_cols = [
                (label, column_name)
                for label, column_name in pop_group_columns.items()
                if column_name in pop_assets.columns
            ]
            if valid_label_cols and provider_map:
                provider_label = pd.Series(np.nan, index=pop_assets.index, dtype=object)
                for provider_id, pop_indices in provider_map.items():
                    provider_label.loc[list(pop_indices)] = provider_id
                assigned_mask = provider_label.notna()
                if assigned_mask.any():
                    cols = [column_name for _, column_name in valid_label_cols]
                    grouped_sums = (
                        pop_assets.loc[assigned_mask, cols]
                        .groupby(provider_label[assigned_mask])
                        .sum()
                    )
                    for label, column_name in valid_label_cols:
                        col_sums = grouped_sums[column_name]
                        for provider_id, value in col_sums.items():
                            group_maps[label][provider_id] = group_maps[label].get(
                                provider_id, 0.0
                            ) + float(value)

        if has_provider:
            for function_alias in _expand_function_category_equivalents(str(function_name)):
                function_maps[function_alias] = group_maps

    _SERVICE_AREA_POP_MAP_CACHE[cache_key] = _copy_service_area_population_maps(function_maps)
    return function_maps


def _apply_service_area_societal_scalars(
    fields: Dict[str, float],
    operational_asset_ids_by_function: Dict[str, Set[Any]],
    service_area_population_maps: Dict[str, Dict[str, Dict[Any, float]]],
    pop_group_columns: Dict[str, str],
    reference_group: str,
    numpy_maps: Optional[Dict[str, Dict[str, Tuple[Any, Any]]]] = None,
    position_maps: Optional[Dict[str, Dict[str, Dict[Any, int]]]] = None,
) -> Dict[str, float]:
    """Override function metrics for service-area-based services such as electricity.

    Parameters
    ----------
    numpy_maps:
        Optional precomputed numpy arrays per function/label.  Each entry is
        ``{func: {label: (ids_array, pops_array)}}``.  When provided, the
        O(N×G) Python generator sums are replaced by O(N) masked numpy sums.
    position_maps:
        Optional precomputed ``{func: {label: {provider_id: position}}}``
        mapping, aligned with *numpy_maps*' ``ids_array``.  When provided,
        the operational-provider membership mask is built via O(len(
        operational_ids)) position lookups instead of rebuilding a list from
        *operational_ids* and sorting it on every call via ``np.isin``.
    """
    if not service_area_population_maps:
        return fields

    derived_totals: Dict[str, float] = {}
    for function_name, group_maps in service_area_population_maps.items():
        operational_ids = operational_asset_ids_by_function.get(function_name, set())
        func_numpy = numpy_maps.get(function_name) if numpy_maps is not None else None
        func_positions = position_maps.get(function_name) if position_maps is not None else None
        for label in pop_group_columns:
            if func_numpy is not None and label in func_numpy:
                ids_arr, pops_arr = func_numpy[label]
                total = float(pops_arr.sum())
                if len(ids_arr) > 0 and operational_ids:
                    label_positions = func_positions.get(label) if func_positions is not None else None
                    if label_positions is not None:
                        mask = np.zeros(len(ids_arr), dtype=bool)
                        for provider_id in operational_ids:
                            position = label_positions.get(provider_id)
                            if position is not None:
                                mask[position] = True
                    else:
                        mask = np.isin(ids_arr, list(operational_ids))
                    with_access = float(pops_arr[mask].sum())
                else:
                    with_access = 0.0
            else:
                provider_population = group_maps.get(label, {})
                total = float(sum(provider_population.values()))
                with_access = float(
                    sum(
                        population
                        for provider_id, population in provider_population.items()
                        if provider_id in operational_ids
                    )
                )
            without_access = total - with_access
            pct = round(100.0 * with_access / total, 2) if total > 0 else float("nan")

            fields[f"societal_access_pct__{function_name}__{label}"] = pct
            fields[f"societal_access_population__{function_name}__{label}"] = with_access
            fields[f"societal_no_access_population__{function_name}__{label}"] = without_access
            derived_totals[label] = total

        ref_pct = fields.get(
            f"societal_access_pct__{function_name}__{reference_group}",
            float("nan"),
        )
        for label in pop_group_columns:
            if label == reference_group:
                continue
            grp_pct = fields.get(f"societal_access_pct__{function_name}__{label}", float("nan"))
            if np.isnan(ref_pct) or np.isnan(grp_pct):
                abs_gap = float("nan")
                rel_gap = float("nan")
            else:
                abs_gap = round(ref_pct - grp_pct, 2)
                rel_gap = round(grp_pct / ref_pct, 4) if ref_pct > 0 else float("nan")
            fields[f"societal_equity_absolute_gap__{function_name}__{label}"] = abs_gap
            fields[f"societal_equity_relative_gap__{function_name}__{label}"] = rel_gap

    for label, total in derived_totals.items():
        total_key = f"societal_total_population__{label}"
        if np.isnan(fields.get(total_key, float("nan"))):
            fields[total_key] = total

    return fields


def postprocess_societal_access_results(
    summary_results: List[Dict[str, Any]],
    detailed_results: List[Dict[str, Any]],
    gdf_assets: gpd.GeoDataFrame,
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    pop_group_columns: Optional[Dict[str, str]] = None,
    taxonomy: Optional[Dict[str, str]] = None,
    asset_type_column: str = "type",
    asset_id_column: Optional[str] = None,
    island_id_column: str = "island_id",
    allocation_cache: Optional[Dict[str, pd.DataFrame]] = None,
    all_functions: Optional[List[str]] = None,
    reference_group: str = "total",
    nearest_max_distance: float = 200.0,
    islands_gdf_cache: Optional[Dict[str, gpd.GeoDataFrame]] = None,
    service_area_function_provider_types: Optional[Dict[str, FrozenSet[str]]] = None,
    fail_on_missing_allocation: bool = False,
    verbose: bool = False,
    profiler: Optional[Any] = NULL_PROFILER,
    pop_grid_area_warn_buffer_m: float = 5000.0,
    pop_grid_area_warn_ratio: float = 25.0,
    shared_realized_state_cache: Optional[Any] = None,
    shared_cache_fail_hard: bool = False,
    cache_telemetry: Optional[Dict[str, int]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Compute societal access metrics per timestep and merge into summary results.

    This function is the realization-aware postprocessor.  It must be called
    **after** the simulation loop completes so that dependency-adjusted
    ``operational`` states are already recorded in *detailed_results*.

    For each timestep:

    1. Read service-provider IDs, island IDs, and dependency-adjusted
       ``operational`` states from *detailed_results*.
    2. Build ``island_id → set(functions supplied by operational providers)``.
    3. Retrieve (or build) the geometry-only allocation for the road state
       recorded in that timestep.
    4. Apply population to allocations.
    5. Compute per-(function, group) access scalars.
    6. Merge flat scalars into the matching summary dict.

    Parameters
    ----------
    summary_results:
        List of per-timestep summary dicts (``results`` list from simulation).
    detailed_results:
        List of per-timestep detailed asset-state dicts (``timestep_results``
        from simulation).  Each must contain ``timestep``, ``map``,
        ``asset_id``, ``operational``, and ``island_id``.  Simulation output
        also supplies ``road_state_key`` for adaptation-aware cache lookup.
    gdf_assets:
        Asset GeoDataFrame — provides ``asset_type_column`` and optionally
        a stable ``asset_id_column``.
    pop_grid_gdf:
        Population grid GeoDataFrame with demographic columns.
    cell_id_column:
        Stable cell identifier column in *pop_grid_gdf*.
    pop_group_columns:
        ``{display_label: column_name}`` demographic mapping.
    taxonomy:
        Node-type → function-category mapping.
    asset_type_column:
        Column in *gdf_assets* holding the node type string.
    asset_id_column:
        Stable asset ID column.  If ``None``, uses positional index.
    island_id_column:
        Column holding island IDs in *islands_gdf*.
    allocation_cache:
        In-memory allocation cache (mutated in-place on cache misses).
    all_functions:
        Explicit list of function categories to always emit (stabilises
        output shape even when all providers of a function fail).
    reference_group:
        Reference group for equity gap computation.
    nearest_max_distance:
        Nearest-island fallback maximum distance for allocation.
    islands_gdf_cache:
        Dict mapping each ``road_state_key`` to its island GeoDataFrame (road
        segments with ``island_id`` and geometry).  When provided, allocations
        are built deterministically via :func:`get_or_build_allocation` for
        every road state encountered.  When ``None``, the legacy metadata-scan
        path is used as a backward-compatible fallback.
    service_area_function_provider_types:
        Optional ``{function_name: frozenset({provider_type, ...})}`` override
        for service-area routing.  Provider type tokens must match
        ``asset_type_column`` values exactly.  When ``None``, defaults to
        :data:`SERVICE_AREA_FUNCTION_PROVIDER_TYPES`.
    pop_grid_area_warn_buffer_m, pop_grid_area_warn_ratio:
        Work item B: cheap, non-fatal guard against an unclipped population
        grid.  If the bounding-box area of *pop_grid_gdf* exceeds
        ``pop_grid_area_warn_ratio`` times the area of the buffered
        (``pop_grid_area_warn_buffer_m``) convex hull of *gdf_assets*, a
        ``RuntimeWarning`` is emitted suggesting
        :func:`clip_population_to_service_area`.  This never raises or
        changes behaviour -- some callers may have legitimate reasons for a
        larger grid (e.g. a shared national grid reused across studies).

    Returns
    -------
    (updated_summary_results, updated_allocation_cache)
        *summary_results* is returned with societal fields merged in-place.
        *updated_allocation_cache* contains any newly built allocations.
    """
    if profiler is None:
        profiler = NULL_PROFILER
    if pop_group_columns is None:
        pop_group_columns = POPULATION_GROUP_COLUMNS
    if taxonomy is None:
        taxonomy = SERVICE_NODE_TAXONOMY
    if allocation_cache is None:
        allocation_cache = {}
    if service_area_function_provider_types is None:
        service_area_function_provider_types = SERVICE_AREA_FUNCTION_PROVIDER_TYPES

    # Determine asset types for service-node filtering
    asset_types = gdf_assets[asset_type_column].values if asset_type_column in gdf_assets.columns else np.array(["unknown"] * len(gdf_assets))

    # Determine stable asset IDs
    if asset_id_column and asset_id_column in gdf_assets.columns:
        stable_asset_ids = gdf_assets[asset_id_column].values
    else:
        stable_asset_ids = np.arange(len(gdf_assets))

    # Asset taxonomy and function alias preparation
    _num_assets_static = len(asset_types)
    asset_func_aliases: List[Tuple[str, ...]] = [()] * _num_assets_static
    for _i in range(_num_assets_static):
        _func_cat = taxonomy.get(str(asset_types[_i]))
        if _func_cat is not None:
            asset_func_aliases[_i] = tuple(_expand_function_category_equivalents(str(_func_cat)))
    provider_positions: List[int] = [
        _i for _i, _aliases in enumerate(asset_func_aliases) if _aliases
    ]

    _unknown_func_cat = taxonomy.get("unknown")
    unknown_func_aliases: Tuple[str, ...] = (
        tuple(_expand_function_category_equivalents(str(_unknown_func_cat)))
        if _unknown_func_cat is not None
        else ()
    )

    if verbose:
        if asset_type_column in gdf_assets.columns:
            asset_counts_by_type = (
                gdf_assets[asset_type_column].astype(str).value_counts().to_dict()
            )
            electricity_provider_count = int(
                gdf_assets[asset_type_column].astype(str).eq("msls").sum()
            )
            hospital_provider_count = int(
                gdf_assets[asset_type_column].astype(str).eq("hospital").sum()
            )
        else:
            asset_counts_by_type = {}
            electricity_provider_count = 0
            hospital_provider_count = 0
        print(f"Simulation assets: {len(gdf_assets)}")
        print(f"Asset counts by type: {asset_counts_by_type}")
        print(f"Electricity providers: {electricity_provider_count} MSLS")
        print(f"Hospital providers: {hospital_provider_count}")

    try:
        # Clip population grid to the service area defined by the assets
        if not pop_grid_gdf.empty and not gdf_assets.empty:
            _pgb = pop_grid_gdf.total_bounds
            _pop_grid_area = float((_pgb[2] - _pgb[0]) * (_pgb[3] - _pgb[1]))
            _hull_area = _service_area_hull_area(
                gdf_assets, pop_grid_gdf.crs, pop_grid_area_warn_buffer_m
            )
            if (
                _hull_area is not None
                and _pop_grid_area > 0
                and (_pop_grid_area / _hull_area) > pop_grid_area_warn_ratio
            ):
                warnings.warn(
                    "Population grid bounding-box area is "
                    f"{_pop_grid_area / _hull_area:.1f}x the buffered "
                    f"({pop_grid_area_warn_buffer_m:.0f}m) convex hull of "
                    "gdf_assets -- this looks unclipped and will make every "
                    "per-state population aggregation more expensive than "
                    "necessary. Consider pre-clipping with "
                    "clip_population_to_service_area().",
                    RuntimeWarning,
                    stacklevel=2,
                )
    except Exception:
        pass

    # Build service area population maps for the assets
    with profiler.section("societal_access._build_service_area_population_maps"):
        service_area_assets = gdf_assets.copy()
        service_area_assets.index = stable_asset_ids
        service_area_population_maps = _build_service_area_population_maps(
            gdf_assets=service_area_assets,
            pop_grid_gdf=pop_grid_gdf,
            pop_group_columns=pop_group_columns,
            asset_type_column=asset_type_column,
            service_area_function_provider_types=service_area_function_provider_types,
        )
    service_area_maps_digest = _service_area_population_maps_digest(
        service_area_population_maps
    )

    # Determine which functions to always emit
    if all_functions is None:
        all_functions = sorted(set(taxonomy.values()))

    # Build a lookup: summary_results keyed by timestep
    summary_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in summary_results}

    # Group detailed results by timestep
    detailed_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in detailed_results}

    # Pre-compute population grid hashes once
    pop_grid_geom_hash: str = _geometry_hash(pop_grid_gdf)
    pop_cell_id_hash: str = _column_hash(pop_grid_gdf, cell_id_column)

    # Caches for islands and population allocations
    _islands_hash_cache: Dict[str, Tuple[str, str]] = {}
    _pop_alloc_cache: Dict[str, Dict[str, Any]] = {}
    _pop_value_arrays_state: Optional[Tuple[pd.Index, Dict[str, np.ndarray]]] = None

    def _get_population_value_arrays() -> Tuple[pd.Index, Dict[str, np.ndarray]]:
        nonlocal _pop_value_arrays_state
        if _pop_value_arrays_state is None:
            _pop_value_arrays_state = _build_population_value_arrays(
                pop_grid_gdf, cell_id_column, pop_group_columns
            )
        return _pop_value_arrays_state

    # Precompute numpy arrays and positions for service area population maps
    _scalar_fields_cache: Dict[Any, Dict[str, float]] = {}
    _shared_realized_key_cache: Dict[Any, str] = {}
    _cache_metrics: Dict[str, int] = {
        "local_lookups": 0,
        "local_hits": 0,
        "local_misses": 0,
        "shared_lookups": 0,
        "shared_hits": 0,
        "shared_misses": 0,
        "shared_writes": 0,
        "shared_write_conflicts": 0,
        "shared_errors": 0,
    }
    _service_area_numpy: Dict[str, Dict[str, Any]] = {}
    _service_area_positions: Dict[str, Dict[str, Dict[Any, int]]] = {}
    for _func, _group_maps in service_area_population_maps.items():
        _service_area_numpy[_func] = {}
        _service_area_positions[_func] = {}
        for _label, _provider_pop in _group_maps.items():
            if _provider_pop:
                _ids = np.array(list(_provider_pop.keys()))
                _pops = np.array(list(_provider_pop.values()), dtype=float)
            else:
                _ids = np.array([])
                _pops = np.array([], dtype=float)
            _service_area_numpy[_func][_label] = (_ids, _pops)
            _service_area_positions[_func][_label] = {
                _provider_id: _position for _position, _provider_id in enumerate(_ids)
            }

    # Process each timestep that has detailed output
    for ts_idx, ts_summary in enumerate(summary_results):
        ts = ts_summary["timestep"]
        ts_detail = detailed_by_ts.get(ts)

        # Process the current timestep
        with profiler.timestep(ts, loop="societal_postprocess"):
            with profiler.section("societal_access.prepare_timestep_state"):
                # Prepare the state for this timestep, including early-exit checks and reading operational data
                if ts_detail is None:
                    # No detailed data → emit NaN
                    _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
                    continue

                # Read dependency-adjusted operational states
                operational = np.asarray(ts_detail.get("operational", []))
                island_ids = np.asarray(ts_detail.get("island_id", []))

                if len(operational) == 0 or len(island_ids) == 0:
                    _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
                    continue

            with profiler.section("societal_access.build_operational_provider_map"):
                # Build maps of operational assets by function and islands by function
                island_function_map: Dict[int, Set[str]] = {}
                operational_asset_ids_by_function: Dict[str, Set[Any]] = {}
                detail_len = min(len(operational), len(island_ids))
                for i in provider_positions:
                    if i >= detail_len or not operational[i]:
                        continue
                    isl_id_int = int(island_ids[i])
                    for func_alias in asset_func_aliases[i]:
                        operational_asset_ids_by_function.setdefault(func_alias, set()).add(stable_asset_ids[i])
                        island_function_map.setdefault(isl_id_int, set()).add(func_alias)

                if unknown_func_aliases and detail_len > _num_assets_static:
                    for i in range(_num_assets_static, detail_len):
                        if not operational[i]:
                            continue
                        isl_id_int = int(island_ids[i])
                        for func_alias in unknown_func_aliases:
                            operational_asset_ids_by_function.setdefault(func_alias, set()).add(stable_asset_ids[i])
                            island_function_map.setdefault(isl_id_int, set()).add(func_alias)

                frozen_island_function_map: Dict[int, FrozenSet[str]] = {
                    iid: frozenset(cats) for iid, cats in island_function_map.items()
                }

            with profiler.section("societal_access.prepare_timestep_state"):
                # Extract the road_state_key from the timestep detail and handle missing keys.
                road_state_key_raw = ts_detail.get("road_state_key")
                if not road_state_key_raw:
                    warnings.warn(
                        f"Timestep {ts}: road_state_key is absent; emitting NaN societal metrics. "
                        "Ensure the simulation records road_state_key in its timestep detail.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
                    continue
                road_state_key = str(road_state_key_raw)

            with profiler.section("societal_access.resolve_allocation"):
                # Get or populate islands hashes for this road_state_key ---
                if road_state_key not in _islands_hash_cache and islands_gdf_cache is not None:
                    islands_gdf_tmp = islands_gdf_cache.get(road_state_key)
                    if islands_gdf_tmp is not None:
                        _islands_hash_cache[road_state_key] = (
                            _geometry_hash(islands_gdf_tmp),
                            _column_hash(islands_gdf_tmp, island_id_column),
                        )
                _cur_islands_geom_hash, _cur_islands_id_hash = _islands_hash_cache.get(
                    road_state_key, (None, None)
                )

                # Resolve allocation_df — deterministic path when islands_gdf_cache is
                # available, legacy metadata-scan path otherwise (backward compat).
                allocation_cache_key = None
                if islands_gdf_cache is not None:
                    islands_gdf = islands_gdf_cache.get(road_state_key)
                    if islands_gdf is None:
                        message = (
                            f"Timestep {ts}: no islands_gdf found for road_state_key "
                            f"'{road_state_key}'; no societal allocation can be built."
                        )
                        if fail_on_missing_allocation:
                            raise RuntimeError(message)
                        warnings.warn(
                            f"{message} Emitting NaN societal metrics.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
                        continue
                    allocation_df, allocation_cache_key, _ = get_or_build_allocation(
                        allocation_cache,
                        pop_grid_gdf,
                        cell_id_column,
                        islands_gdf,
                        island_id_column,
                        nearest_max_distance=nearest_max_distance,
                        road_state_key=road_state_key,
                        pop_grid_geom_hash=pop_grid_geom_hash,
                        pop_cell_id_hash=pop_cell_id_hash,
                        islands_geom_hash=_cur_islands_geom_hash,
                        islands_id_hash=_cur_islands_id_hash,
                    )
                else:
                    # Pass pre-computed pop grid hashes ---
                    allocation_cache_key, allocation_df = _find_allocation_cache_entry(
                        allocation_cache,
                        pop_grid_gdf,
                        cell_id_column,
                        road_state_key,
                        nearest_max_distance,
                        pop_grid_geom_hash=pop_grid_geom_hash,
                        pop_cell_id_hash=pop_cell_id_hash,
                    )

                if allocation_df is None and fail_on_missing_allocation:
                    available_road_state_keys = sorted(
                        {
                            str(df.attrs.get("road_state_key"))
                            for df in allocation_cache.values()
                            if df.attrs.get("road_state_key") is not None
                        }
                    )
                    raise RuntimeError(
                        f"Timestep {ts}: no societal allocation found for road_state_key "
                        f"'{road_state_key}'. Available cached road_state_keys="
                        f"{available_road_state_keys[:10]}"
                    )

                if allocation_df is not None:
                    allocation_road_state_key = str(allocation_df.attrs.get("road_state_key"))
                    if allocation_road_state_key != road_state_key:
                        message = (
                            f"Timestep {ts}: allocation road_state_key '{allocation_road_state_key}' "
                            f"does not match detailed_results road_state_key '{road_state_key}'."
                        )
                        if fail_on_missing_allocation:
                            raise RuntimeError(message)
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
                        continue

            with profiler.section("societal_access.compute_scalars"):
                # Compute societal access scalars (i.e. the metrics that quantify societal access) for the current timestep.
                with profiler.section("societal_access.compute_scalars.pop_alloc"):
                    _cached_entry: Optional[Dict[str, Any]] = None
                    if allocation_cache_key is not None and allocation_df is not None:
                        if allocation_cache_key not in _pop_alloc_cache:
                            with profiler.section("societal_access.compute_scalars.pop_alloc_build"):
                                try:
                                    _cell_id_index, _value_arrays = _get_population_value_arrays()
                                    _isl_pop, _tot_pop, _agc = _aggregate_population_by_island(
                                        allocation_df=allocation_df,
                                        cell_id_column=cell_id_column,
                                        cell_id_index=_cell_id_index,
                                        value_arrays=_value_arrays,
                                    )

                                    _pop_alloc_cache[allocation_cache_key] = {
                                        "island_pop": _isl_pop,
                                        "total_pop": _tot_pop,
                                        "available_group_cols": _agc,
                                    }
                                except Exception:
                                    pass
                        _cached_entry = _pop_alloc_cache.get(allocation_cache_key)

                # Compute the cache key for the scalar fields and compute scalar values if cache miss.
                with profiler.section("societal_access.compute_scalars.cache_key"):
                    _frozen_ifm_key = frozenset(frozen_island_function_map.items())
                    _op_signature = tuple(
                        (func, frozenset(ids))
                        for func, ids in sorted(operational_asset_ids_by_function.items())
                    )
                    _scalar_cache_key = (allocation_cache_key, _frozen_ifm_key, _op_signature)
                    _cache_metrics["local_lookups"] += 1
                    _scalar_cache_hit = _scalar_cache_key in _scalar_fields_cache
                    if _scalar_cache_hit:
                        _cache_metrics["local_hits"] += 1
                    else:
                        _cache_metrics["local_misses"] += 1

                if _scalar_cache_hit:
                    with profiler.section("societal_access.compute_scalars.cache_hit"):
                        # A cache hit must never hand out the same mutable dict twice:
                        # _apply_service_area_societal_scalars mutates its
                        # `fields` argument in place, so always return a copy.
                        societal_fields = dict(_scalar_fields_cache[_scalar_cache_key])
                else:
                    _shared_cache_hit = False
                    _shared_cache_key = ""
                    if shared_realized_state_cache is not None:
                        _shared_cache_key = _shared_realized_key_cache.get(_scalar_cache_key, "")
                        if not _shared_cache_key:
                            _shared_cache_key = _build_realized_state_cache_key(
                                frozen_island_function_map=frozen_island_function_map,
                                operational_asset_ids_by_function=operational_asset_ids_by_function,
                                island_pop=_cached_entry["island_pop"] if _cached_entry else None,
                                total_pop=_cached_entry["total_pop"] if _cached_entry else None,
                                available_group_cols=_cached_entry["available_group_cols"] if _cached_entry else None,
                                all_functions=all_functions,
                                pop_group_columns=pop_group_columns,
                                reference_group=reference_group,
                                service_area_function_provider_types=service_area_function_provider_types,
                                allocation_df=allocation_df,
                                service_area_population_maps_digest=service_area_maps_digest,
                            )
                            if _shared_cache_key:
                                _shared_realized_key_cache[_scalar_cache_key] = _shared_cache_key
                    if _shared_cache_key and shared_realized_state_cache is not None:
                        _cache_metrics["shared_lookups"] += 1
                        try:
                            _shared_fields = shared_realized_state_cache.get(_shared_cache_key)
                            if isinstance(_shared_fields, dict):
                                societal_fields = dict(_shared_fields)
                                _scalar_fields_cache[_scalar_cache_key] = dict(societal_fields)
                                _cache_metrics["shared_hits"] += 1
                                _shared_cache_hit = True
                            else:
                                _cache_metrics["shared_misses"] += 1
                        except Exception as _shared_cache_err:
                            _cache_metrics["shared_errors"] += 1
                            if shared_cache_fail_hard:
                                raise SharedRealizedStateCacheError(
                                    "Shared realized-state cache read failed"
                                ) from _shared_cache_err

                    if not _shared_cache_hit:
                        with profiler.section("societal_access.compute_scalars.cache_miss"):
                            societal_fields = _compute_societal_scalars(
                                frozen_island_function_map=frozen_island_function_map,
                                allocation_df=allocation_df,
                                pop_grid_gdf=pop_grid_gdf,
                                cell_id_column=cell_id_column,
                                pop_group_columns=pop_group_columns,
                                all_functions=all_functions,
                                reference_group=reference_group,
                                island_pop=_cached_entry["island_pop"] if _cached_entry else None,
                                total_pop=_cached_entry["total_pop"] if _cached_entry else None,
                                available_group_cols=_cached_entry["available_group_cols"] if _cached_entry else None,
                            )

                            # Apply service area societal scalars to the computed societal fields.
                            societal_fields = _apply_service_area_societal_scalars(
                                societal_fields,
                                operational_asset_ids_by_function=operational_asset_ids_by_function,
                                service_area_population_maps=service_area_population_maps,
                                pop_group_columns=pop_group_columns,
                                reference_group=reference_group,
                                numpy_maps=_service_area_numpy if _service_area_numpy else None,
                                position_maps=_service_area_positions if _service_area_positions else None,
                            )
                            _scalar_fields_cache[_scalar_cache_key] = dict(societal_fields)
                            if _shared_cache_key and shared_realized_state_cache is not None:
                                try:
                                    _inserted = shared_realized_state_cache.set_if_absent(
                                        _shared_cache_key, dict(societal_fields)
                                    )
                                    if _inserted:
                                        _cache_metrics["shared_writes"] += 1
                                    else:
                                        _cache_metrics["shared_write_conflicts"] += 1
                                except Exception as _shared_cache_err:
                                    _cache_metrics["shared_errors"] += 1
                                    if shared_cache_fail_hard:
                                        raise SharedRealizedStateCacheError(
                                            "Shared realized-state cache write failed"
                                        ) from _shared_cache_err

            with profiler.section("societal_access.merge_fields"):
                ts_summary.update(societal_fields)
                ts_summary["allocation_cache_key"] = allocation_cache_key
                ts_summary["allocation_road_state_key"] = road_state_key

    if cache_telemetry is not None:
        for _metric_name, _metric_value in _cache_metrics.items():
            cache_telemetry[_metric_name] = cache_telemetry.get(_metric_name, 0) + int(_metric_value)
        if shared_realized_state_cache is not None and hasattr(shared_realized_state_cache, "get_stats"):
            try:
                _backend_stats = shared_realized_state_cache.get_stats()
                for _metric_name, _metric_value in _backend_stats.items():
                    cache_telemetry[f"shared_backend_{_metric_name}"] = int(_metric_value)
            except Exception:
                cache_telemetry["shared_backend_stats_error"] = 1

    return summary_results, allocation_cache


def _compute_societal_scalars(
    frozen_island_function_map: Dict[int, FrozenSet[str]],
    allocation_df: Optional[pd.DataFrame],
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    pop_group_columns: Dict[str, str],
    all_functions: List[str],
    reference_group: str,
    pop_alloc: Optional[pd.DataFrame] = None,
    island_pop: Optional[pd.DataFrame] = None,
    total_pop: Optional[Dict[str, float]] = None,
    available_group_cols: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute societal metric scalars for one timestep.

    Parameters
    ----------
    frozen_island_function_map:
        Mapping of island_id → frozenset of function categories supplied by
        operational providers on that island.
    allocation_df:
        Pre-built (or freshly built) population-cell → island allocation
        DataFrame.  When ``None``, all output metrics are emitted as NaN.
    pop_grid_gdf:
        Population grid GeoDataFrame used to apply demographic weights.
    cell_id_column:
        Stable identifier column in *pop_grid_gdf*.
    pop_group_columns:
        ``{label: column_name}`` demographic mapping.
    all_functions:
        Complete list of function categories to emit.
    reference_group:
        Reference group for equity-gap computation.
    pop_alloc:
        Optional pre-built output of :func:`apply_population_to_allocations`.
        When provided, the internal ``apply_population_to_allocations`` call is
        skipped (Problem 3 optimisation).
    island_pop:
        Optional pre-aggregated per-island population DataFrame (Fix 1).
        When provided, the groupby is skipped entirely.
    total_pop:
        Optional pre-computed total population per group (Fix 2).
        When provided, the per-group sum is skipped.
    available_group_cols:
        Optional pre-computed ``{label: weighted_col_name}`` mapping (Fix 1).

    Returns a dict of flat metric names → scalar values.
    """
    fields: Dict[str, float] = {}

    def _emit_nan() -> Dict[str, float]:
        nan_fields: Dict[str, float] = {}
        for func in all_functions:
            for group in pop_group_columns:
                nan_fields[f"societal_access_pct__{func}__{group}"] = float("nan")
                nan_fields[f"societal_access_population__{func}__{group}"] = float("nan")
                nan_fields[f"societal_no_access_population__{func}__{group}"] = float("nan")
        for group in pop_group_columns:
            nan_fields[f"societal_total_population__{group}"] = float("nan")
        for func in all_functions:
            for group in pop_group_columns:
                if group == reference_group:
                    continue
                nan_fields[f"societal_equity_absolute_gap__{func}__{group}"] = float("nan")
                nan_fields[f"societal_equity_relative_gap__{func}__{group}"] = float("nan")
        return nan_fields

    if allocation_df is None:
        return _emit_nan()

    # Apply population to allocations (skip if pre-built pop_alloc provided)
    if pop_alloc is None:
        try:
            pop_alloc = apply_population_to_allocations(
                allocation_df=allocation_df,
                pop_grid_gdf=pop_grid_gdf,
                cell_id_column=cell_id_column,
                pop_group_columns=pop_group_columns,
            )
        except Exception:
            return _emit_nan()

    # --- Fix 1: use pre-aggregated island_pop / total_pop when available ---
    if available_group_cols is None:
        group_cols = {label: f"{label}_weighted" for label in pop_group_columns}
        available_group_cols = {
            label: col for label, col in group_cols.items()
            if col in pop_alloc.columns
        }

    if island_pop is None:
        weighted_col_list = list(available_group_cols.values())
        island_pop = (
            pop_alloc.groupby("island_id")[weighted_col_list].sum()
            if weighted_col_list
            else pd.DataFrame()
        )

    if total_pop is None:
        total_pop = {}
        if not island_pop.empty:
            for label, col in available_group_cols.items():
                total_pop[label] = float(island_pop[col].sum())
        else:
            for label, col in available_group_cols.items():
                total_pop[label] = float(pop_alloc[col].sum())

    # Population with access per (function, group) — use island_pop for efficiency
    for func in all_functions:
        islands_with_func = [
            iid for iid, cats in frozen_island_function_map.items()
            if func in cats and iid != -1
        ]

        for label, col in available_group_cols.items():
            total = total_pop.get(label, 0.0)
            if islands_with_func and not island_pop.empty:
                valid_ids = [iid for iid in islands_with_func if iid in island_pop.index]
                with_access = float(island_pop.loc[valid_ids, col].sum()) if valid_ids else 0.0
            else:
                with_access = 0.0
            without_access = total - with_access
            pct = round(100.0 * with_access / total, 2) if total > 0 else float("nan")

            fields[f"societal_access_pct__{func}__{label}"] = pct
            fields[f"societal_access_population__{func}__{label}"] = with_access
            fields[f"societal_no_access_population__{func}__{label}"] = without_access

    for label in available_group_cols:
        fields[f"societal_total_population__{label}"] = total_pop.get(label, 0.0)

    # Equity gaps vs reference group
    ref_label = reference_group
    for func in all_functions:
        ref_pct = fields.get(f"societal_access_pct__{func}__{ref_label}", float("nan"))
        for label in pop_group_columns:
            if label == ref_label:
                continue
            grp_pct = fields.get(f"societal_access_pct__{func}__{label}", float("nan"))
            if np.isnan(ref_pct) or np.isnan(grp_pct):
                abs_gap = float("nan")
                rel_gap = float("nan")
            else:
                abs_gap = round(ref_pct - grp_pct, 2)
                rel_gap = round(grp_pct / ref_pct, 4) if ref_pct > 0 else float("nan")
            fields[f"societal_equity_absolute_gap__{func}__{label}"] = abs_gap
            fields[f"societal_equity_relative_gap__{func}__{label}"] = rel_gap

    return fields


def _merge_zero_societal_metrics(
    ts_summary: Dict[str, Any],
    all_functions: List[str],
    pop_group_columns: Dict[str, str],
    reference_group: str,
) -> None:
    """Emit NaN societal metrics when no detailed data is available."""
    for func in all_functions:
        for group in pop_group_columns:
            ts_summary[f"societal_access_pct__{func}__{group}"] = float("nan")
            ts_summary[f"societal_access_population__{func}__{group}"] = float("nan")
            ts_summary[f"societal_no_access_population__{func}__{group}"] = float("nan")
        for group in pop_group_columns:
            ts_summary[f"societal_total_population__{group}"] = float("nan")
    for func in all_functions:
        for group in pop_group_columns:
            if group == reference_group:
                continue
            ts_summary[f"societal_equity_absolute_gap__{func}__{group}"] = float("nan")
            ts_summary[f"societal_equity_relative_gap__{func}__{group}"] = float("nan")


def list_societal_metric_names(
    all_functions: List[str],
    pop_group_columns: Dict[str, str],
    reference_group: str = "total",
) -> List[str]:
    """Return the complete list of flat societal metric field names.

    Used to pre-initialise EMA output arrays before running experiments.
    """
    names: List[str] = []
    for func in all_functions:
        for group in pop_group_columns:
            names.append(f"societal_access_pct__{func}__{group}")
            names.append(f"societal_access_population__{func}__{group}")
            names.append(f"societal_no_access_population__{func}__{group}")
    for group in pop_group_columns:
        names.append(f"societal_total_population__{group}")
    for func in all_functions:
        for group in pop_group_columns:
            if group == reference_group:
                continue
            names.append(f"societal_equity_absolute_gap__{func}__{group}")
            names.append(f"societal_equity_relative_gap__{func}__{group}")
    return names
