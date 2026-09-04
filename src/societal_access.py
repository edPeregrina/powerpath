"""Generalised access-to-critical-societal-functions analysis.

Conceptual model
----------------
This module operationalises access as **shared island membership** in a
disrupted graph, following the set-theoretic framework of Paper 4:

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

Implementation layers
---------------------
A — Graph-native (preferred): nodes are simulation entities directly
    embedded in the graph; island assignment is read from graph connectivity.
B — Spatial preprocessing: service nodes and population zones are geographic
    objects spatially joined to island polygons.
C — Aggregate metrics: shared by both paths.
D — High-level wrapper: ``analyse_societal_access``.
E — Allocation layer: geometry-only cell→island fraction caching plus
    population application and realization-aware postprocessing.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.timing_profiler import NULL_PROFILER


# ---------------------------------------------------------------------------
# Service node taxonomy
# ---------------------------------------------------------------------------

#: Default mapping: node *type* string → *function category* label.
SERVICE_NODE_TAXONOMY: Dict[str, str] = {
    # Health
    "hospital": "health",
    "clinic": "health",
    "huisartsenpraktijk": "health",
    "apotheek": "health",
    # Emergency response
    "fire_station": "emergency_response",
    "brandweerkazerne": "emergency_response",
    "emergency_operations_centre": "emergency_response",
    # Climate resilience
    "cooling_centre": "climate_resilience",
    "water_supply_point": "climate_resilience",
    "drinking_water": "climate_resilience",
    # Social continuity
    "school": "education",
    "basisonderwijs": "education",
    "voortgezet_onderwijs": "education",
    "repair_depot": "repair_logistics",
    # Electricity
    "msls": "electricity",
}

#: Default demographic columns in the CBS population grid.
POPULATION_GROUP_COLUMNS: Dict[str, str] = {
    "total": "aantal_inwoners",
    "elderly": "aantal_inwoners_65_jaar_en_ouder",
    "children": "aantal_inwoners_0_tot_15_jaar",
    "working_age": "aantal_inwoners_25_tot_45_jaar",
}

#: Current version of the allocation algorithm — increment when the
#: spatial logic changes so cached allocations are automatically invalidated.
ALLOCATION_ALGORITHM_VERSION: str = "1.0.0"
SERVICE_AREA_FUNCTION_PROVIDER_TYPES: Dict[str, FrozenSet[str]] = {
    "electricity": frozenset({"msls"}),
}
_LEGACY_ALLOCATION_LOOKUP_WARNED = False
_FUNCTION_CATEGORY_EQUIVALENTS: Dict[str, FrozenSet[str]] = {
    "health": frozenset({"health", "hospital"}),
    "hospital": frozenset({"health", "hospital"}),
}


def _expand_function_category_equivalents(function_name: str) -> FrozenSet[str]:
    """Return equivalent function-category labels for compatibility outputs."""
    return _FUNCTION_CATEGORY_EQUIVALENTS.get(function_name, frozenset({function_name}))


# ---------------------------------------------------------------------------
# Layer A — node-level, graph-based (primary path)
# ---------------------------------------------------------------------------

def build_island_assignment(graph) -> Dict[Any, int]:
    """Compute the island partition of a disrupted graph."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "NetworkX is required for graph-based island assignment. "
            "Install it with: pip install networkx"
        ) from exc

    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    assignment: Dict[Any, int] = {}
    for island_id, component in enumerate(components):
        for node in component:
            assignment[node] = island_id

    return assignment


def build_destination_function_map(
    destination_nodes: Mapping[Any, str],
    node_island_assignment: Mapping[Any, int],
    taxonomy: Optional[Dict[str, str]] = None,
) -> Dict[int, FrozenSet[str]]:
    """Map each island to the set of societal functions it contains."""
    if taxonomy is None:
        taxonomy = SERVICE_NODE_TAXONOMY

    island_functions: Dict[int, Set[str]] = {}
    missing: List[Any] = []

    for node_id, node_type in destination_nodes.items():
        island_id = node_island_assignment.get(node_id)
        if island_id is None:
            missing.append(node_id)
            continue
        func_cat = taxonomy.get(str(node_type))
        if func_cat is not None:
            island_functions.setdefault(island_id, set()).add(func_cat)

    if missing:
        warnings.warn(
            f"{len(missing)} destination node(s) not found in island assignment "
            f"(likely removed by disruption): {missing[:5]}"
            + (" …" if len(missing) > 5 else ""),
            UserWarning,
            stacklevel=2,
        )

    return {iid: frozenset(cats) for iid, cats in island_functions.items()}


def compute_origin_access(
    origin_island_ids: Mapping[Any, int],
    island_function_map: Mapping[int, FrozenSet[str]],
    all_functions: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Evaluate per-origin access for every societal function."""
    if all_functions is None:
        all_functions = sorted(
            {f for cats in island_function_map.values() for f in cats}
        )
    else:
        all_functions = sorted(all_functions)

    records = {}
    for origin_id, island_id in origin_island_ids.items():
        island_cats = island_function_map.get(island_id, frozenset())
        records[origin_id] = {f: (f in island_cats) for f in all_functions}

    df = pd.DataFrame.from_dict(records, orient="index", columns=all_functions)
    df.index.name = "origin_id"
    df.columns.name = "function"
    return df


def compute_access_matrix_from_origins(
    origin_access_df: pd.DataFrame,
    stakeholder_groups: Mapping[str, Mapping[Any, float]],
) -> pd.DataFrame:
    """Aggregate per-origin access flags into group-level percentage metrics."""
    functions = list(origin_access_df.columns)
    groups = list(stakeholder_groups.keys())

    result = pd.DataFrame(
        index=pd.Index(functions, name="function"),
        columns=pd.Index(groups, name="population_group"),
        dtype=float,
    )

    for group_label, members in stakeholder_groups.items():
        valid_ids = [oid for oid in members if oid in origin_access_df.index]
        if not valid_ids:
            result[group_label] = np.nan
            continue
        weights = pd.Series({oid: members[oid] for oid in valid_ids}, dtype=float)
        total_weight = weights.sum()
        if total_weight == 0:
            result[group_label] = np.nan
            continue
        sub_access = origin_access_df.loc[valid_ids]
        weighted_access = sub_access.mul(weights, axis=0).sum(axis=0)
        result[group_label] = (weighted_access / total_weight * 100).round(2)

    return result


# ---------------------------------------------------------------------------
# Layer B — spatial preprocessing helpers (optional)
# ---------------------------------------------------------------------------

def assign_destinations_to_islands_spatial(
    service_nodes_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    taxonomy: Optional[Dict[str, str]] = None,
    node_type_column: str = "type",
    island_id_column: str = "island_id",
    buffer_m: float = 50.0,
) -> Dict[int, FrozenSet[str]]:
    """Return the set of function categories reachable within each island (spatial path)."""
    if taxonomy is None:
        taxonomy = SERVICE_NODE_TAXONOMY

    if service_nodes_gdf is None or service_nodes_gdf.empty:
        return {}

    target_crs = islands_gdf.crs or "EPSG:28992"
    service_nodes = service_nodes_gdf.copy().to_crs(target_crs)
    islands = islands_gdf[[island_id_column, "geometry"]].copy()

    islands_dissolved = islands.dissolve(by=island_id_column).reset_index()

    if buffer_m > 0:
        service_nodes = service_nodes.copy()
        service_nodes["geometry"] = service_nodes.geometry.centroid.buffer(buffer_m)

    joined = gpd.sjoin(
        service_nodes[[node_type_column, "geometry"]],
        islands_dissolved[[island_id_column, "geometry"]],
        how="left",
        predicate="intersects",
    )

    island_functions: Dict[int, Set[str]] = {}
    for _, row in joined.iterrows():
        if pd.isna(row.get(island_id_column)):
            continue
        island_id = int(row[island_id_column])
        node_type = row[node_type_column]
        func_cat = taxonomy.get(str(node_type))
        if func_cat is not None:
            island_functions.setdefault(island_id, set()).add(func_cat)

    return {iid: frozenset(cats) for iid, cats in island_functions.items()}


# Backward-compatibility alias
compute_function_access_per_island = assign_destinations_to_islands_spatial


def assign_origins_to_islands_spatial(
    population_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    pop_columns: Optional[Sequence[str]] = None,
    island_id_column: str = "island_id",
) -> pd.DataFrame:
    """Spatially assign each population zone to an island (spatial path)."""
    if pop_columns is None:
        pop_columns = list(POPULATION_GROUP_COLUMNS.values())

    target_crs = islands_gdf.crs or "EPSG:28992"
    pop = population_gdf[list(pop_columns) + ["geometry"]].copy().to_crs(target_crs)
    islands = islands_gdf[[island_id_column, "geometry"]].copy()

    islands_dissolved = islands.dissolve(by=island_id_column).reset_index()

    joined = gpd.sjoin_nearest(
        pop,
        islands_dissolved[[island_id_column, "geometry"]],
        how="left",
        max_distance=200,
    )

    if joined.index.duplicated().any():
        joined = joined.reset_index(drop=False)
        joined["_inter_area"] = joined.apply(
            lambda r: pop.loc[r["index"], "geometry"].intersection(
                islands_dissolved.loc[
                    islands_dissolved[island_id_column] == r[island_id_column],
                    "geometry",
                ].iloc[0]
                if not islands_dissolved[
                    islands_dissolved[island_id_column] == r[island_id_column]
                ].empty
                else pop.loc[r["index"], "geometry"]
            ).area,
            axis=1,
        )
        joined = (
            joined.sort_values("_inter_area", ascending=False)
            .drop_duplicates(subset=["index"])
            .set_index("index")
        )

    joined[island_id_column] = joined[island_id_column].fillna(-1).astype(int)

    result = joined[list(pop_columns) + [island_id_column]].copy()
    for col in pop_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result[col] = result[col].where(result[col] >= 0, 0)

    return result


# Backward-compatibility alias
join_population_to_islands = assign_origins_to_islands_spatial


# ---------------------------------------------------------------------------
# Layer C — aggregate metrics (shared by both paths)
# ---------------------------------------------------------------------------

def compute_access_matrix(
    island_function_map: Dict[int, FrozenSet[str]],
    island_population_df: pd.DataFrame,
    pop_columns: Optional[Dict[str, str]] = None,
    island_id_column: str = "island_id",
    all_functions: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute percentage access for every (function, population group) pair (spatial path)."""
    if pop_columns is None:
        pop_columns = POPULATION_GROUP_COLUMNS

    if all_functions is None:
        all_functions = sorted(
            {f for cats in island_function_map.values() for f in cats}
        )
    else:
        all_functions = sorted(all_functions)

    if not all_functions:
        return pd.DataFrame(
            index=pd.Index([], name="function"),
            columns=list(pop_columns.keys()),
            dtype=float,
        )

    df = island_population_df.copy()
    col_names = list(pop_columns.values())
    totals = df[col_names].sum()

    rows = []
    for func in all_functions:
        islands_with_func = {
            iid for iid, cats in island_function_map.items() if func in cats
        }
        mask = df[island_id_column].isin(islands_with_func)
        accessible_pop = df.loc[mask, col_names].sum()
        pct = {}
        for label, col in pop_columns.items():
            total = totals.get(col, 0)
            acc = accessible_pop.get(col, 0)
            pct[label] = round(100.0 * acc / total, 2) if total > 0 else np.nan
        rows.append(pct)

    result = pd.DataFrame(rows, index=pd.Index(all_functions, name="function"))
    result.columns.name = "population_group"
    return result


def compute_equity_gaps(
    access_matrix: pd.DataFrame,
    reference_group: str = "total",
) -> pd.DataFrame:
    """Surface the equity gap between a reference group and all other groups."""
    if reference_group not in access_matrix.columns:
        raise ValueError(
            f"Reference group '{reference_group}' not found in access_matrix columns: "
            f"{list(access_matrix.columns)}"
        )

    ref = access_matrix[reference_group]
    other_groups = [c for c in access_matrix.columns if c != reference_group]

    result = pd.DataFrame(index=access_matrix.index)

    for group in other_groups:
        result[f"{group}_absolute_gap"] = (ref - access_matrix[group]).round(2)
        result[f"{group}_relative_gap"] = (
            (access_matrix[group] / ref).where(ref > 0).round(4)
        )

    if other_groups:
        abs_gap_cols = [f"{g}_absolute_gap" for g in other_groups]
        result["most_disadvantaged_group"] = (
            result[abs_gap_cols]
            .idxmax(axis=1)
            .str.replace("_absolute_gap", "", regex=False)
        )
        result["max_absolute_gap"] = result[abs_gap_cols].max(axis=1).round(2)
    else:
        result["most_disadvantaged_group"] = np.nan
        result["max_absolute_gap"] = np.nan

    return result


# ---------------------------------------------------------------------------
# Layer E — allocation caching + realization-aware postprocessor
# ---------------------------------------------------------------------------

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
    grid geometry.  It does **not** include population attribute values,
    selected demographic columns, stochastic seed, or provider outcomes.

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

    # Resolve hashes — use pre-computed values when available (Problem 1)
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


def _canonicalize_service_area_provider_types(
    spec: Mapping[str, Iterable[str]],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Return a hashable, order-independent form of a provider-types spec."""
    return tuple(
        (str(function_name), tuple(sorted(str(provider_type) for provider_type in provider_types)))
        for function_name, provider_types in sorted(spec.items(), key=lambda kv: str(kv[0]))
    )


def _service_area_pop_map_cache_key(
    asset_cache_key: str,
    pop_grid_gdf: gpd.GeoDataFrame,
    pop_group_columns: Dict[str, str],
    service_area_function_provider_types: Dict[str, FrozenSet[str]],
    asset_type_column: str,
) -> Tuple[Any, ...]:
    """Build the memo key for :func:`_build_service_area_population_maps`."""
    return (
        _SERVICE_AREA_POP_MAP_CACHE_VERSION,
        asset_cache_key,
        _geometry_hash(pop_grid_gdf),
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

    if service_area_function_provider_types is None:
        service_area_function_provider_types = SERVICE_AREA_FUNCTION_PROVIDER_TYPES

    cache_key = _service_area_pop_map_cache_key(
        asset_cache_key,
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

    # Determine asset types for service-node filtering
    asset_types = gdf_assets[asset_type_column].values if asset_type_column in gdf_assets.columns else np.array(["unknown"] * len(gdf_assets))

    # Determine stable asset IDs
    if asset_id_column and asset_id_column in gdf_assets.columns:
        stable_asset_ids = gdf_assets[asset_id_column].values
    else:
        stable_asset_ids = np.arange(len(gdf_assets))

    # --- Work item 3: hoist per-asset taxonomy work out of the timestep loop.
    # `asset_types` and `taxonomy` are fixed for the whole call, so the
    # function-category → alias expansion (previously repeated for every
    # asset on every timestep) is computed once here.  `asset_func_aliases[i]`
    # is the tuple of function aliases for asset `i` (empty for
    # non-providers); `provider_positions` lists only the positions with a
    # non-empty alias tuple, so the per-timestep loop below need only visit
    # actual providers instead of every asset.
    _num_assets_static = len(asset_types)
    asset_func_aliases: List[Tuple[str, ...]] = [()] * _num_assets_static
    for _i in range(_num_assets_static):
        _func_cat = taxonomy.get(str(asset_types[_i]))
        if _func_cat is not None:
            asset_func_aliases[_i] = tuple(_expand_function_category_equivalents(str(_func_cat)))
    provider_positions: List[int] = [
        _i for _i, _aliases in enumerate(asset_func_aliases) if _aliases
    ]
    # Preserve the `i >= len(asset_types) -> "unknown"` fallback semantics for
    # the (essentially never triggered) case where the per-timestep detail
    # arrays are longer than the static asset arrays.
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

    # Determine which functions to always emit
    if all_functions is None:
        all_functions = sorted(set(taxonomy.values()))

    # Build a lookup: summary_results keyed by timestep
    summary_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in summary_results}

    # Group detailed results by timestep
    detailed_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in detailed_results}

    # --- Problem 1: Pre-compute population grid hashes once ---
    pop_grid_geom_hash: str = _geometry_hash(pop_grid_gdf)
    pop_cell_id_hash: str = _column_hash(pop_grid_gdf, cell_id_column)

    # Per road_state_key islands hash cache (populated on first encounter)
    _islands_hash_cache: Dict[str, Tuple[str, str]] = {}

    # --- Problem 3: pop_alloc cache keyed by allocation_cache_key ---
    # Each entry: {"pop_alloc": df, "island_pop": df, "total_pop": dict, "available_group_cols": dict}
    _pop_alloc_cache: Dict[str, Dict[str, Any]] = {}

    # --- Problem 6: cache _compute_societal_scalars output per (allocation_cache_key, frozen_island_function_map) ---
    _scalar_fields_cache: Dict[Any, Dict[str, float]] = {}

    # --- Problem 4: precompute numpy arrays for _apply_service_area_societal_scalars ---
    # --- Work item 5: also precompute a {provider_id: position} mapping per
    # function/label, aligned with the ids array, so the operational-provider
    # membership mask can be built via O(len(operational_ids)) position
    # lookups instead of rebuilding a list from operational_ids and sorting
    # it on every call (np.isin) -- ids_arr is fixed for the whole run.
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

        with profiler.timestep(ts, loop="societal_postprocess"):
            # --- Instrumentation follow-up: these two early-exit checks were
            # previously outside any named section (part of the profiler's
            # "societal_postprocess.unattributed" blind spot). They are cheap
            # per-timestep validation/state-prep, grouped under the same
            # section name as the road_state_key resolution below since both
            # are "prepare this timestep's state" work.
            with profiler.section("societal_access.prepare_timestep_state"):
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
                # Build island → functions map from operational providers.
                # --- Work item 3: iterate only over precomputed provider
                # positions instead of every asset on every timestep.
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
                # Strict road_state_key resolution — no fallback to map index.
                # Falling back to the map counter would silently reuse the baseline
                # topology for adapted states that happen to share the same counter.
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
                # --- Problem 1: get or populate islands hashes for this road_state_key ---
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
                    # --- Problem 5: pass pre-computed pop grid hashes ---
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
                # --- Fix 1+2: retrieve or build pop_alloc + island_pop + total_pop ---
                # --- Investigation follow-up: named as its own subphase so the
                # allocation (pop_alloc) build work is distinguishable from
                # cache-key construction and the scalar cache hit/miss paths
                # below (all previously folded into one "compute_scalars" time).
                with profiler.section("societal_access.compute_scalars.pop_alloc"):
                    _cached_entry: Optional[Dict[str, Any]] = None
                    if allocation_cache_key is not None and allocation_df is not None:
                        if allocation_cache_key not in _pop_alloc_cache:
                            with profiler.section("societal_access.compute_scalars.pop_alloc_build"):
                                try:
                                    _pop_alloc_df = apply_population_to_allocations(
                                        allocation_df=allocation_df,
                                        pop_grid_gdf=pop_grid_gdf,
                                        cell_id_column=cell_id_column,
                                        pop_group_columns=pop_group_columns,
                                    )
                                    _gc = {lbl: f"{lbl}_weighted" for lbl in pop_group_columns}
                                    _agc = {lbl: col for lbl, col in _gc.items() if col in _pop_alloc_df.columns}
                                    _wcl = list(_agc.values())
                                    _isl_pop = (
                                        _pop_alloc_df.groupby("island_id")[_wcl].sum()
                                        if _wcl
                                        else pd.DataFrame()
                                    )
                                    _tot_pop: Dict[str, float] = {}
                                    if not _isl_pop.empty:
                                        for _lbl, _col in _agc.items():
                                            _tot_pop[_lbl] = float(_isl_pop[_col].sum())
                                    else:
                                        for _lbl, _col in _agc.items():
                                            _tot_pop[_lbl] = float(_pop_alloc_df[_col].sum())
                                    _pop_alloc_cache[allocation_cache_key] = {
                                        "pop_alloc": _pop_alloc_df,
                                        "island_pop": _isl_pop,
                                        "total_pop": _tot_pop,
                                        "available_group_cols": _agc,
                                    }
                                except Exception:
                                    pass
                        _cached_entry = _pop_alloc_cache.get(allocation_cache_key)

                # --- Fix 6 / Work item 4: cache _compute_societal_scalars AND
                # _apply_service_area_societal_scalars output together, keyed
                # by (allocation_cache_key, frozen_island_function_map,
                # op_signature). op_signature captures the operational
                # provider set per function -- the missing determinant for
                # _apply_service_area_societal_scalars, which used to be
                # re-run unconditionally on every timestep even on a cache
                # hit. Keying on providers only (never the full operational /
                # island_id arrays) keeps memory independent of total asset
                # count. This cache is function-local: it is freed when
                # postprocess_societal_access_results returns.
                with profiler.section("societal_access.compute_scalars.cache_key"):
                    _frozen_ifm_key = frozenset(frozen_island_function_map.items())
                    _op_signature = tuple(
                        (func, frozenset(ids))
                        for func, ids in sorted(operational_asset_ids_by_function.items())
                    )
                    _scalar_cache_key = (allocation_cache_key, _frozen_ifm_key, _op_signature)
                    _scalar_cache_hit = _scalar_cache_key in _scalar_fields_cache

                if _scalar_cache_hit:
                    with profiler.section("societal_access.compute_scalars.cache_hit"):
                        # A cache hit must never hand out the same mutable dict
                        # twice: _apply_service_area_societal_scalars mutates its
                        # `fields` argument in place, so always return a copy.
                        societal_fields = dict(_scalar_fields_cache[_scalar_cache_key])
                else:
                    with profiler.section("societal_access.compute_scalars.cache_miss"):
                        societal_fields = _compute_societal_scalars(
                            frozen_island_function_map=frozen_island_function_map,
                            allocation_df=allocation_df,
                            pop_grid_gdf=pop_grid_gdf,
                            cell_id_column=cell_id_column,
                            pop_group_columns=pop_group_columns,
                            all_functions=all_functions,
                            reference_group=reference_group,
                            pop_alloc=_cached_entry["pop_alloc"] if _cached_entry else None,
                            island_pop=_cached_entry["island_pop"] if _cached_entry else None,
                            total_pop=_cached_entry["total_pop"] if _cached_entry else None,
                            available_group_cols=_cached_entry["available_group_cols"] if _cached_entry else None,
                        )
                        # --- Fix 4 / Work item 5: use precomputed numpy arrays +
                        # provider→position maps for service area computation.
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

            with profiler.section("societal_access.merge_fields"):
                ts_summary.update(societal_fields)
                ts_summary["allocation_cache_key"] = allocation_cache_key
                ts_summary["allocation_road_state_key"] = road_state_key

    return summary_results, allocation_cache


def _find_allocation_in_cache(
    allocation_cache: Dict[str, pd.DataFrame],
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    road_state_key: str,
    nearest_max_distance: float,
) -> Optional[pd.DataFrame]:
    """Scan *allocation_cache* for an entry matching all metadata fields.

    This is the legacy (backward-compatible) lookup path used when no
    ``islands_gdf_cache`` is provided to
    :func:`postprocess_societal_access_results`.  Returns ``None`` when no
    unique matching entry is found.
    """
    global _LEGACY_ALLOCATION_LOOKUP_WARNED
    if not _LEGACY_ALLOCATION_LOOKUP_WARNED:
        warnings.warn(
            "Using legacy allocation-cache metadata scan path; pass "
            "'islands_gdf_cache' to postprocess_societal_access_results for "
            "deterministic road-state-aware allocation lookup.",
            DeprecationWarning,
            stacklevel=2,
        )
        _LEGACY_ALLOCATION_LOOKUP_WARNED = True

    _, allocation_df = _find_allocation_cache_entry(
        allocation_cache,
        pop_grid_gdf,
        cell_id_column,
        road_state_key,
        nearest_max_distance,
    )
    return allocation_df


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
    """Compute flat societal metric scalars for one timestep.

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


# ---------------------------------------------------------------------------
# Layer D — high-level convenience wrapper
# ---------------------------------------------------------------------------

def analyse_societal_access(
    islands_gdf: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
    service_nodes_gdf: Optional[gpd.GeoDataFrame] = None,
    taxonomy: Optional[Dict[str, str]] = None,
    pop_columns: Optional[Dict[str, str]] = None,
    node_type_column: str = "type",
    island_id_column: str = "island_id",
    reference_group: str = "total",
    graph=None,
    destination_nodes: Optional[Mapping[Any, str]] = None,
    origin_island_ids: Optional[Mapping[Any, int]] = None,
    stakeholder_groups: Optional[Mapping[str, Mapping[Any, float]]] = None,
) -> Dict[str, Any]:
    """Run the full societal access pipeline and return all result tables."""
    use_graph_path = (
        graph is not None
        and destination_nodes is not None
        and stakeholder_groups is not None
    )

    if use_graph_path:
        full_assignment = build_island_assignment(graph)

        if origin_island_ids is None:
            origin_ids: Set[Any] = set()
            for grp in stakeholder_groups.values():
                origin_ids |= set(grp.keys())
            origin_island_ids = {
                oid: full_assignment[oid]
                for oid in origin_ids
                if oid in full_assignment
            }

        island_function_map = build_destination_function_map(
            destination_nodes, full_assignment, taxonomy=taxonomy
        )
        origin_access_df = compute_origin_access(origin_island_ids, island_function_map)
        access_matrix = compute_access_matrix_from_origins(origin_access_df, stakeholder_groups)
        equity_gaps = compute_equity_gaps(access_matrix, reference_group=reference_group)

        return {
            "island_functions": island_function_map,
            "island_population": None,
            "origin_access": origin_access_df,
            "access_matrix": access_matrix,
            "equity_gaps": equity_gaps,
        }

    else:
        island_function_map = assign_destinations_to_islands_spatial(
            service_nodes_gdf, islands_gdf,
            taxonomy=taxonomy, node_type_column=node_type_column,
            island_id_column=island_id_column,
        )
        island_population_df = assign_origins_to_islands_spatial(
            population_gdf, islands_gdf,
            pop_columns=list(pop_columns.values()) if pop_columns else None,
            island_id_column=island_id_column,
        )
        access_matrix = compute_access_matrix(
            island_function_map, island_population_df,
            pop_columns=pop_columns, island_id_column=island_id_column,
        )
        equity_gaps = compute_equity_gaps(access_matrix, reference_group=reference_group)

        return {
            "island_functions": island_function_map,
            "island_population": island_population_df,
            "origin_access": None,
            "access_matrix": access_matrix,
            "equity_gaps": equity_gaps,
        }
