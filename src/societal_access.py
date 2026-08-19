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
) -> str:
    """Build a deterministic cache key for the geometry-only allocation.

    The key covers all deterministic inputs that affect the road partition or
    grid geometry.  It does **not** include population attribute values,
    selected demographic columns, stochastic seed, or provider outcomes.
    """
    key_dict: Dict[str, Any] = {
        "pop_grid_hash": _geometry_hash(pop_grid_gdf),
        "cell_id_column": cell_id_column,
        "cell_id_hash": _column_hash(pop_grid_gdf, cell_id_column),
        "islands_hash": _geometry_hash(islands_gdf),
        "island_id_column": island_id_column,
        "island_id_hash": _column_hash(islands_gdf, island_id_column),
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

    for idx, cell_row in pop.iterrows():
        cell_id = cell_row[cell_id_column]
        cell_geom = cell_row["geometry"]

        # Step 1–3: intersection-based allocation
        overlap_areas: Dict[int, float] = {}
        for _, isl_row in islands_dissolved.iterrows():
            isl_id = int(isl_row[island_id_column])
            isl_geom = isl_row["geometry"]
            if not cell_geom.intersects(isl_geom):
                continue
            inter = cell_geom.intersection(isl_geom)
            area = inter.area
            if area > 0:
                overlap_areas[isl_id] = overlap_areas.get(isl_id, 0.0) + area

        if overlap_areas:
            total_area = sum(overlap_areas.values())
            for isl_id, area in overlap_areas.items():
                rows.append({
                    cell_id_column: cell_id,
                    "island_id": isl_id,
                    "allocation_fraction": area / total_area,
                    "allocation_method": "intersection",
                    "road_state_key": road_state_key,
                    "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
                })
            continue

        # Step 4: nearest-island fallback
        cell_centroid = cell_geom.centroid
        min_dist = float("inf")
        nearest_id: Optional[int] = None
        for _, isl_row in islands_dissolved.iterrows():
            isl_id = int(isl_row[island_id_column])
            dist = cell_centroid.distance(isl_row["geometry"])
            if dist < min_dist:
                min_dist = dist
                nearest_id = isl_id

        if nearest_id is not None and min_dist <= nearest_max_distance:
            rows.append({
                cell_id_column: cell_id,
                "island_id": nearest_id,
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
    allocation_df.attrs.update({
        "population_grid_hash": _geometry_hash(pop_grid_gdf),
        "cell_id_column": cell_id_column,
        "cell_id_hash": _column_hash(pop_grid_gdf, cell_id_column),
        "islands_hash": _geometry_hash(islands_gdf),
        "island_id_column": island_id_column,
        "island_id_hash": _column_hash(islands_gdf, island_id_column),
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

    merged = allocation_df.copy().merge(pop_attrs, on=cell_id_column, how="left")

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
) -> Tuple[pd.DataFrame, str, bool]:
    """Return a cached geometry allocation or build and cache a new one.

    Parameters
    ----------
    allocation_cache:
        In-memory dict keyed by allocation cache key → allocation DataFrame.
    pop_grid_gdf, cell_id_column, islands_gdf, island_id_column,
    nearest_max_distance, road_state_key:
        Forwarded to :func:`build_origin_island_allocations`.

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
    )
    allocation_cache[cache_key] = allocation_df
    return allocation_df, cache_key, True


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

    Returns
    -------
    (updated_summary_results, updated_allocation_cache)
        *summary_results* is returned with societal fields merged in-place.
        *updated_allocation_cache* contains any newly built allocations.
    """
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

    # Determine which functions to always emit
    if all_functions is None:
        all_functions = sorted(set(taxonomy.values()))

    # Build a lookup: summary_results keyed by timestep
    summary_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in summary_results}

    # Group detailed results by timestep
    detailed_by_ts: Dict[int, Dict[str, Any]] = {d["timestep"]: d for d in detailed_results}

    # Process each timestep that has detailed output
    for ts_idx, ts_summary in enumerate(summary_results):
        ts = ts_summary["timestep"]
        ts_detail = detailed_by_ts.get(ts)

        if ts_detail is None:
            # No detailed data → emit zeros
            _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
            continue

        # Read dependency-adjusted operational states
        operational = np.asarray(ts_detail.get("operational", []))
        island_ids = np.asarray(ts_detail.get("island_id", []))

        if len(operational) == 0 or len(island_ids) == 0:
            _merge_zero_societal_metrics(ts_summary, all_functions, pop_group_columns, reference_group)
            continue

        # Build island → functions map from operational providers
        island_function_map: Dict[int, Set[str]] = {}
        for i, (is_op, isl_id) in enumerate(zip(operational, island_ids)):
            if not is_op:
                continue
            atype = asset_types[i] if i < len(asset_types) else "unknown"
            func_cat = taxonomy.get(str(atype))
            if func_cat is not None:
                isl_id_int = int(isl_id)
                island_function_map.setdefault(isl_id_int, set()).add(func_cat)

        frozen_island_function_map: Dict[int, FrozenSet[str]] = {
            iid: frozenset(cats) for iid, cats in island_function_map.items()
        }

        # Retrieve/build geometry-only allocation
        road_state_key = str(
            ts_detail.get("road_state_key") or ts_detail.get("map", ts)
        )

        # We need an islands_gdf to compute the allocation.
        # Build a minimal one from the unique island IDs seen in detailed results.
        # If no island geometry is available, fall back to island-ID-only path.
        unique_island_ids = np.unique(island_ids[island_ids >= 0])

        # Build population-weighted access using allocation approach.
        # We use the simplified path: assign each pop cell to one island based on
        # the island IDs from the asset detail, building a synthetic islands gdf
        # from pop_grid centroids is complex; use a direct numeric path instead.

        # Direct path: for each pop cell, determine which island(s) it belongs to
        # using stored allocation or build from scratch.
        # Since we may not have island geometries here, use a simplified approach:
        # treat each cell as fully assigned to one island via nearest provider logic.
        # The full spatial allocation requires islands_gdf; callers should use
        # the `societal_access_config` dict which carries the needed data.
        # Here we fall back to a simplified scalar computation per island.

        # Compute pop metrics per island from pop_grid
        societal_fields = _compute_societal_scalars(
            frozen_island_function_map=frozen_island_function_map,
            pop_grid_gdf=pop_grid_gdf,
            cell_id_column=cell_id_column,
            pop_group_columns=pop_group_columns,
            all_functions=all_functions,
            reference_group=reference_group,
            allocation_cache=allocation_cache,
            road_state_key=road_state_key,
            nearest_max_distance=nearest_max_distance,
        )

        ts_summary.update(societal_fields)

    return summary_results, allocation_cache


def _compute_societal_scalars(
    frozen_island_function_map: Dict[int, FrozenSet[str]],
    pop_grid_gdf: gpd.GeoDataFrame,
    cell_id_column: str,
    pop_group_columns: Dict[str, str],
    all_functions: List[str],
    reference_group: str,
    allocation_cache: Dict[str, pd.DataFrame],
    road_state_key: str,
    nearest_max_distance: float,
) -> Dict[str, float]:
    """Compute flat societal metric scalars for one timestep.

    Uses allocation fractions from *allocation_cache* if available; otherwise
    treats each population cell as fully allocated to a single island using a
    pre-built (island_id → weighted_population) mapping derived from cached
    allocations.

    Returns a dict of flat metric names → scalar values.
    """
    fields: Dict[str, float] = {}

    expected_metadata = {
        "population_grid_hash": _geometry_hash(pop_grid_gdf),
        "cell_id_column": cell_id_column,
        "cell_id_hash": _column_hash(pop_grid_gdf, cell_id_column),
        "nearest_max_distance": nearest_max_distance,
        "road_state_key": road_state_key,
        "allocation_algorithm_version": ALLOCATION_ALGORITHM_VERSION,
    }

    # Try to find an allocation matching both the road state and grid inputs.
    matching_allocations: list[pd.DataFrame] = []
    for alloc_df in allocation_cache.values():
        if all(
            alloc_df.attrs.get(name) == value
            for name, value in expected_metadata.items()
        ):
            matching_allocations.append(alloc_df)

    matching_alloc = (
        matching_allocations[0] if len(matching_allocations) == 1 else None
    )

    if matching_alloc is None:
        # No allocation available: emit NaN
        for func in all_functions:
            for group in pop_group_columns:
                fields[f"societal_access_pct__{func}__{group}"] = float("nan")
                fields[f"societal_access_population__{func}__{group}"] = float("nan")
                fields[f"societal_no_access_population__{func}__{group}"] = float("nan")
            for group in pop_group_columns:
                fields[f"societal_total_population__{group}"] = float("nan")
        return fields

    # Apply population to allocations
    try:
        pop_alloc = apply_population_to_allocations(
            allocation_df=matching_alloc,
            pop_grid_gdf=pop_grid_gdf,
            cell_id_column=cell_id_column,
            pop_group_columns=pop_group_columns,
        )
    except Exception:
        for func in all_functions:
            for group in pop_group_columns:
                fields[f"societal_access_pct__{func}__{group}"] = float("nan")
                fields[f"societal_access_population__{func}__{group}"] = float("nan")
                fields[f"societal_no_access_population__{func}__{group}"] = float("nan")
            for group in pop_group_columns:
                fields[f"societal_total_population__{group}"] = float("nan")
        return fields

    # Aggregate weighted population per island
    group_cols = {label: f"{label}_weighted" for label in pop_group_columns}
    available_group_cols = {
        label: col for label, col in group_cols.items()
        if col in pop_alloc.columns
    }

    # Total population per group (all islands, including -1)
    total_pop: Dict[str, float] = {}
    for label, col in available_group_cols.items():
        total_pop[label] = float(pop_alloc[col].sum())

    # Population with access per (function, group)
    for func in all_functions:
        islands_with_func = {
            iid for iid, cats in frozen_island_function_map.items()
            if func in cats and iid != -1
        }
        accessible_mask = pop_alloc["island_id"].isin(islands_with_func)

        for label, col in available_group_cols.items():
            total = total_pop.get(label, 0.0)
            with_access = float(pop_alloc.loc[accessible_mask, col].sum())
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
