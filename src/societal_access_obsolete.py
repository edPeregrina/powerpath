"""Discarded societal-access exploration paths (graph-native + spatial + wrapper).

These functions were an earlier implementation of the access-to-critical-
societal-functions model described in Paper 4's set-theoretic framework:
population/service nodes are assigned to graph- or geometry-based "islands",
and access is evaluated as shared island membership. `societal_access.py`
now implements the same conceptual model in a way that is wired into the
production simulation loop; this module is what came before that. 

Why this was discarded
-----------------------
`societal_access.py`'s production path (``postprocess_societal_access_results``
and its allocation-cache / service-area helpers) superseded the code below
because it:

- Works directly from ``simulation.py``'s per-timestep ``detailed_results``
  (``island_id`` + ``operational`` arrays), instead of requiring a NetworkX
  graph (the graph-native path here) or repeating a spatial join on every
  call (the spatial path here).
- Caches the expensive population -> island geometry allocation across
  timesteps *and* across EMA experiments (``get_or_build_allocation``),
  which neither path here did.
- Adds Voronoi-service-area overrides for functions such as electricity.

Kept here for potential future reuse. Do not wire this into ``simulation.py``
without first re-validating it against the allocation-cache invariants
exercised in ``tests/test_societal_access.py``.

Contents:

A — Graph-native: ``build_island_assignment``, ``build_destination_function_map``,
    ``compute_origin_access``, ``compute_access_matrix_from_origins``.
B — Spatial preprocessing: ``assign_destinations_to_islands_spatial``,
    ``assign_origins_to_islands_spatial`` (+ back-compat aliases
    ``compute_function_access_per_island``, ``join_population_to_islands``).
C — Aggregate metrics, shared by A and B: ``compute_access_matrix``,
    ``compute_equity_gaps``.
D — High-level wrapper dispatching to A or B: ``analyse_societal_access``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set

import geopandas as gpd
import numpy as np
import pandas as pd

from src.societal_access import (
    POPULATION_GROUP_COLUMNS,
    SERVICE_NODE_TAXONOMY,
    _find_allocation_cache_entry,
)

# Local warn-once flag for `_find_allocation_in_cache` below. This used to
# live in `societal_access.py`, but that module no longer references it now
# that the only caller of `_find_allocation_in_cache` lives here.
_LEGACY_ALLOCATION_LOOKUP_WARNED = False


# ---------------------------------------------------------------------------
# Layer A — node-level, graph-based
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
