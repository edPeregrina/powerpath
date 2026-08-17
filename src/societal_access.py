"""Generalised access-to-critical-societal-functions analysis.

Conceptual model
----------------
This module operationalises access as **shared island membership** in a
disrupted graph, following the set-theoretic framework of Paper 4:

- ``V``  — the complete set of graph nodes (entities).
- ``G = (V, E)``  — the infrastructure graph before disruption.
- After disruption, unavailable nodes/edges are removed, yielding a
  disrupted graph ``G'``.
- The connected components of ``G'`` form a **partition** of the surviving
  node set: every node belongs to exactly one component (island).
- ``O ⊆ V``  — origins (demand locations: population zones, households, …).
- ``S_f ⊆ V``  — destination set for societal function *f* (hospitals,
  fire stations, schools, …).
- Access rule: origin *o* has access to function *f* iff
  ``island_id[o] == island_id[d]`` for at least one ``d ∈ S_f``.
- ``P_k ⊆ O``  — a stakeholder sub-group (elderly, low-income, rural, …).
- Performance metric for group *k* and function *f*:

  .. math::

      \\text{access}_{f,k} =
          \\frac{|\\{o \\in P_k \\mid \\exists\\,d \\in S_f :
                         \\text{island\\_id}[o] = \\text{island\\_id}[d]\\}|}
               {|P_k|} \\times 100

Infrastructure (roads, electricity, pipes) is the *medium* through which
access is maintained or lost; the societal *function* is the analytical
unit.  Adding a new function requires only a taxonomy entry.

Module structure
----------------
Layer A — **node-level** (graph-based, primary):
    :func:`build_island_assignment`
        Compute the island-id partition from a NetworkX graph after
        disruption.  Returns ``{node_id: island_id}``.
    :func:`build_destination_function_map`
        Map destination nodes to function categories using the taxonomy.
        Returns ``{island_id: frozenset_of_function_categories}`` derived
        from node-level island assignments.
    :func:`compute_origin_access`
        For each origin node decide whether it has access to each function
        (shared island membership).  Returns a boolean DataFrame indexed by
        origin node id.
    :func:`compute_access_matrix_from_origins`
        Aggregate origin-level access flags over stakeholder sub-groups to
        produce the ``(function × group)`` percentage matrix.

Layer B — **spatial preprocessing** (optional helper):
    :func:`assign_origins_to_islands_spatial`
        Spatially assign population grid cells to island ids.  Use this
        when origin entities are geographic zones rather than graph nodes.
    :func:`assign_destinations_to_islands_spatial`
        Spatially assign service-node point locations to island ids.  Use
        this when service nodes are not embedded in the graph as nodes.

Layer C — **equity analysis**:
    :func:`compute_equity_gaps`
        Surface absolute and relative access gaps between a reference
        group and each vulnerable sub-group.

Layer D — **convenience wrapper**:
    :func:`analyse_societal_access`
        High-level pipeline that accepts either graph-node inputs (Layer A)
        or spatial inputs (Layer B) and returns all result tables.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set

import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Service node taxonomy
# ---------------------------------------------------------------------------

#: Default mapping: node *type* string → *function category* label.
#: Extend or override this dict to add new service types without touching
#: the graph or metrics code.
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

#: Default demographic columns in the CBS population grid and their
#: human-readable group labels used in the access matrix output.
#: Values are the CBS column names; keys are the display labels.
POPULATION_GROUP_COLUMNS: Dict[str, str] = {
    "total": "aantal_inwoners",
    "elderly": "aantal_inwoners_65_jaar_en_ouder",
    "children": "aantal_inwoners_0_tot_15_jaar",
    "working_age": "aantal_inwoners_25_tot_45_jaar",
}


# ---------------------------------------------------------------------------
# Layer A — node-level, graph-based (primary path)
# ---------------------------------------------------------------------------

def build_island_assignment(graph) -> Dict[Any, int]:
    """Compute the island partition of a disrupted graph.

    Each connected component of *graph* is interpreted as an island.  Every
    surviving node is assigned to exactly one island; no node is assigned to
    multiple islands (the components form a partition of the node set).

    Parameters
    ----------
    graph:
        A NetworkX ``Graph`` or ``DiGraph`` representing the **disrupted**
        infrastructure network (unavailable nodes/edges already removed by
        the caller).

    Returns
    -------
    dict[node_id, int]
        ``{node_id: island_id}`` for every node in *graph*.
        Island ids are stable integers (0, 1, 2, …) assigned in the order
        that NetworkX returns components.

    Notes
    -----
    - Isolated nodes (no edges) form single-node islands and are included.
    - For directed graphs, **weakly** connected components are used so that
      isolated nodes with only one edge direction are still assigned an
      island.  Use ``nx.strongly_connected_components`` explicitly if the
      analysis requires strong connectivity.
    - The original (pre-disruption) graph is not modified; the caller is
      responsible for constructing *graph* as an independent disrupted copy.
    """
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
    """Map each island to the set of societal functions it contains.

    Parameters
    ----------
    destination_nodes:
        ``{node_id: node_type}`` for all destination/service nodes ``D ⊆ V``.
        Node types are looked up in *taxonomy* to obtain function categories.
    node_island_assignment:
        ``{node_id: island_id}`` as returned by :func:`build_island_assignment`.
    taxonomy:
        Node-type → function-category mapping.  Defaults to
        :data:`SERVICE_NODE_TAXONOMY`.

    Returns
    -------
    dict[int, frozenset[str]]
        ``{island_id: frozenset_of_function_categories}``

    Notes
    -----
    Destination nodes that are not present in *node_island_assignment* (i.e.
    were removed during disruption) are silently skipped — they are not
    reachable and therefore do not contribute to any island's function set.
    """
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
        import warnings
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
    """Evaluate per-origin access for every societal function.

    Access is determined by **shared island membership**: origin *o* has
    access to function *f* iff its island contains at least one destination
    node providing *f*.

    Parameters
    ----------
    origin_island_ids:
        ``{origin_id: island_id}`` for all origin entities ``O ⊆ V``.
        Origins are population zones, grid cells, or demand nodes.
    island_function_map:
        ``{island_id: frozenset_of_function_categories}`` as returned by
        :func:`build_destination_function_map`.
    all_functions:
        Explicit list of function categories to include as columns.  If
        ``None``, all categories present in *island_function_map* are used.

    Returns
    -------
    pd.DataFrame
        Index: origin ids.
        Columns: function category labels.
        Values: ``True`` if the origin has access, ``False`` otherwise.
    """
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
    """Aggregate per-origin access flags into group-level percentage metrics.

    For each stakeholder group *k* and function *f*:

    .. math::

        \\text{access}_{f,k} =
            \\frac{\\sum_{o \\in P_k} w_o \\cdot \\mathbf{1}[\\text{access}_{f,o}]}
                 {\\sum_{o \\in P_k} w_o} \\times 100

    where *w_o* is the weight (e.g. population count) of origin *o*.

    Parameters
    ----------
    origin_access_df:
        Output of :func:`compute_origin_access`.  Index = origin ids;
        columns = function labels; values = bool.
    stakeholder_groups:
        ``{group_label: {origin_id: weight}}`` — for each stakeholder
        group, the origin ids that belong to it and their weights (e.g.
        number of people in that demographic in that grid cell).

    Returns
    -------
    pd.DataFrame
        Index: function category labels.
        Columns: stakeholder group labels.
        Values: percentage with access (0–100), or ``NaN`` if the group
        has no members.
    """
    functions = list(origin_access_df.columns)
    groups = list(stakeholder_groups.keys())

    result = pd.DataFrame(index=pd.Index(functions, name="function"),
                          columns=pd.Index(groups, name="population_group"),
                          dtype=float)

    for group_label, members in stakeholder_groups.items():
        # Only keep origins that exist in origin_access_df
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
        # weighted sum of True (=1) values per function
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
    """Return the set of function categories reachable within each island.

    This is the **spatial preprocessing** variant for use when service
    nodes are geographic point locations not embedded as graph nodes.
    Service nodes are spatially joined to road-network island polygons.

    For the graph-native path, use :func:`build_destination_function_map`
    with node-level island assignments instead.

    Parameters
    ----------
    service_nodes_gdf:
        GeoDataFrame of service-node locations.  Must have a column
        ``node_type_column`` with node type strings.
    islands_gdf:
        GeoDataFrame of road-network islands (connected components after
        disruption), with buffered polygon geometries and
        ``island_id_column``.
    taxonomy:
        Node-type → function-category mapping.  Defaults to
        :data:`SERVICE_NODE_TAXONOMY`.
    node_type_column:
        Column in *service_nodes_gdf* holding the node type string.
    island_id_column:
        Column in *islands_gdf* holding the island identifier.
    buffer_m:
        Additional buffer (metres) applied to service-node centroids
        before the spatial join.  Set to 0 to disable.

    Returns
    -------
    dict[int, frozenset[str]]
        ``{island_id: frozenset_of_function_categories}``
    """
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


# Preserve the original name as an alias for backward compatibility.
compute_function_access_per_island = assign_destinations_to_islands_spatial


def assign_origins_to_islands_spatial(
    population_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    pop_columns: Optional[Sequence[str]] = None,
    island_id_column: str = "island_id",
) -> pd.DataFrame:
    """Spatially assign each population zone to an island.

    This is the **spatial preprocessing** variant for use when origin
    entities are geographic zones (e.g. CBS 100 m grid cells) rather than
    embedded graph nodes.  Each zone is assigned the id of the nearest
    island (within 200 m); zones further than 200 m receive island id
    ``-1`` (isolated/disconnected).

    For the graph-native path, supply ``{origin_id: island_id}`` directly
    from :func:`build_island_assignment`.

    Parameters
    ----------
    population_gdf:
        GeoDataFrame of population zones with geometry and *pop_columns*.
    islands_gdf:
        GeoDataFrame of road-network islands with buffered polygon
        geometries and an ``island_id_column`` column.
    pop_columns:
        Population count columns to carry forward.  Defaults to the values
        of :data:`POPULATION_GROUP_COLUMNS`.
    island_id_column:
        Column in *islands_gdf* holding the island identifier.

    Returns
    -------
    pd.DataFrame
        One row per population zone; columns include ``island_id`` and all
        *pop_columns*.  Rows with ``island_id == -1`` are not connected to
        any road island.
    """
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

    # If a zone matched multiple islands (boundary overlap), keep the
    # one with the largest intersection area.
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
    # Replace suppressed CBS values (-99997 / -99998 / -99999) with 0
    for col in pop_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result[col] = result[col].where(result[col] >= 0, 0)

    return result


# Preserve the original name as an alias for backward compatibility.
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
    """Compute percentage access for every (function, population group) pair.

    This function operates on the output of the **spatial preprocessing
    path** (Layer B).  For the graph-native path use
    :func:`compute_access_matrix_from_origins` instead.

    Access for a population zone is defined by shared island membership:
    the zone's island must contain at least one provider of the function.

    .. math::

        \\text{access}_{f,g} = \\frac{
            \\sum_{z : \\text{island\\_id}[z] \\in \\text{islands}(f)}
                \\text{pop}_{z,g}
        }{
            \\sum_{z} \\text{pop}_{z,g}
        } \\times 100

    Parameters
    ----------
    island_function_map:
        ``{island_id: frozenset_of_function_categories}`` as returned by
        :func:`assign_destinations_to_islands_spatial` or
        :func:`build_destination_function_map`.
    island_population_df:
        Output of :func:`assign_origins_to_islands_spatial`.  Must have an
        ``island_id_column`` column and one column per population group.
    pop_columns:
        ``{display_label: dataframe_column}`` mapping.  Defaults to
        :data:`POPULATION_GROUP_COLUMNS`.
    island_id_column:
        Column in *island_population_df* holding island identifiers.
    all_functions:
        Explicit list of function categories to include as rows.  If
        ``None``, all categories found in *island_function_map* are used.

    Returns
    -------
    pd.DataFrame
        Index: function category labels.
        Columns: population group display labels.
        Values: percentage with access (0–100).
    """
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
        # Access = shared island membership (island_id of zone ∈ islands_with_func)
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
    """Surface the equity gap between a reference group and all other groups.

    Parameters
    ----------
    access_matrix:
        Output of :func:`compute_access_matrix` or
        :func:`compute_access_matrix_from_origins`.
    reference_group:
        Column label of the reference (baseline) population group.
        Defaults to ``"total"``.

    Returns
    -------
    pd.DataFrame
        Same index as *access_matrix*; columns include:

        * ``{group}_absolute_gap`` — ``reference_access − group_access``
          (positive value means the group is disadvantaged).
        * ``{group}_relative_gap`` — ratio ``group_access / reference_access``
          (1.0 = equal access; < 1.0 = disadvantaged).
        * ``most_disadvantaged_group`` — the group with the largest absolute
          gap for each function.
        * ``max_absolute_gap`` — the value of that largest gap.
    """
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
    # Graph-native inputs (Layer A) — take priority when provided
    graph=None,
    destination_nodes: Optional[Mapping[Any, str]] = None,
    origin_island_ids: Optional[Mapping[Any, int]] = None,
    stakeholder_groups: Optional[Mapping[str, Mapping[Any, float]]] = None,
) -> Dict[str, Any]:
    """Run the full societal access pipeline and return all result tables.

    Two input paths are supported:

    **Graph-native path** (conceptually correct, preferred):
        Supply *graph*, *destination_nodes*, *origin_island_ids*, and
        *stakeholder_groups*.  Access is computed directly from node-level
        island assignments (``island_id[o] == island_id[d]``).

    **Spatial path** (preprocessing helper):
        Supply *service_nodes_gdf*, *islands_gdf*, and *population_gdf*.
        Origins and destinations are assigned to islands by spatial join.

    Parameters
    ----------
    islands_gdf:
        GeoDataFrame of road-network island polygons (always required for
        the spatial path; ignored when all graph-native inputs are given).
    population_gdf:
        Population zone GeoDataFrame (spatial path only).
    service_nodes_gdf:
        Service-node point GeoDataFrame (spatial path only).
    taxonomy:
        Node-type → function-category mapping.
    pop_columns:
        ``{display_label: dataframe_column}`` for demographic groups
        (spatial path only).
    node_type_column:
        Column in *service_nodes_gdf* with node type strings.
    island_id_column:
        Column in *islands_gdf* / *population_gdf* with island ids.
    reference_group:
        Reference group label for equity gap computation.
    graph:
        Disrupted NetworkX graph (graph-native path).
    destination_nodes:
        ``{node_id: node_type}`` for service/destination nodes
        (graph-native path).
    origin_island_ids:
        ``{origin_id: island_id}`` for origin entities
        (graph-native path; if omitted, derived from *graph*).
    stakeholder_groups:
        ``{group_label: {origin_id: weight}}`` (graph-native path).

    Returns
    -------
    dict with keys:

    ``"island_functions"``
        ``Dict[int, FrozenSet[str]]`` — function categories per island.
    ``"island_population"``
        ``pd.DataFrame`` — population/origin attributes per island
        (spatial path) or ``None`` (graph-native path).
    ``"access_matrix"``
        ``pd.DataFrame`` — % access per (function, group).
    ``"equity_gaps"``
        ``pd.DataFrame`` — equity gap metrics.
    ``"origin_access"``
        ``pd.DataFrame`` — per-origin boolean access flags
        (graph-native path only, else ``None``).
    """
    use_graph_path = (
        graph is not None
        and destination_nodes is not None
        and stakeholder_groups is not None
    )

    if use_graph_path:
        # --- Graph-native path ---
        full_assignment = build_island_assignment(graph)

        if origin_island_ids is None:
            # Derive origin assignments from the full graph assignment
            origin_ids = set(stakeholder_groups[next(iter(stakeholder_groups))].keys())
            for grp in stakeholder_groups.values():
                origin_ids |= set(grp.keys())
            origin_island_ids = {
                oid: full_assignment[oid]
                for oid in origin_ids
                if oid in full_assignment
            }

        island_function_map = build_destination_function_map(
            destination_nodes,
            full_assignment,
            taxonomy=taxonomy,
        )

        origin_access_df = compute_origin_access(
            origin_island_ids,
            island_function_map,
        )

        access_matrix = compute_access_matrix_from_origins(
            origin_access_df,
            stakeholder_groups,
        )

        equity_gaps = compute_equity_gaps(access_matrix, reference_group=reference_group)

        return {
            "island_functions": island_function_map,
            "island_population": None,
            "origin_access": origin_access_df,
            "access_matrix": access_matrix,
            "equity_gaps": equity_gaps,
        }

    else:
        # --- Spatial path ---
        island_function_map = assign_destinations_to_islands_spatial(
            service_nodes_gdf,
            islands_gdf,
            taxonomy=taxonomy,
            node_type_column=node_type_column,
            island_id_column=island_id_column,
        )

        island_population_df = assign_origins_to_islands_spatial(
            population_gdf,
            islands_gdf,
            pop_columns=list(pop_columns.values()) if pop_columns else None,
            island_id_column=island_id_column,
        )

        access_matrix = compute_access_matrix(
            island_function_map,
            island_population_df,
            pop_columns=pop_columns,
            island_id_column=island_id_column,
        )

        equity_gaps = compute_equity_gaps(access_matrix, reference_group=reference_group)

        return {
            "island_functions": island_function_map,
            "island_population": island_population_df,
            "origin_access": None,
            "access_matrix": access_matrix,
            "equity_gaps": equity_gaps,
        }
