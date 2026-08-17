"""Generalised access-to-critical-societal-functions analysis.

This module shifts the unit of analysis from infrastructure assets
(substations, road segments) to **critical societal functions** — such as
health care, emergency response, or education — and measures how access to
those functions is distributed across demographic groups after a disruption.

Conceptual pipeline
-------------------
1. **Service node taxonomy** — map each node *type* to a *function category*
   (e.g. ``"hospital"`` → ``"health"``).
2. **Island function mapping** — after disruption, identify which function
   categories are reachable within each connected component (island) of the
   surviving network.
3. **Population join** — spatially assign each population zone (e.g. CBS
   100 m grid cell) to an island.
4. **Access matrix** — for every ``(function, population group)`` pair,
   compute the percentage of that group that is connected to at least one
   service provider.
5. **Equity gaps** — surface the disparity between the total-population
   access share and each vulnerable sub-group.

The key design principle is that infrastructure (roads, electricity,
pipes) is the *medium* through which access is maintained or lost; the
*function* is the analytical unit.  Adding a new function category
requires only a taxonomy entry — no changes to the graph or metrics logic.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Service Node Taxonomy
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
# 2. Island function mapping
# ---------------------------------------------------------------------------

def compute_function_access_per_island(
    service_nodes_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    taxonomy: Optional[Dict[str, str]] = None,
    node_type_column: str = "type",
    island_id_column: str = "island_id",
    buffer_m: float = 50.0,
) -> Dict[int, FrozenSet[str]]:
    """Return the set of function categories reachable within each island.

    Service nodes are spatially joined to road-network islands (connected
    components).  For each island the function categories of all nodes that
    fall within it are collected and returned as a ``frozenset``.

    Parameters
    ----------
    service_nodes_gdf:
        GeoDataFrame of service-node locations.  Must have a column
        ``node_type_column`` with node type strings.
    islands_gdf:
        GeoDataFrame of road-network islands (output of
        :func:`~src.island_analysis.compute_island_geodataframe_from_graph`).
        Geometries should be buffered polygons; must contain
        ``island_id_column``.
    taxonomy:
        Node-type → function-category mapping.  Defaults to
        :data:`SERVICE_NODE_TAXONOMY`.
    node_type_column:
        Column in *service_nodes_gdf* that holds the node type string.
    island_id_column:
        Column in *islands_gdf* that holds the island identifier.
    buffer_m:
        Additional buffer (metres) applied to service-node centroids before
        the spatial join, so that nodes located just outside a road segment
        buffer are still captured.  Set to 0 to disable.

    Returns
    -------
    dict[int, frozenset[str]]
        ``{island_id: frozenset_of_function_categories}``

    Notes
    -----
    Islands with no service nodes receive an empty ``frozenset``.
    """
    if taxonomy is None:
        taxonomy = SERVICE_NODE_TAXONOMY

    if service_nodes_gdf is None or service_nodes_gdf.empty:
        return {}

    # Ensure matching CRS
    target_crs = islands_gdf.crs or "EPSG:28992"
    service_nodes = service_nodes_gdf.copy().to_crs(target_crs)
    islands = islands_gdf[[island_id_column, "geometry"]].copy()

    # Dissolve island polygons to one polygon per island_id for efficiency
    islands_dissolved = islands.dissolve(by=island_id_column).reset_index()

    # Optionally buffer service node centroids
    if buffer_m > 0:
        service_nodes = service_nodes.copy()
        service_nodes["geometry"] = service_nodes.geometry.centroid.buffer(buffer_m)

    # Spatial join: nodes → islands
    joined = gpd.sjoin(
        service_nodes[[node_type_column, "geometry"]],
        islands_dissolved[[island_id_column, "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Build island → function-category set
    island_functions: Dict[int, set] = {}
    for _, row in joined.iterrows():
        if pd.isna(row.get(island_id_column)):
            continue
        island_id = int(row[island_id_column])
        node_type = row[node_type_column]
        func_cat = taxonomy.get(str(node_type))
        if func_cat is not None:
            island_functions.setdefault(island_id, set()).add(func_cat)

    return {iid: frozenset(cats) for iid, cats in island_functions.items()}


# ---------------------------------------------------------------------------
# 3. Population join
# ---------------------------------------------------------------------------

def join_population_to_islands(
    population_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    pop_columns: Optional[Sequence[str]] = None,
    island_id_column: str = "island_id",
) -> pd.DataFrame:
    """Spatially assign each population zone to an island.

    Each population zone (e.g. CBS 100 m grid cell) is assigned to the
    island whose geometry it overlaps the most (largest intersection area).
    If a zone does not overlap any island it is assigned island ``-1``
    (disconnected / isolated).

    Parameters
    ----------
    population_gdf:
        GeoDataFrame of population zones.  Must contain geometry and the
        columns listed in *pop_columns*.
    islands_gdf:
        GeoDataFrame of road-network islands with buffered polygon geometries
        and an ``island_id_column`` column.
    pop_columns:
        Population count columns to carry forward.  Defaults to the values
        of :data:`POPULATION_GROUP_COLUMNS`.
    island_id_column:
        Column in *islands_gdf* holding the island identifier.

    Returns
    -------
    pd.DataFrame
        One row per population zone; columns include ``island_id`` and all
        *pop_columns*.  Rows with ``island_id == -1`` are zones not
        connected to any road island.
    """
    if pop_columns is None:
        pop_columns = list(POPULATION_GROUP_COLUMNS.values())

    target_crs = islands_gdf.crs or "EPSG:28992"
    pop = population_gdf[list(pop_columns) + ["geometry"]].copy().to_crs(target_crs)
    islands = islands_gdf[[island_id_column, "geometry"]].copy()

    # Dissolve to one polygon per island
    islands_dissolved = islands.dissolve(by=island_id_column).reset_index()

    # Spatial join by largest overlap
    joined = gpd.sjoin_nearest(
        pop,
        islands_dissolved[[island_id_column, "geometry"]],
        how="left",
        max_distance=200,  # metres; zones beyond this are treated as isolated
    )

    # If a zone matched multiple islands (due to boundary overlap), keep
    # the one with the most overlap.
    if joined.index.duplicated().any():
        # Compute intersection area per match
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

    # Fill unmatched zones with island_id = -1
    joined[island_id_column] = joined[island_id_column].fillna(-1).astype(int)

    result = joined[list(pop_columns) + [island_id_column]].copy()
    # Replace suppressed values (CBS uses -99997 / -99998 / -99999) with 0
    for col in pop_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
            result[col] = result[col].where(result[col] >= 0, 0)

    return result


# ---------------------------------------------------------------------------
# 4. Access matrix
# ---------------------------------------------------------------------------

def compute_access_matrix(
    island_function_map: Dict[int, FrozenSet[str]],
    island_population_df: pd.DataFrame,
    pop_columns: Optional[Dict[str, str]] = None,
    island_id_column: str = "island_id",
    all_functions: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute percentage access for every (function, population group) pair.

    For each function category *f* and each population group *g*:

    .. math::

        \\text{access}_{f,g} = \\frac{
            \\sum_{i \\in \\text{islands with } f} \\text{pop}_{i,g}
        }{
            \\sum_{i} \\text{pop}_{i,g}
        } \\times 100

    Parameters
    ----------
    island_function_map:
        Output of :func:`compute_function_access_per_island`.
    island_population_df:
        Output of :func:`join_population_to_islands`.  Must have an
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

    # Collect the universe of functions
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

    # Total population per group (denominator)
    col_names = list(pop_columns.values())
    totals = df[col_names].sum()

    rows = []
    for func in all_functions:
        # Islands that provide this function
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


# ---------------------------------------------------------------------------
# 5. Equity gap analysis
# ---------------------------------------------------------------------------

def compute_equity_gaps(
    access_matrix: pd.DataFrame,
    reference_group: str = "total",
) -> pd.DataFrame:
    """Surface the equity gap between a reference group and all other groups.

    Parameters
    ----------
    access_matrix:
        Output of :func:`compute_access_matrix`.
    reference_group:
        Column label of the reference (baseline) population group.
        Defaults to ``"total"``.

    Returns
    -------
    pd.DataFrame
        Same index as *access_matrix*; columns include:

        * one column per non-reference group showing **absolute gap**
          (``reference_access − group_access``, positive = disadvantaged)
        * ``{group}_relative_gap``: ratio ``group_access / reference_access``
          (1.0 = equal access; < 1.0 = disadvantaged)
        * ``most_disadvantaged_group``: the group with the largest absolute
          gap for each function
        * ``max_absolute_gap``: the value of that largest gap
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
# 6. High-level convenience function
# ---------------------------------------------------------------------------

def analyse_societal_access(
    service_nodes_gdf: gpd.GeoDataFrame,
    islands_gdf: gpd.GeoDataFrame,
    population_gdf: gpd.GeoDataFrame,
    taxonomy: Optional[Dict[str, str]] = None,
    pop_columns: Optional[Dict[str, str]] = None,
    node_type_column: str = "type",
    island_id_column: str = "island_id",
    reference_group: str = "total",
) -> Dict[str, pd.DataFrame]:
    """Run the full societal access pipeline and return all result tables.

    This is the primary entry point for a single disruption scenario.

    Parameters
    ----------
    service_nodes_gdf:
        Locations and types of service nodes.
    islands_gdf:
        Road-network islands (connected components) after disruption.
    population_gdf:
        Population zone grid (e.g. CBS 100 m cells).
    taxonomy:
        Node-type → function-category mapping.
    pop_columns:
        ``{display_label: dataframe_column}`` for demographic groups.
    node_type_column:
        Column in *service_nodes_gdf* with node type strings.
    island_id_column:
        Column in *islands_gdf* with island identifiers.
    reference_group:
        Reference group label for equity gap computation.

    Returns
    -------
    dict with keys:

    ``"island_functions"``
        ``Dict[int, FrozenSet[str]]`` — function categories per island.
    ``"island_population"``
        ``pd.DataFrame`` — population attributes per island.
    ``"access_matrix"``
        ``pd.DataFrame`` — % access per (function, group).
    ``"equity_gaps"``
        ``pd.DataFrame`` — equity gap metrics.
    """
    island_function_map = compute_function_access_per_island(
        service_nodes_gdf,
        islands_gdf,
        taxonomy=taxonomy,
        node_type_column=node_type_column,
        island_id_column=island_id_column,
    )

    island_population_df = join_population_to_islands(
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
        "access_matrix": access_matrix,
        "equity_gaps": equity_gaps,
    }
