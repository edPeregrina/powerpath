"""Hazard-agnostic road connectivity: hazard-filtered road graph -> islands, and asset-to-road/island matching.
No config, caches or crews; shared by src.island_analysis and single-map mode."""
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import shapely.geometry as sg
from pyproj import Transformer
from shapely.geometry import LineString

from src.utils import create_spatial_index


def project_graph_coords(G: nx.Graph, from_crs: str, to_crs: str) -> nx.Graph:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for n, d in G.nodes(data=True):
        d["x_m"], d["y_m"] = transformer.transform(d["x"], d["y"])
    for u, v, d in G.edges(data=True):
        if 'geometry' in d and d['geometry'] is not None:
            if hasattr(d['geometry'], 'coords'):
                coords = list(d['geometry'].coords)
                transformed_coords = [transformer.transform(x, y) for x, y in coords]
                transformed_geometry = sg.LineString(transformed_coords)
                d["length"] = transformed_geometry.length
                # Cached so per-cache-miss island builds don't re-transform;
                # 'geometry' stays in from_crs (filter_hazard_graph footprint checks assume EPSG:4326).
                d["geometry_m"] = transformed_geometry
            else:
                x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
                x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
                d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        else:
            x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
            x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
            d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    return G


# Keyed by (path, from_crs, to_crs); always returned via .copy() because filter_hazard_graph mutates in place.
_projected_graph_cache = {}


def _load_and_project_graph(graph_pickle_path, from_crs="EPSG:4326", to_crs="EPSG:28992"):
    key = (str(graph_pickle_path), from_crs, to_crs)
    cached = _projected_graph_cache.get(key)
    if cached is None:
        with open(graph_pickle_path, "rb") as f:
            cached = pickle.load(f)
        cached = project_graph_coords(cached, from_crs=from_crs, to_crs=to_crs)
        _projected_graph_cache[key] = cached
    return cached.copy()


def is_motorway(highway):
    if isinstance(highway, str):
        return "motorway" in highway.lower()
    elif isinstance(highway, list):
        return any("motorway" in str(h).lower() for h in highway)
    return False


def _tag_is_set(val):
    """True if an OSM-style tag (str/list/scalar) carries a real value."""
    if val is None:
        return False
    if isinstance(val, list):
        return any(
            item is not None and
            str(item).strip().lower() not in ['', 'nan', 'none', 'no']
            for item in val
        )
    if isinstance(val, str):
        return val.strip().lower() not in ['', 'nan', 'none', 'no']
    return pd.notna(val)


def edge_exempt_mask(edge_data, exempt_motorways=True, exempt_bridges=True,
                     exempt_tunnels=True):
    """Edges never removed by hazard thresholding. Edges tagged ``protected`` are always exempt."""
    return np.array([
        (exempt_motorways and is_motorway(d.get("highway")))
        or (exempt_bridges and _tag_is_set(d.get("bridge")))
        or (exempt_tunnels and _tag_is_set(d.get("tunnel")))
        or _tag_is_set(d.get("protected"))
        for d in edge_data
    ], dtype=bool)


def l1_edge_depth_reduction(edge_geoms, adapt_geoms, depth_red):
    """Per-edge L1 reduction: max ``depth_red`` over intersecting polygons (0 if none)."""
    reduction = np.zeros(len(edge_geoms))
    if len(edge_geoms) == 0 or len(adapt_geoms) == 0:
        return reduction
    tree = shapely.STRtree(adapt_geoms)
    edge_idx, adapt_idx = tree.query(
        np.array(edge_geoms, dtype=object), predicate="intersects"
    )
    np.maximum.at(reduction, edge_idx, depth_red[adapt_idx])
    return reduction


def hazard_removal_mask(hazard, threshold, depth_reduction, exempt):
    """Remove where hazard > threshold + L1 reduction, unless exempt."""
    return (hazard > threshold + depth_reduction) & ~exempt


def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str, 
                        l1_area_geojson=None,
                        event_footprint=None, outside_footprint_policy="permissive",
                        verbose=False, exempt_motorways=True, exempt_bridges=True,
                        exempt_tunnels=True) -> nx.Graph:
    """Remove edges whose hazard exceeds ``threshold`` plus the edge's L1 depth reduction (in place).

    L2 (asset-level) adaptations are deliberately not accepted: they never apply to roads.

    ``outside_footprint_policy``: "permissive" (default) leaves roads outside
    ``event_footprint`` to the normal threshold; "strict" removes roads not
    intersecting the footprint first (no exterior bypass) and requires a footprint.
    ``event_footprint``: path/GeoDataFrame (reprojected to EPSG:4326) or a shapely
    geometry already in the graph CRS.
    ``exempt_*``: True (default) keeps motorways/bridges/tunnels regardless of hazard.
    """
    if outside_footprint_policy not in ("strict", "permissive"):
        raise ValueError(
            f"Invalid outside_footprint_policy: {outside_footprint_policy!r}. "
            "Must be 'strict' or 'permissive'."
        )
    if outside_footprint_policy == "strict" and event_footprint is None:
        raise ValueError(
            "outside_footprint_policy='strict' requires an 'event_footprint' "
            "(none was provided). Provide a footprint geometry/GeoDataFrame/path, "
            "or use outside_footprint_policy='permissive'."
        )

    edges = list(
        G.edges(keys=True, data=True) if G.is_multigraph()
        else ((u, v, None, data) for u, v, data in G.edges(data=True))
    )

    def edge_geometries():
        geoms = []
        for u, v, _, data in edges:
            geom = data.get("geometry")
            if geom is None:
                geom = sg.LineString([
                    (G.nodes[u]["x"], G.nodes[u]["y"]),
                    (G.nodes[v]["x"], G.nodes[v]["y"]),
                ])
            geoms.append(geom)
        return geoms

    depth_reduction = np.zeros(len(edges))
    if l1_area_geojson is not None:
        adaptation_gdf = (
            gpd.read_file(l1_area_geojson)
            if isinstance(l1_area_geojson, (str, Path))
            else l1_area_geojson.copy()
        )
        if "depth_red" not in adaptation_gdf.columns:
            adaptation_gdf['depth_red'] = 0.3
            if verbose:
                print("Warning: L1 GeoJSON missing 'depth_red' column, using default 0.3m")
        if adaptation_gdf.crs != "EPSG:4326":
            adaptation_gdf = adaptation_gdf.to_crs("EPSG:4326")
        depth_reduction = l1_edge_depth_reduction(
            edge_geometries(), adaptation_gdf.geometry.values,
            adaptation_gdf["depth_red"].to_numpy(dtype=float),
        )
        if verbose:
            print(f"Applied L1 to {int((depth_reduction != 0).sum())} edges")

    footprint_geom = None
    if event_footprint is not None:
        if isinstance(event_footprint, (str, Path)):
            footprint_gdf = gpd.read_file(event_footprint)
        elif isinstance(event_footprint, gpd.GeoDataFrame):
            footprint_gdf = event_footprint
        elif isinstance(event_footprint, gpd.GeoSeries):
            footprint_gdf = gpd.GeoDataFrame(geometry=event_footprint)
        else:
            footprint_gdf = None
            footprint_geom = event_footprint

        if footprint_gdf is not None:
            if footprint_gdf.crs is not None and str(footprint_gdf.crs) != "EPSG:4326":
                footprint_gdf = footprint_gdf.to_crs("EPSG:4326")
            footprint_geom = (
                footprint_gdf.union_all()
                if hasattr(footprint_gdf, "union_all")
                else footprint_gdf.unary_union
            )

    footprint_removed = np.zeros(len(edges), dtype=bool)
    if outside_footprint_policy == "strict" and footprint_geom is not None:
        footprint_removed = ~shapely.intersects(
            np.array(edge_geometries(), dtype=object), footprint_geom
        )

    hazard = np.array([d.get(hazard_column, 0) for _, _, _, d in edges], dtype=float)
    exempt = edge_exempt_mask(
        [d for _, _, _, d in edges],
        exempt_motorways=exempt_motorways,
        exempt_bridges=exempt_bridges,
        exempt_tunnels=exempt_tunnels,
    )
    remove = footprint_removed | hazard_removal_mask(hazard, threshold, depth_reduction, exempt)

    edges_to_remove = [
        (u, v, k) if G.is_multigraph() else (u, v)
        for (u, v, k, _), r in zip(edges, remove) if r
    ]
    footprint_edges_removed = int(footprint_removed.sum())
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if verbose:
        print(f"Removed {len(edges_to_remove)} edges (adjusted thresholds)")
        if outside_footprint_policy == "strict" and footprint_geom is not None:
            print(f"  of which {footprint_edges_removed} removed as outside the event footprint (strict policy)")
    
    return G


def label_road_islands(G):
    """Map each node of a hazard-filtered graph to its island id (connected-component enumeration order)."""
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    fid_to_island = {}
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            fid_to_island[u] = i
            fid_to_island[v] = i
    return fid_to_island


def compute_island_geodataframe_from_graph(
    graph_pickle_path: str, 
    hazard_threshold: float, 
    hazard_column: str, 
    buffer_distance: float = 2.5, 
    verbose: bool = False,
    l1_area_geojson=None,
    event_footprint=None,
    outside_footprint_policy="permissive",
    exempt_motorways=True,
    exempt_bridges=True,
    exempt_tunnels=True,
) -> gpd.GeoDataFrame:
    """Buffered road segments with ``island_id``/``island_size_km`` after hazard filtering
    (L1 only; footprint and exemption args are forwarded to :func:`filter_hazard_graph`)."""
    G = _load_and_project_graph(graph_pickle_path, from_crs="EPSG:4326", to_crs="EPSG:28992")

    G = filter_hazard_graph(
        G, hazard_threshold, hazard_column, 
        l1_area_geojson=l1_area_geojson,
        event_footprint=event_footprint,
        outside_footprint_policy=outside_footprint_policy,
        exempt_motorways=exempt_motorways,
        exempt_bridges=exempt_bridges,
        exempt_tunnels=exempt_tunnels,
        verbose=verbose
    )

    fid_to_island = label_road_islands(G)

    # Fallback only for edges lacking 'geometry_m' (graph that bypassed project_graph_coords).
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    records = []

    for u, v, data in G.edges(data=True):
        projected_geom = data.get('geometry_m')
        if projected_geom is None:
            original_geom = data.get('geometry')
            if original_geom is not None and hasattr(original_geom, 'coords'):
                projected_coords = [transformer.transform(x, y) for x, y in original_geom.coords]
                projected_geom = LineString(projected_coords)
            else:
                projected_geom = LineString([(G.nodes[u]["x_m"], G.nodes[u]["y_m"]),
                                           (G.nodes[v]["x_m"], G.nodes[v]["y_m"])])
        
        length_m = data.get("length", None)
        if length_m is None:
            length_m = projected_geom.length

        island_id = fid_to_island.get(v, -1)

        record = data.copy()
        record["geometry"] = projected_geom
        record["length_m"] = length_m
        record["island_id"] = island_id
        records.append(record)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")
    gdf = gdf.drop_duplicates(subset=["geometry", "island_id"])
    
    if verbose:
        print(f"After deduplication: {len(gdf)} road segments")

    original_geometries = gdf.geometry.values
    # Vectorized buffer; per-row fallback only for empty/invalid results.
    buffered_geometries = gdf.geometry.buffer(
        buffer_distance, cap_style='square', join_style='mitre'
    ).values
    bad_mask = np.array([
        geom is None or geom.is_empty or not geom.is_valid for geom in buffered_geometries
    ])
    if bad_mask.any():
        buffered_geometries = buffered_geometries.copy()
        for i in np.flatnonzero(bad_mask):
            original_geom = original_geometries[i]
            try:
                buffered_geom = original_geom.make_valid().buffer(
                    buffer_distance, cap_style='square', join_style='mitre'
                )
            except Exception as e:
                print(f"Warning: Error buffering geometry: {e}, using original")
                buffered_geom = original_geom
            buffered_geometries[i] = buffered_geom
    
    gdf['original_geometry'] = original_geometries
    gdf['geometry'] = buffered_geometries
    
    if verbose:
        print(f"Buffered {len(gdf)} road segments with {buffer_distance}m buffer")

    # Island size from original linestring lengths.
    island_sizes = gdf.groupby("island_id")["length_m"].sum().reset_index()
    island_sizes["island_size_km"] = island_sizes["length_m"] / 1000.0
    island_sizes = island_sizes[["island_id", "island_size_km"]]

    gdf = gdf.merge(island_sizes, on="island_id", how="left")
    gdf["island_size_km"] = gdf["island_size_km"].fillna(0.0)

    if verbose:
        print(f"Island distribution: {gdf['island_id'].value_counts().sort_index().to_dict()}")
    return gdf


def nearest_road_rfids(assets_gdf, islands_gdf, search_radius=100, spatial_idx=None):
    """Series (indexed like ``assets_gdf``) of the rfid nearest each asset centroid
    within ``search_radius`` m, else -1. Both frames must share a projected CRS."""
    if spatial_idx is None:
        spatial_idx = create_spatial_index(islands_gdf)
    centroids = assets_gdf.geometry.centroid

    def find_nearest_road(asset_centroid):
        candidate_idxs = list(spatial_idx.intersection(asset_centroid.buffer(search_radius).bounds))
        if candidate_idxs:
            candidates = islands_gdf.iloc[candidate_idxs]
            if not candidates.empty:
                distances = candidates.geometry.distance(asset_centroid)
                if not distances.empty:
                    nearest_idx = distances.idxmin()
                    return candidates.loc[nearest_idx, 'rfid']
        return -1

    return centroids.apply(find_nearest_road)


def baseline_boundary(islands_gdf, asset_island_ids, main_island_id):
    """Boundary assets (off the main island, or on none) and boundary roads
    (all sections of non-main islands) of an undisrupted network.
    Returns ``(boundary_asset_indices, boundary_islands_rfids)``."""
    boundary_asset_mask = (asset_island_ids != main_island_id) | (asset_island_ids == -1)
    boundary_asset_indices = asset_island_ids[boundary_asset_mask].index.tolist()
    boundary_islands_rfids = islands_gdf[islands_gdf['island_id'] != main_island_id]['rfid'].to_list()
    return boundary_asset_indices, boundary_islands_rfids


def assign_asset_islands(islands_gdf, access_rfids, boundary_asset_indices, main_island_id):
    """Asset island ids on a road state (``islands_gdf`` excludes baseline boundary roads).
    Flooded/absent roads give -1; boundary assets are forced onto ``main_island_id``.
    Returns ``(asset_island_ids, rfids_islands)``."""
    rfid_to_island = dict(zip(islands_gdf['rfid'], islands_gdf['island_id']))
    asset_island_ids = [rfid_to_island.get(rfid, -1) for rfid in access_rfids]
    asset_island_ids = [asset_island_ids[i] if i not in boundary_asset_indices else main_island_id
                        for i in range(len(asset_island_ids))]
    return np.array(asset_island_ids, dtype=int), dict(zip(islands_gdf['rfid'], islands_gdf['island_id']))
