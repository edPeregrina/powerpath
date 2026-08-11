import networkx as nx
from pyproj import Transformer
import shapely.geometry as sg
import pandas as pd
import geopandas as gpd
from scipy.spatial import Voronoi
from rtree import index
from typing import List

def create_spatial_index(gdf):
    """
    Create R-tree spatial index for fast spatial queries.
    
    Arguments:
    - gdf: GeoDataFrame containing geometries to index

    Returns:
    - R-tree spatial index
    """
    idx = index.Index()
    
    # Insert each geometry's bounding box into the index
    for i, row in gdf.iterrows():
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        idx.insert(i, bounds)
    
    return idx


def project_graph_coords(G: nx.Graph, from_crs: str, to_crs: str) -> nx.Graph:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for n, d in G.nodes(data=True):
        d["x_m"], d["y_m"] = transformer.transform(d["x"], d["y"])
    for u, v, d in G.edges(data=True):
        if 'geometry' in d and d['geometry'] is not None:
            # Transform the geometry coordinates
            if hasattr(d['geometry'], 'coords'):
                coords = list(d['geometry'].coords)
                transformed_coords = [transformer.transform(x, y) for x, y in coords]
                transformed_geometry = sg.LineString(transformed_coords)
                d["length"] = transformed_geometry.length
            else:
                # Fallback to straight-line distance if geometry doesn't have coords
                x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
                x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
                d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        else:
            # No geometry available, use straight-line distance between nodes
            x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
            x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
            d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    return G

def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str, 
                        l1_area_geojson=None, verbose=False) -> nx.Graph:
    """
    Filter graph edges based on hazard values, excluding protected infrastructure.
    Applies L1 depth reductions by adjusting threshold per edge (not modifying graph).
    
    Args:
        G: NetworkX graph with hazard values on edges
        threshold: Base hazard value threshold for edge removal
        hazard_column: Name of edge attribute containing hazard values
        l1_area_geojson: Optional path/GeoDataFrame for L1 depth reductions
        verbose: Print progress messages
    
    Returns:
        Filtered graph with hazard edges removed
    """
    import pandas as pd
    import geopandas as gpd
    import shapely
    from pathlib import Path
    
    def is_motorway(highway):
        if isinstance(highway, str):
            return "motorway" in highway.lower()
        elif isinstance(highway, list):
            return any("motorway" in str(h).lower() for h in highway)
        return False

    def is_protected(d):
        """Check if an edge represents protected infrastructure (bridge/tunnel)"""
        def check_attribute(val):
            if val is None:
                return False
            
            if isinstance(val, list):
                return any(
                    item is not None and 
                    str(item).strip().lower() not in ['', 'nan', 'none', 'no'] 
                    for item in val
                )
            elif isinstance(val, str):
                val_clean = val.strip().lower()
                return val_clean not in ['', 'nan', 'none', 'no']
            else:
                return pd.notna(val)

        return check_attribute(d.get("bridge")) or check_attribute(d.get("tunnel")) or check_attribute(d.get("protected"))

    # Get L1 depth reductions per edge (if applicable)
    edge_depth_reductions = {}
    
    if l1_area_geojson is not None:
        l1_gdf = (gpd.read_file(l1_area_geojson) 
                  if isinstance(l1_area_geojson, (str, Path)) 
                  else l1_area_geojson)
        
        if "depth_red" not in l1_gdf.columns:
            l1_gdf['depth_red'] = 0.3
            if verbose:
                print("Warning: L1 GeoJSON missing 'depth_red' column, using default 0.3m")
        
        # Ensure correct CRS (graph is in EPSG:4326)
        if l1_gdf.crs != "EPSG:4326":
            l1_gdf = l1_gdf.to_crs("EPSG:4326")
        
        # Build STRtree from L1 polygons
        l1_tree = shapely.STRtree(l1_gdf.geometry.values)
        
        # For each edge, check if it intersects any L1 polygon
        for u, v, data in G.edges(data=True):
            edge_geom = data.get('geometry')
            
            if edge_geom is None:
                from shapely.geometry import LineString
                edge_geom = LineString([
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ])
            
            # Find intersecting L1 polygons
            intersecting_l1_indices = l1_tree.query(edge_geom, predicate='intersects')
            
            if len(intersecting_l1_indices) > 0:
                # Take maximum depth reduction if multiple polygons overlap
                max_reduction = l1_gdf.iloc[intersecting_l1_indices]['depth_red'].max()
                edge_depth_reductions[(u, v)] = max_reduction
        
        if verbose:
            print(f"Applied L1 to {len(edge_depth_reductions)} edges")
    
    # Filter edges based on adjusted thresholds
    edges_to_remove = []
    
    for u, v, d in G.edges(data=True):
        hazard_value = d.get(hazard_column, 0)
        
        # Get edge-specific depth reduction
        depth_reduction = edge_depth_reductions.get((u, v), 0.0)
        
        # Adjusted threshold: hazard must exceed (base_threshold + reduction)
        adjusted_threshold = threshold + depth_reduction
        
        # Remove edge if hazard exceeds adjusted threshold (and not protected)
        if (hazard_value > adjusted_threshold and 
            not is_motorway(d.get("highway")) and 
            not is_protected(d)):
            edges_to_remove.append((u, v))
    
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if verbose:
        print(f"Removed {len(edges_to_remove)} edges (adjusted thresholds)")
    
    return G

def compile_asset_gdfs(gdf_list: List) -> "gpd.GeoDataFrame":
    """Concatenate multiple per-type asset GeoDataFrames into one unified GeoDataFrame.

    The resulting index is a RangeIndex (0, 1, 2, …) so that each row's positional
    index is also its label.  This is required by :func:`build_voronoi_service_area_map`
    and by :func:`~src.impacts.create_voronoi_for_asset_type`, which both rely on
    the index being the positional identifier of each asset in the combined array.

    Args:
        gdf_list: Ordered list of GeoDataFrames to combine (e.g.
            ``[gdf_substations, gdf_hospitals]``).  All GDFs must share the same
            CRS and must contain a ``'type'`` column.

    Returns:
        A single GeoDataFrame with a reset RangeIndex.

    Example::

        gdf_assets = compile_asset_gdfs([gdf_substations, gdf_hospitals])
        # Rows 0..N-1  → substations
        # Rows N..N+M-1 → hospitals
    """
    if not gdf_list:
        raise ValueError("gdf_list must contain at least one GeoDataFrame.")
    combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    return combined


def build_voronoi_service_area_map(voronoi_gdf: "gpd.GeoDataFrame",
                                    gdf_secondary_assets: "gpd.GeoDataFrame") -> dict:
    """Map each secondary asset to the primary asset whose Voronoi polygon contains it.

    The mapping is built once before the simulation starts so the O(N·M) spatial
    search is performed only once rather than repeated every timestep.

    **Spatial search strategy** (reuses :func:`create_spatial_index`):

    1. Build an R-tree on the Voronoi polygons using bounding-box entries.
    2. For each secondary asset, query the R-tree with the asset's centroid to get
       a small set of candidate Voronoi polygons (bbox filter).
    3. Run an exact geometric ``contains`` / ``intersects`` check against each
       candidate (fine intersection).

    This two-step approach mirrors the pattern already used elsewhere in the codebase
    (e.g. ``assign_impact_metric_to_voronoi``) and keeps the function fast for large
    asset sets.

    Each secondary asset is assigned to **at most one** Voronoi polygon.  Because a
    Voronoi tessellation is a partition of space, each point falls in exactly one
    cell; the loop breaks after the first match is found.  For asset types where
    multiple primary assets could cover the same point (e.g. telecom towers), this
    function can be extended to collect all matches before breaking.

    Args:
        voronoi_gdf: GeoDataFrame of Voronoi polygons.  Must have an ``'asset_id'``
            column whose values are the **positional indices** of the primary assets
            in the combined ``gdf_assets`` array (as produced by
            :func:`~src.impacts.create_voronoi_for_asset_type` when called on a
            combined GDF with a RangeIndex).
        gdf_secondary_assets: GeoDataFrame of secondary assets (e.g. hospitals).
            Must be in the same CRS as *voronoi_gdf*, or will be reprojected.
            The DataFrame **index** must hold the positional indices of these assets
            in the combined ``gdf_assets`` array (i.e. call this function on the
            slice ``gdf_assets[gdf_assets['type'] == 'hospital']`` after compiling
            with :func:`compile_asset_gdfs`).

    Returns:
        dict: ``{primary_pos: [secondary_pos, …]}`` – maps each primary asset's
        positional index to the list of secondary asset positional indices that
        fall within its Voronoi polygon.  Primary assets with no secondary assets
        in their polygon are omitted.

    Example::

        gdf_assets = compile_asset_gdfs([gdf_substations, gdf_hospitals])
        voronoi_gdf = create_voronoi_for_asset_type(gdf_assets, 'msls')
        gdf_hospitals_slice = gdf_assets[gdf_assets['type'] == 'hospital']
        gdf_hospitals_proj = gdf_hospitals_slice.to_crs(voronoi_gdf.crs)
        service_area_map = build_voronoi_service_area_map(voronoi_gdf, gdf_hospitals_proj)
        config['dependency_parameters']['service_area_map'] = service_area_map
    """
    if voronoi_gdf.empty or gdf_secondary_assets.empty:
        return {}

    # Reproject secondary assets to match the Voronoi CRS if needed.
    if gdf_secondary_assets.crs is not None and voronoi_gdf.crs is not None:
        if gdf_secondary_assets.crs != voronoi_gdf.crs:
            gdf_secondary_assets = gdf_secondary_assets.to_crs(voronoi_gdf.crs)

    # Build R-tree spatial index on Voronoi polygons.
    # create_spatial_index inserts using the iterrows() index label as the key.
    # voronoi_gdf has a default RangeIndex, so label == iloc position.
    voronoi_sindex = create_spatial_index(voronoi_gdf)

    # Reset voronoi to a plain list for O(1) positional look-up using the R-tree hits.
    voronoi_records = list(voronoi_gdf.itertuples(index=True, name=None))  # (idx, asset_id, geom)

    service_area_map: dict = {}

    for sec_pos, sec_row in gdf_secondary_assets.iterrows():
        sec_geom = sec_row.geometry
        # Use the centroid for point-in-polygon tests (works for both point and polygon assets).
        point = sec_geom.centroid if hasattr(sec_geom, 'centroid') else sec_geom

        # Step 1 – bounding-box candidates from R-tree.
        candidate_labels = list(voronoi_sindex.intersection(point.bounds))

        # Step 2 – fine geometric intersection against each candidate.
        for vor_label in candidate_labels:
            vor_row = voronoi_gdf.loc[vor_label]
            if vor_row.geometry.contains(point) or vor_row.geometry.intersects(point):
                primary_pos = int(vor_row['asset_id'])
                service_area_map.setdefault(primary_pos, []).append(int(sec_pos))
                break  # Voronoi is a partition – each point belongs to exactly one cell.

    return service_area_map
