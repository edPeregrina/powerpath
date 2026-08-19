import networkx as nx
from pyproj import Transformer
import shapely.geometry as sg
import pandas as pd
from scipy.spatial import Voronoi

def create_spatial_index(gdf):
    """
    Create R-tree spatial index for fast spatial queries.
    
    Arguments:
    - gdf: GeoDataFrame containing geometries to index

    Returns:
    - R-tree spatial index
    """
    from rtree import index

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
                        l1_area_geojson=None, l2_asset_geojson=None,
                        verbose=False) -> nx.Graph:
    """
    Filter graph edges based on hazard values, excluding protected infrastructure.
    Applies active L1/L2 depth reductions by adjusting each edge threshold.
    
    Args:
        G: NetworkX graph with hazard values on edges
        threshold: Base hazard value threshold for edge removal
        hazard_column: Name of edge attribute containing hazard values
        l1_area_geojson: Optional path/GeoDataFrame for L1 depth reductions
        l2_asset_geojson: Optional path/GeoDataFrame for L2 depth reductions
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

    edge_depth_reductions = {}

    def iter_edges_with_keys():
        if G.is_multigraph():
            return G.edges(keys=True, data=True)
        return ((u, v, None, data) for u, v, data in G.edges(data=True))

    def add_adaptation_depth_reductions(adaptation, label, default_reduction):
        if adaptation is None:
            return

        adaptation_gdf = (
            gpd.read_file(adaptation)
            if isinstance(adaptation, (str, Path))
            else adaptation.copy()
        )
        if "depth_red" not in adaptation_gdf.columns:
            adaptation_gdf['depth_red'] = default_reduction
            if verbose:
                print(
                    f"Warning: {label} GeoJSON missing 'depth_red' column, "
                    f"using default {default_reduction}m"
                )

        # Ensure correct CRS (graph is in EPSG:4326)
        if adaptation_gdf.crs != "EPSG:4326":
            adaptation_gdf = adaptation_gdf.to_crs("EPSG:4326")

        adaptation_tree = shapely.STRtree(adaptation_gdf.geometry.values)
        adapted_edge_count = 0
        for u, v, edge_key, data in iter_edges_with_keys():
            edge_geom = data.get('geometry')

            if edge_geom is None:
                from shapely.geometry import LineString
                edge_geom = LineString([
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ])

            intersecting_indices = adaptation_tree.query(
                edge_geom, predicate='intersects'
            )
            if len(intersecting_indices) > 0:
                max_reduction = adaptation_gdf.iloc[
                    intersecting_indices
                ]['depth_red'].max()
                cache_key = (u, v, edge_key)
                edge_depth_reductions[cache_key] = (
                    edge_depth_reductions.get(cache_key, 0.0) + max_reduction
                )
                adapted_edge_count += 1

        if verbose:
            print(f"Applied {label} to {adapted_edge_count} edges")

    add_adaptation_depth_reductions(l1_area_geojson, "L1", 0.3)
    add_adaptation_depth_reductions(l2_asset_geojson, "L2", 0.15)
    
    # Filter edges based on adjusted thresholds
    edges_to_remove = []
    
    for u, v, edge_key, d in iter_edges_with_keys():
        hazard_value = d.get(hazard_column, 0)
        
        # Get edge-specific depth reduction
        depth_reduction = edge_depth_reductions.get((u, v, edge_key), 0.0)
        
        # Adjusted threshold: hazard must exceed (base_threshold + reduction)
        adjusted_threshold = threshold + depth_reduction
        
        # Remove edge if hazard exceeds adjusted threshold (and not protected)
        if (hazard_value > adjusted_threshold and 
            not is_motorway(d.get("highway")) and 
            not is_protected(d)):
            edge = (u, v, edge_key) if G.is_multigraph() else (u, v)
            edges_to_remove.append(edge)
    
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if verbose:
        print(f"Removed {len(edges_to_remove)} edges (adjusted thresholds)")
    
    return G