import networkx as nx
from pyproj import Transformer
import shapely.geometry as sg
import pandas as pd
from scipy.spatial import Voronoi


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


def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str) -> nx.Graph:
    def is_motorway(highway):
        if isinstance(highway, str):
            return "motorway" in highway.lower()
        elif isinstance(highway, list):
            return any("motorway" in str(h).lower() for h in highway)
        return False

    def is_protected(d):
        """Check if an edge represents a protected infrastructure (bridge or tunnel)"""
        def check_attribute(val):
            if val is None:
                return False
            
            if isinstance(val, list):
                # For lists, check if any element indicates protection
                # Look for 'yes' or any non-empty meaningful value
                return any(
                    item is not None and 
                    str(item).strip().lower() not in ['', 'nan', 'none', 'no'] 
                    for item in val
                )
            elif isinstance(val, str):
                # For strings, check if it indicates protection
                val_clean = val.strip().lower()
                return val_clean not in ['', 'nan', 'none', 'no'] and val_clean != ''
            else:
                # For float/numeric, use pandas notna (handles NaN properly)
                return pd.notna(val)
        
        bridge_val = d.get("bridge")
        tunnel_val = d.get("tunnel")
        
        return check_attribute(bridge_val) or check_attribute(tunnel_val)

    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get(hazard_column, 0) > threshold and not is_motorway(d.get("highway")) and not is_protected(d)
    ]

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


# def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str) -> nx.Graph:
#     def is_motorway(highway):
#         if isinstance(highway, str):
#             return "motorway" in highway.lower()
#         elif isinstance(highway, list):
#             return any("motorway" in str(h).lower() for h in highway)
#         return False

#     def is_protected(d):
#         # Keep edge if it has a bridge or tunnel value
#         return pd.notna(d.get("bridge")) or pd.notna(d.get("tunnel"))

#     edges_to_remove = [
#         (u, v) for u, v, d in G.edges(data=True)
#         if d.get(hazard_column, 0) > threshold and not is_motorway(d.get("highway")) and not is_protected(d)
#     ]

#     G.remove_edges_from(edges_to_remove)
#     G.remove_nodes_from(list(nx.isolates(G)))

#     return G

# def voronoi_polygons(gdf, bounding_box):
#     points = gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
#     vor = Voronoi(points)
#     polygons = []
#     for region in vor.regions:
#         if not -1 in region and len(region) > 0:
#             polygon = Polygon([vor.vertices[i] for i in region])
#             clipped_polygon = polygon.intersection(bounding_box)
#             polygons.append(clipped_polygon)
#     return gpd.GeoDataFrame(geometry=polygons)