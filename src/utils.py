import networkx as nx
from pyproj import Transformer
import shapely.geometry as sg
import pandas as pd
from scipy.spatial import Voronoi
from rtree import index

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


def analyze_simulation_performance(gdf_assets, hazard_maps, config, max_maps=3):
    """
    Analyze simulation performance using the current workflow and functions.
    Tracks execution time and memory usage for key steps.
    """
    import time
    import psutil
    import functools

    performance_data = {
        'timing': {},
        'memory': [],
        'function_calls': {}
    }

    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Starting performance analysis (max {max_maps} maps)")
    print(f"Baseline memory: {baseline_memory:.1f} MB")

    # Decorator to time function execution
    def timed_function(func_name, original_func):
        @functools.wraps(original_func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = original_func(*args, **kwargs)
            duration = time.time() - start
            performance_data['function_calls'].setdefault(func_name, []).append(duration)
            return result
        return wrapper

    # Patch key functions for timing
    from src.island_analysis import (
        match_island_ids_assets as orig_match_islands,
        update_repair_crew_islands_with_overlap_cached as orig_island_overlaps,
        initialize_island_cache as orig_initialize_island_cache,
        create_spatial_index as orig_create_spatial_index,
        compute_island_geodataframe_from_graph as orig_compute_island_geodataframe_from_graph,
    )
    from src.hazard_analysis_electricity import (
        find_hazard_value_at_points_optimized as orig_hazard_extract,
        get_valid_mean as orig_get_valid_mean,
        _fallback_rasterio_method as orig_fallback_rasterio_method,
    )
    from src.simulation import (
        update_repair_crew_assignment_optimized as orig_crew_assignment,
        _update_hazard_map_states as orig_update_hazard_map_states,
        # simulate_asset_damage_recovery_access_optimized as orig_simulate_asset_damage_recovery_access_optimized,
        # analyze_simulation_performance as orig_analyze_simulation_performance,
    )
    # import src.grid_based_accessibility_hex as grid_hex

    patched = {
        'match_island_ids_assets': timed_function('match_islands', orig_match_islands),
        'update_repair_crew_islands_with_overlap_cached': timed_function('island_overlaps', orig_island_overlaps),
        'initialize_island_cache': timed_function('initialize_island_cache', orig_initialize_island_cache),
        'create_spatial_index': timed_function('create_spatial_index', orig_create_spatial_index),
        'compute_island_geodataframe_from_graph': timed_function('compute_island_geodataframe_from_graph', orig_compute_island_geodataframe_from_graph),
        'find_hazard_value_at_points_optimized': timed_function('hazard_extract', orig_hazard_extract),
        'get_valid_mean': timed_function('get_valid_mean', orig_get_valid_mean),
        '_fallback_rasterio_method': timed_function('fallback_rasterio_method', orig_fallback_rasterio_method),
        'update_repair_crew_assignment_optimized': timed_function('crew_assignment', orig_crew_assignment),
        '_update_hazard_map_states': timed_function('update_hazard_map_states', orig_update_hazard_map_states),
    }

    # Monkey patch
    import src.island_analysis
    import src.hazard_analysis_electricity
    import src.simulation
    src.island_analysis.match_island_ids_assets = patched['match_island_ids_assets']
    src.island_analysis.update_repair_crew_islands_with_overlap_cached = patched['update_repair_crew_islands_with_overlap_cached']
    src.island_analysis.initialize_island_cache = patched['initialize_island_cache']
    src.island_analysis.create_spatial_index = patched['create_spatial_index']
    src.island_analysis.compute_island_geodataframe_from_graph = patched['compute_island_geodataframe_from_graph']
    src.hazard_analysis_electricity.find_hazard_value_at_points_optimized = patched['find_hazard_value_at_points_optimized']
    src.hazard_analysis_electricity.get_valid_mean = patched['get_valid_mean']
    src.hazard_analysis_electricity._fallback_rasterio_method = patched['_fallback_rasterio_method']
    src.simulation.update_repair_crew_assignment_optimized = patched['update_repair_crew_assignment_optimized']
    src.simulation._update_hazard_map_states = patched['_update_hazard_map_states']

    try:
        start_time = time.time()
        performance_data['memory'].append(process.memory_info().rss / 1024 / 1024)

        # Run simulation with limited maps
        results, final_state = simulate_asset_damage_recovery_access_optimized(
            gdf_assets=gdf_assets,
            hazard_maps=hazard_maps[:max_maps],
            number_repair_crews=config['simulation_config']['number_repair_crews'],
            repair_crew_assignment_method=config['simulation_config']['repair_crew_assignment_method'],
            flood_threshold=config['simulation_config']['flood_threshold'],
            recovery_parameters=config['recovery_parameters'],
            root_dir=config['root_dir'],
            verbose=False,
            timestep_output=True,
            execution_id=None,
            config=config,
            major_timestep=config['simulation_config'].get('major_timestep', 24),
            iterations=10
        )

        total_time = time.time() - start_time
        performance_data['timing']['total'] = total_time
        performance_data['memory'].append(process.memory_info().rss / 1024 / 1024)

        print(f"\nPERFORMANCE REPORT")
        print(f"Total simulation time: {total_time:.2f}s")
        print(f"Memory usage: {performance_data['memory'][0]:.1f} MB → {performance_data['memory'][-1]:.1f} MB (Δ{performance_data['memory'][-1]-performance_data['memory'][0]:+.1f} MB)")
        print(f"Timesteps processed: {len(results)}")

        print(f"\nFUNCTION PERFORMANCE:")
        for func_name, times in performance_data['function_calls'].items():
            if times:
                avg_time = sum(times) / len(times)
                total_func_time = sum(times)
                print(f"  {func_name}:")
                print(f"    Calls: {len(times)}")
                print(f"    Total time: {total_func_time:.3f}s ({total_func_time/total_time*100:.1f}% of simulation)")
                print(f"    Average per call: {avg_time:.3f}s")
                print(f"    Min/Max: {min(times):.3f}s / {max(times):.3f}s")

        print(f'\nPercentage of total time spent in key functions:')
        for func_name, times in performance_data['function_calls'].items():
            if times:
                total_func_time = sum(times)
                percentage = total_func_time / total_time * 100
                print(f"  {func_name}: {percentage:.1f}%")

        print(f"\nBOTTLENECK ANALYSIS:")
        bottlenecks = []
        for func_name, times in performance_data['function_calls'].items():
            if times:
                total_func_time = sum(times)
                percentage = total_func_time / total_time * 100
                if percentage > 10:
                    bottlenecks.append((func_name, percentage, total_func_time))
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        if bottlenecks:
            for func_name, percentage, total_func_time in bottlenecks:
                print(f"  🔴 {func_name}: {percentage:.1f}% ({total_func_time:.2f}s)")
        else:
            print("  ✅ All functions <10% of total time")

        return performance_data, results, total_time

    # finally:
    #     # Restore original functions
    #     src.island_analysis.match_island_ids_assets = orig_match_islands
    #     src.hazard_analysis_electricity.find_hazard_value_at_points_optimized = orig_hazard_extract
    #     src.simulation.update_repair_crew_assignment_optimized = orig_crew_assignment
    #     src.island_analysis.update_repair_crew_islands_with_overlap_cached = orig_island_overlaps
    #     # grid_hex.accessibility_model = orig_grid_accessibility

    finally:
        # Restore original functions
        src.island_analysis.match_island_ids_assets = orig_match_islands
        src.island_analysis.update_repair_crew_islands_with_overlap_cached = orig_island_overlaps
        src.island_analysis.initialize_island_cache = orig_initialize_island_cache
        src.island_analysis.create_spatial_index = orig_create_spatial_index
        src.island_analysis.compute_island_geodataframe_from_graph = orig_compute_island_geodataframe_from_graph
        src.hazard_analysis_electricity.find_hazard_value_at_points_optimized = orig_hazard_extract
        src.hazard_analysis_electricity.get_valid_mean = orig_get_valid_mean
        src.hazard_analysis_electricity._fallback_rasterio_method = orig_fallback_rasterio_method
        src.simulation.update_repair_crew_assignment_optimized = orig_crew_assignment
        src.simulation._update_hazard_map_states = orig_update_hazard_map_states
        # src.simulation.simulate_asset_damage_recovery_access_optimized = orig_simulate_asset_damage_recovery_access_optimized
        # src.simulation.analyze_simulation_performance = orig_analyze_simulation_performance
        # grid_hex.accessibility_model = grid_hex.accessibility_model  # Restore if you have orig_grid_accessibility

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