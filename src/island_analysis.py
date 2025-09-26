
from rtree import index
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from datetime import datetime
import numpy as np
import networkx as nx
from shapely.geometry import LineString
import geopandas as gpd
import pandas as pd
import pickle
from pyproj import Transformer

import time

from src.utils import project_graph_coords, filter_hazard_graph
from src.caching import load_island_cache, save_island_cache_silent, save_island_cache, create_overlap_cache_key, save_overlap_cache

# Import hazard extraction method from config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config


def initialize_island_cache(interim_dir, hazard_dir):
    """
    Initialize island assignment cache, sets up the cache structure for on-demand computation.
    """
    
    cache_dir = interim_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing cache (don't overwrite existing computations)
    island_cache = load_island_cache(cache_dir, hazard_dir)
    
    return island_cache

def create_spatial_index(gdf):
    """
    Create R-tree spatial index for fast spatial queries
    
    Arguments:
    - gdf: GeoDataFrame containing geometries to index

    Returns:
    - R-tree spatial index
    """
    # Build R-tree index
    idx = index.Index()
    
    # Insert each geometry's bounding box into the index
    for i, row in gdf.iterrows():
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        idx.insert(i, bounds, obj=row)
    
    return idx

# def _optimized_overlap_calculation(current_islands, previous_islands, buffer_distance=1):
#     """
#     Geometric intersection computation with R-tree spatial indexing and bounds checking
#     """
#     # Pre-buffer all geometries once
#     current_buffered = current_islands.copy()
#     current_buffered['geometry'] = current_islands.geometry.buffer(buffer_distance, cap_style='square', join_style='mitre') 
    
#     previous_buffered = previous_islands.copy()  
#     previous_buffered['geometry'] = previous_islands.geometry.buffer(buffer_distance, cap_style='square', join_style='mitre')#TODO: Add union to merge
    
#     spatial_idx = create_spatial_index(current_buffered)
    
#     overlaps_by_prev_island = {}
    
#     for _, prev_island in previous_buffered.iterrows():
#         prev_geom = prev_island.geometry
#         prev_area = prev_geom.area  
#         prev_bounds = prev_geom.bounds  
        
#         overlaps = {}
        
#         # Use R-tree to get candidates (prevents checking all)
#         try:
#             # Get candidate islands from spatial index
#             candidate_indices = list(spatial_idx.intersection(prev_bounds))
            
#             # If no spatial candidates found, skip this island
#             if not candidate_indices:
#                 overlaps_by_prev_island[prev_island['island_id']] = overlaps
#                 continue
                
#         except Exception as e:
#             # Fallback to all islands if R-tree fails
#             print(f"Warning: R-tree query failed, falling back to full scan: {e}")
#             candidate_indices = list(current_buffered.index)
        
#         # Process only candidate islands
#         for candidate_idx in candidate_indices:
#             try:
#                 current_island = current_buffered.iloc[candidate_idx]
#                 current_geom = current_island.geometry
#                 current_bounds = current_geom.bounds
                
#                 # Check if within bounds
#                 if not (prev_bounds[2] >= current_bounds[0] and  # prev_maxx >= curr_minx
#                         prev_bounds[0] <= current_bounds[2] and  # prev_minx <= curr_maxx
#                         prev_bounds[3] >= current_bounds[1] and  # prev_maxy >= curr_miny
#                         prev_bounds[1] <= current_bounds[3]):    # prev_miny <= curr_maxy
#                     continue
                    
#                 # Geometric intersection check
#                 if prev_geom.intersects(current_geom):  # Boolean check first
#                     intersection = prev_geom.intersection(current_geom)
#                     if not intersection.is_empty:
#                         overlap_pct = (intersection.area / prev_area)
#                         overlaps[current_island['island_id']] = overlap_pct
                        
#             except Exception:
#                 continue
        
#         overlaps_by_prev_island[prev_island['island_id']] = overlaps
    
#     return overlaps_by_prev_island

def _optimized_overlap_calculation(current_islands, previous_islands, buffer_distance=1):
    """
    Geometric intersection computation with R-tree spatial indexing and vectorized boolean masks.
    """
    # Pre-buffer all geometries once
    current_buffered = current_islands.copy()
    current_buffered['geometry'] = current_islands.geometry.buffer(buffer_distance, cap_style='square', join_style='mitre')
    current_buffered = current_buffered.reset_index(drop=True) 
    previous_buffered = previous_islands.copy()  
    previous_buffered['geometry'] = previous_islands.geometry.buffer(buffer_distance, cap_style='square', join_style='mitre')

    spatial_idx = create_spatial_index(current_buffered)
    overlaps_by_prev_island = {}

    # Convert geometry column to array for fast access
    current_geoms = current_buffered.geometry.values
    current_island_ids = current_buffered['island_id'].values

    for _, prev_island in previous_buffered.iterrows():
        prev_geom = prev_island.geometry
        prev_area = prev_geom.area
        prev_bounds = prev_geom.bounds

        overlaps = {}

        # R-tree bounding box filter
        candidate_indices = list(spatial_idx.intersection(prev_bounds))
        if not candidate_indices:
            overlaps_by_prev_island[prev_island['island_id']] = overlaps
            continue

        # Vectorized intersects filter
        candidate_geoms = current_geoms[candidate_indices]
        candidate_ids = current_island_ids[candidate_indices]
        intersects_mask = np.array([prev_geom.intersects(g) for g in candidate_geoms])

        # Only process true intersections
        if np.any(intersects_mask):
            intersecting_geoms = candidate_geoms[intersects_mask]
            intersecting_ids = candidate_ids[intersects_mask]
            intersections = [prev_geom.intersection(g) for g in intersecting_geoms]
            for island_id, intersection in zip(intersecting_ids, intersections):
                if not intersection.is_empty:
                    overlap_pct = intersection.area / prev_area
                    overlaps[island_id] = overlap_pct

        overlaps_by_prev_island[prev_island['island_id']] = overlaps

    return overlaps_by_prev_island


def update_repair_crew_islands_with_overlap_cached(
    available_repair_crews, island_ids, dissolved_roads, 
    previous_dissolved_roads=None, buffer_distance=1,
    current_map=None, previous_map=None, hazard_threshold=None,
    overlap_cache=None, hazard_dir=None, _config=None, verbose=False
):
    """
    Update repair crew distribution with cached overlap percentages.
    """
    present_islands = dissolved_roads.copy()
    unique_islands = np.unique(island_ids)
    unique_islands = unique_islands[~pd.isna(unique_islands)]

    if verbose: print(f"Updating repair crew distribution for {len(unique_islands)} islands.")

    if isinstance(available_repair_crews, int):
        # Initial distribution logic when crews are given as an integer
        if verbose: print(f"Initial distribution of {available_repair_crews} crews across {len(unique_islands)} islands")
        
        if len(unique_islands) == 0:
            return {}
        
        island_sizes = []
        for island in unique_islands:
            island_mask = island_ids == island
            asset_count = np.sum(island_mask)
            island_sizes.append(asset_count)
        
        total_assets = sum(island_sizes)
        
        if total_assets > 0:
            probabilities = np.array(island_sizes) / total_assets
            assigned_crews = np.random.choice(
                unique_islands, size=available_repair_crews, 
                p=probabilities, replace=True
            )
            
            unique_assigned, crew_counts = np.unique(assigned_crews, return_counts=True)
            available_repair_crews_by_island = {island: 0 for island in unique_islands}
            for island, count in zip(unique_assigned, crew_counts):
                available_repair_crews_by_island[island] = count
        
        total_distributed = sum(available_repair_crews_by_island.values())
        if total_distributed != available_repair_crews:
            print(f"Warning: Crew distribution mismatch. Input: {available_repair_crews}, Distributed: {total_distributed}")
                
        if verbose: print(f"Initial crew distribution: {[(island, crews) for island, crews in available_repair_crews_by_island.items() if crews > 0]}")
        return available_repair_crews_by_island
    
    elif isinstance(available_repair_crews, dict) and previous_dissolved_roads is not None:
        if verbose: print("Performing overlap-based crew redistribution with caching...")
        
        input_total_crews = sum(available_repair_crews.values())
        
        # Check cache first
        overlap_cache_key = None
        cached_overlaps = None
        
        if (overlap_cache is not None and current_map is not None and 
            previous_map is not None and hazard_threshold is not None):
            
            overlap_cache_key = create_overlap_cache_key(previous_map, current_map, hazard_threshold, hazard_dir)
            cached_overlaps = overlap_cache.get(overlap_cache_key)
            
            if cached_overlaps is not None:
                if verbose: print(f"Using cached overlaps for {overlap_cache_key}")
                overlaps_by_prev_island = cached_overlaps
            else:
                if verbose: print(f"Computing overlaps for {overlap_cache_key} (cache miss)")
        
        # Compute overlaps if not cached
        if cached_overlaps is None:
            current_islands = present_islands.copy()
            previous_islands = previous_dissolved_roads.copy()
            
            # Handle CRS
            if current_islands.crs is None:
                current_islands = current_islands.set_crs('EPSG:28992')
            if previous_islands.crs is None:
                previous_islands = previous_islands.set_crs('EPSG:28992')
                
            current_islands = current_islands.to_crs('EPSG:28992')
            previous_islands = previous_islands.to_crs('EPSG:28992')
            
            overlaps_by_prev_island = _optimized_overlap_calculation(
                current_islands, previous_islands, buffer_distance
            )
            print("Overlap computation complete.")
            # Cache the result (only store percentages)
            if overlap_cache is not None and overlap_cache_key is not None:
                overlap_cache[overlap_cache_key] = overlaps_by_prev_island
                cache_dir = _config['interim_dir']
                save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                print(f"Cached overlaps for {overlap_cache_key}")
        
        new_crew_distribution = {island: 0 for island in present_islands['island_id']}
        total_redistributed_crews = 0
        
        for prev_island_id, crew_count in available_repair_crews.items():
                if crew_count <= 0:
                    continue
                    
                overlaps = overlaps_by_prev_island.get(prev_island_id, {})
                
                if not overlaps:
                    if verbose: print(f"No overlaps found for previous island {prev_island_id}, assigning to nearest current island")
                    
                    if not present_islands.empty:
                        if previous_dissolved_roads is not None:
                            prev_island_geom = previous_dissolved_roads[previous_dissolved_roads['island_id'] == prev_island_id]
                            if not prev_island_geom.empty:
                                # Handle CRS for distance calculation
                                prev_islands_crs = prev_island_geom.copy()
                                current_islands_crs = present_islands.copy()
                                
                                if prev_islands_crs.crs is None:
                                    prev_islands_crs = prev_islands_crs.set_crs('EPSG:28992')
                                if current_islands_crs.crs is None:
                                    current_islands_crs = current_islands_crs.set_crs('EPSG:28992')
                                    
                                prev_islands_crs = prev_islands_crs.to_crs('EPSG:28992')
                                current_islands_crs = current_islands_crs.to_crs('EPSG:28992')
                                
                                prev_centroid = prev_islands_crs.geometry.iloc[0].centroid
                                distances = current_islands_crs.geometry.centroid.distance(prev_centroid)
                                nearest_island_idx = distances.idxmin()
                                nearest_island_id = current_islands_crs.loc[nearest_island_idx, 'island_id']
                            else:
                                nearest_island_id = present_islands.iloc[0]['island_id']
                        else:
                            nearest_island_id = present_islands.iloc[0]['island_id']
                    
                    new_crew_distribution[nearest_island_id] += crew_count
                    total_redistributed_crews += crew_count
                    if verbose: print(f"Assigned {crew_count} crews to nearest island {nearest_island_id}")
                    continue
                
                total_overlap_pct = sum(overlaps.values())
                
                if total_overlap_pct > 0:
                    overlap_proportions = []
                    overlap_island_ids = []
                    
                    for island_id, overlap_pct in overlaps.items():
                        overlap_proportions.append(overlap_pct / total_overlap_pct)
                        overlap_island_ids.append(island_id)
                    
                    if len(overlap_island_ids) > 0:
                        assigned_crews = np.random.choice(
                            overlap_island_ids, size=crew_count,
                            p=overlap_proportions, replace=True
                        )
                        
                        unique_assigned, crew_counts = np.unique(assigned_crews, return_counts=True)
                        crews_distributed_this_island = 0
                        
                        for island_id, count in zip(unique_assigned, crew_counts):
                            new_crew_distribution[island_id] += count
                            crews_distributed_this_island += count
                        
                        total_redistributed_crews += crews_distributed_this_island
                        if verbose: 
                            print(f"Redistributed {crew_count} crews from previous island {prev_island_id} based on cached overlaps to:")
                            print([(island, crews) for (island, crews) in new_crew_distribution.items() if crews > 0])

        if total_redistributed_crews != input_total_crews:
            if verbose: 
                print(f"Crew redistribution mismatch. Input: {input_total_crews}, Redistributed: {total_redistributed_crews}")

        if verbose: 
            print(f"Overlap-based crew redistribution complete: {[(island, crews) for (island, crews) in new_crew_distribution.items() if crews > 0]}")

        return new_crew_distribution

    # Handle other cases
    elif isinstance(available_repair_crews, dict):
        if verbose: print("No previous dissolved roads provided, treating as initial distribution")
        current_crew_distribution = {island: 0 for island in unique_islands}
        for island_id, crew_count in available_repair_crews.items():
            if island_id in current_crew_distribution:
                current_crew_distribution[island_id] = crew_count
            else:
                if current_crew_distribution:
                    first_island = list(current_crew_distribution.keys())[0]
                    current_crew_distribution[first_island] += crew_count
                    print(f"Warning! Redistributed {crew_count} crews from missing island {island_id} to island {first_island}")
        
        print(f"Updated crew distribution: {current_crew_distribution}")
        return current_crew_distribution
    
    else:
        print("Unexpected crew distribution format, treating as initial distribution")
        return update_repair_crew_islands_with_overlap_cached(
            len(unique_islands) * 2, island_ids, dissolved_roads,
            current_map=current_map, previous_map=previous_map,
            hazard_threshold=hazard_threshold, overlap_cache=overlap_cache,
            hazard_dir=hazard_dir
        )

def compute_island_geodataframe_from_graph(graph_pickle_path: str, hazard_threshold: float, hazard_column: str, buffer_distance: float = 2.5, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from graph with buffered road geometries for spatial operations.
    Deduplicates before buffering for efficiency.
    """
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)
        # G = nx.DiGraph(G)

    G = project_graph_coords(G, from_crs="EPSG:4326", to_crs="EPSG:28992")
    G = filter_hazard_graph(G, hazard_threshold, hazard_column)

    # Identify strongly connected components
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    fid_to_island = {}

    # Assign island_id to each fid
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            fid_to_island[u] = i
            fid_to_island[v] = i

    # Create transformer for projecting geometries
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    
    # Build edge records with geometry and length 
    records = []

    for u, v, data in G.edges(data=True):
        # Project the actual edge geometry
        original_geom = data['geometry']
        if hasattr(original_geom, 'coords'):
            # Project all coordinates in the geometry
            projected_coords = [transformer.transform(x, y) for x, y in original_geom.coords]
            projected_geom = LineString(projected_coords)
        else:
            # Fallback: create LineString from node positions if no edge geometry
            projected_geom = LineString([(G.nodes[u]["x_m"], G.nodes[u]["y_m"]),
                                       (G.nodes[v]["x_m"], G.nodes[v]["y_m"])])
        
        length_m = data.get("length", None)
        if length_m is None:
            print(f"Length not found for edge ({u}, {v}), calculating from projected geometry.")
            length_m = projected_geom.length  # Use original linestring for length calculation

        island_id = fid_to_island.get(v, -1)

        record = data.copy()
        record["geometry"] = projected_geom  # Keep original linestring geometry for now
        record["length_m"] = length_m
        record["island_id"] = island_id
        records.append(record)

    # Create initial GeoDataFrame with linestring geometries
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")

    # Deduplication and buffering
    gdf = gdf.drop_duplicates(subset=["geometry", "island_id"])
    if verbose:
        print(f"After deduplication: {len(gdf)} road segments")

    buffered_geometries = []
    original_geometries = []
    
    for _, row in gdf.iterrows():
        original_geom = row.geometry
        
        # Buffer the linestring to create a polygon for spatial operations
        try:
            buffered_geom = original_geom.buffer(buffer_distance, cap_style='square', join_style='mitre')
            if buffered_geom.is_empty or not buffered_geom.is_valid:
                buffered_geom = original_geom.make_valid().buffer(buffer_distance, cap_style='square', join_style='mitre')
        except Exception as e:
            print(f"Warning: Error buffering geometry: {e}, using original")
            buffered_geom = original_geom
        
        buffered_geometries.append(buffered_geom)
        original_geometries.append(original_geom)
    
    # Update geometries
    gdf['original_geometry'] = original_geometries
    gdf['geometry'] = buffered_geometries  # Replace with buffered geometries
    
    if verbose:
        print(f"Buffered {len(gdf)} road segments with {buffer_distance}m buffer")

    # Compute island sizes using original linestring lengths (more accurate for road network analysis)
    island_sizes = gdf.groupby("island_id")["length_m"].sum().reset_index()
    island_sizes["island_size_km"] = island_sizes["length_m"] / 1000.0
    island_sizes = island_sizes[["island_id", "island_size_km"]]

    # Merge island sizes back into GeoDataFrame
    gdf = gdf.merge(island_sizes, on="island_id", how="left")
    gdf["island_size_km"] = gdf["island_size_km"].fillna(0.0)

    if verbose:
        print(f"Island distribution: {gdf['island_id'].value_counts().sort_index().to_dict()}")
    return gdf

def match_island_ids_assets(temp_gdf, hazard_threshold=0.2, hazard_column='EV1_ma', config=None):
    """
    Get islands and assign IDs to assets with stable boundary island identification using geographic features.
    Boundary assets are excluded immediately based on spatial location.
    """
    if config is None:
        _config = get_config()
    else:
        _config = config    

    verbose = _config['simulation_config']['verbose']    
    start_time = time.time()
    boundary_assets_cache_path = _config['interim_dir'] / 'boundary_assets.pkl'
    boundary_islands_cache_path = _config['interim_dir'] / 'boundary_islands.pkl'
    
    # Load cached boundary assets if available
    try:
        hazard_graph_path = _config['hazard_dir'].parent / 'static' / 'output_graph' / f'base_graph_hazard_editted.p'
        if verbose: 
            print(f"Loading graph from: {hazard_graph_path}")
            print(f"Using hazard_threshold={hazard_threshold}, hazard_column={hazard_column}")

        islands_gdf = compute_island_geodataframe_from_graph(
            hazard_graph_path, hazard_threshold=hazard_threshold, 
            hazard_column=hazard_column, buffer_distance=20, verbose=verbose
        )

        # Dissolve roads by island_id to get road network per island
        dissolved_roads = islands_gdf.dissolve(by='island_id', as_index=False) #it is possible that dissolving is not necessary?
        if verbose: 
            print(f"Created {len(dissolved_roads)} dissolved road islands")

        # Initialize island_id column with -1 for all assets
        # temp_gdf = temp_gdf.copy()
        temp_gdf['island_id'] = -1

        projected_crs = 'epsg:28992'
        dissolved_roads = dissolved_roads.to_crs(projected_crs)
        temp_gdf = temp_gdf.to_crs(projected_crs) #TODO: Drop unused columns

        array_island_ids = dissolved_roads.island_id.values
        def assign_island_id(asset_geom) -> int:  
            mask = asset_geom.intersects(dissolved_roads.geometry)
            ids = array_island_ids[mask]
            if len(ids) == 0:
                return -1
            return ids[0]
        
        # Assign island_id for all assets (works for both EV0 and else)
        temp_gdf['island_id'] = temp_gdf.geometry.apply(assign_island_id)
        main_island_id = dissolved_roads['island_id'].value_counts().idxmax()

        # Handle boundary island identification using geographic features
        is_initial_status = 'EV0' in hazard_column
        if is_initial_status: 
            if verbose: 
                print("EV0 initial status - caching and scrapping boundary assets and islands")   

            # Identify unassigned assets (island_id == -1)
            unassigned_mask = temp_gdf['island_id'] == -1
            unassigned_indices = temp_gdf[unassigned_mask].index.tolist()

            # Try to assign unassigned assets to an island by distance
            distance_threshold = 5.0  # meters
            for idx in unassigned_indices:
                asset_geom = temp_gdf.loc[idx, 'geometry']
                # Exclude main island
                distances = dissolved_roads.geometry.distance(asset_geom)
                if not distances.empty and distances.min() < distance_threshold:
                    nearest_idx = distances.idxmin()
                    nearest_island_id = dissolved_roads.loc[nearest_idx, 'island_id']
                    temp_gdf.loc[idx, 'island_id'] = nearest_island_id

            # Now, boundary assets are those with island_id != main_island_id or still -1
            boundary_asset_mask = (temp_gdf['island_id'] != main_island_id) | (temp_gdf['island_id'] == -1)
            boundary_asset_indices = temp_gdf[boundary_asset_mask].index.tolist()

            # Pickle for future use
            with open(boundary_assets_cache_path, 'wb') as f:
                pickle.dump(boundary_asset_indices, f)

            # # Set all island_id to 0 for EV0 (no filtering)
            # temp_gdf['island_id'] = 0

            # Identify boundary islands (non-main islands that exist in baseline)
            boundary_islands = dissolved_roads[dissolved_roads['island_id'] != main_island_id]
            
            # Create stable geographic identifiers for boundary islands
            boundary_island_features = []
            for _, island_row in boundary_islands.iterrows():
                geom = island_row.geometry
                centroid = geom.centroid
                bounds = geom.bounds
                
                # Create a stable geographic signature
                geo_signature = {
                    'centroid_x': round(centroid.x, 1),  # Round to avoid floating point issues
                    'centroid_y': round(centroid.y, 1),
                    'bounds_minx': round(bounds[0], 1),
                    'bounds_miny': round(bounds[1], 1), 
                    'bounds_maxx': round(bounds[2], 1),
                    'bounds_maxy': round(bounds[3], 1),
                    'area': round(geom.area, 1),
                    'osmid': island_row['osmid']
                }
                boundary_island_features.append(geo_signature)

            if verbose:
                print(f"Found {len(boundary_island_features)} boundary islands by geography (all assets assigned to main island)")

            # Cache boundary geographic features (stable across timesteps)
            with open(boundary_islands_cache_path, 'wb') as f:
                pickle.dump(boundary_island_features, f)

            if verbose:
                print(f"Cached {len(boundary_island_features)} boundary island geographic features from {hazard_column}")
            
            # IMMEDIATE EXCLUSION: Return only main island assets and roads
            clean_gdf_assets = temp_gdf  # All assets are already on island 0
            clean_dissolved_roads = dissolved_roads[dissolved_roads['island_id'] == main_island_id]
            
            if verbose:
                print(f"EV0 baseline: {len(clean_gdf_assets)} assets on main island, {len(boundary_islands)} boundary islands excluded")
                print(f"EV0 processing completed in {time.time() - start_time:.2f} seconds")

        else:
            # For non-EV0 timesteps
            spatial_start = time.time()
            if verbose:
                print(f"Performing spatial assignment of {len(temp_gdf)} assets to {len(dissolved_roads)} islands...")

            # Load boundary asset indices
            with open(boundary_assets_cache_path, 'rb') as f:
                boundary_asset_indices = pickle.load(f)

            # Load boundary island features (geo-signatures) from EV0
            with open(boundary_islands_cache_path, 'rb') as f:
                boundary_island_features = pickle.load(f)

            # Identify boundary assets using spatial signature match
            boundary_island_ids_current = []
            tolerance = 5.0  # meters
            current_non_main_islands = dissolved_roads[dissolved_roads['island_id'] != main_island_id]

            for cached_feature in boundary_island_features:
                for _, current_island in current_non_main_islands.iterrows():
                    current_geom = current_island.geometry
                    current_centroid = current_geom.centroid
                    current_bounds = current_geom.bounds

                    centroid_match = (abs(current_centroid.x - cached_feature['centroid_x']) < tolerance and
                                    abs(current_centroid.y - cached_feature['centroid_y']) < tolerance)
                    bounds_match = (abs(current_bounds[0] - cached_feature['bounds_minx']) < tolerance and
                                    abs(current_bounds[1] - cached_feature['bounds_miny']) < tolerance and
                                    abs(current_bounds[2] - cached_feature['bounds_maxx']) < tolerance and
                                    abs(current_bounds[3] - cached_feature['bounds_maxy']) < tolerance)
                    area_match = abs(current_geom.area - cached_feature['area']) < (tolerance * tolerance)

                    if centroid_match or (bounds_match and area_match):
                        boundary_island_ids_current.append(current_island['island_id'])
                        if verbose: 
                            print(f"Matched boundary island ID {current_island['island_id']} using cached geographic signature")
                        break

            # Exclude boundary islands from dissolved_roads
            dissolved_roads = dissolved_roads[~dissolved_roads['island_id'].isin(boundary_island_ids_current)]
            # array_island_ids = dissolved_roads.island_id.values
            # # Assign island_id for non-boundary assets only
            # non_boundary_mask = ~temp_gdf.index.isin(boundary_asset_indices)
            # temp_gdf.loc[non_boundary_mask, 'island_id'] = temp_gdf.loc[non_boundary_mask, 'geometry'].apply(assign_island_id)
            temp_gdf.loc[boundary_asset_indices, 'island_id'] = main_island_id  # Assign boundary assets to main island (or -1 if preferred)
            
            # For any remaining unassigned assets, use spatial index or assign to largest island as fallback
            unassigned_mask = temp_gdf['island_id'] == -1
            unassigned_count = unassigned_mask.sum()

            if unassigned_count > 0:
                print(f"  Using spatial index for {unassigned_count} unassigned assets...")
                spatial_idx = create_spatial_index(dissolved_roads)
                unassigned_assets = temp_gdf[unassigned_mask]
                for idx, asset_row in unassigned_assets.iterrows():
                    asset_geom = asset_row.geometry
                    asset_point = asset_geom if asset_geom.geom_type == 'Point' else asset_geom.centroid
                    nearby_islands = list(spatial_idx.intersection(asset_point.bounds))
                    if nearby_islands:
                        candidate_roads = dissolved_roads.loc[nearby_islands]
                        distances = candidate_roads.geometry.distance(asset_point)
                        nearest_candidate_idx = distances.idxmin()
                        nearest_island_id = candidate_roads.loc[nearest_candidate_idx, 'island_id']
                    else:
                        # Fallback: assign to largest island (usually island 0)
                        # largest_island = dissolved_roads.loc[dissolved_roads.geometry.area.idxmax()]
                        nearest_island_id = main_island_id#largest_island['island_id']
                    temp_gdf.loc[idx, 'island_id'] = nearest_island_id
            
            clean_dissolved_roads=dissolved_roads
            clean_gdf_assets=temp_gdf

            if verbose:
                print(f"Assigned {len(temp_gdf)} assets to {len(dissolved_roads)} islands")
                print(f"Island distribution: {temp_gdf['island_id'].value_counts().sort_index().to_dict()}")
                print(f"Spatial assignment completed in {time.time() - spatial_start:.2f} seconds")
        
        if verbose: 
            print(f"Final results: {len(clean_gdf_assets)} clean assets, {len(clean_dissolved_roads)} clean road islands")
        print(f">>>>Total processing time: {time.time() - start_time:.2f} seconds")

        return clean_gdf_assets, clean_dissolved_roads
        
    except Exception as e:
        print(f"Error in match_island_ids_assets: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        temp_gdf_copy = temp_gdf.copy()
        temp_gdf_copy['island_id'] = 0
        return temp_gdf_copy, None


    
