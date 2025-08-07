
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from rtree import index
from datetime import datetime
import numpy as np
import networkx as nx
from shapely.geometry import LineString
import geopandas as gpd
import pandas as pd
import pickle
from pyproj import Transformer

from src.utils import project_graph_coords, filter_hazard_graph
from src.caching import load_island_cache, save_island_cache_silent, save_island_cache, create_overlap_cache_key

# Import hazard extraction method from config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

# Get the hazard extraction method from config
_config = get_config()

def compute_island_geodataframe_from_graph(graph_pickle_path: str, hazard_threshold: float, hazard_column: str) -> gpd.GeoDataFrame:
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)
        G = nx.DiGraph(G)

    G = project_graph_coords(G, from_crs="EPSG:4326", to_crs="EPSG:28992")
    G = filter_hazard_graph(G, hazard_threshold, hazard_column)

    # Identify strongly connected components
    components = list(nx.strongly_connected_components(G))
    fid_to_island = {}

    # Assign island_id to each fid
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
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
            length_m = projected_geom.length  

        island_id = fid_to_island.get(v, -1)

        record = data.copy()
        record["geometry"] = projected_geom  # Use properly projected geometry
        record["length_m"] = length_m
        record["island_id"] = island_id
        records.append(record)

    # Create GeoDataFrame with correct CRS
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")

    # Compute island sizes using groupby
    island_sizes = gdf.groupby("island_id")["length_m"].sum().reset_index()
    island_sizes["island_size_km"] = island_sizes["length_m"] / 1000.0
    island_sizes = island_sizes[["island_id", "island_size_km"]]

    # Merge island sizes back into GeoDataFrame
    gdf = gdf.merge(island_sizes, on="island_id", how="left")
    gdf["island_size_km"] = gdf["island_size_km"].fillna(0.0)

    return gdf

def precompute_island_assignments(hazard_maps, flood_thresholds, gdf_assets, 
                                       interim_dir, hazard_dir, accessibility_model=None):
    """Pre-compute island assignments for all hazard maps and flood thresholds."""
    print(f"Pre-computing island assignments for {len(hazard_maps)} hazard maps and {len(flood_thresholds)} thresholds...")
    
    cache_dir = interim_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    island_cache = load_island_cache(cache_dir, hazard_dir)
    
    computed_count = 0
    skipped_count = 0
    
    for day_counter, hazard_map in enumerate(hazard_maps):
        for threshold in flood_thresholds:
            haz_col_str = f'EV{day_counter}_ma'
            cache_key = f"{threshold}_{haz_col_str}"  
            
            if cache_key in island_cache:
                skipped_count += 1
                continue
            
            try:
                print(f"  Computing [{computed_count+1}]: {Path(hazard_map).name}, threshold {threshold}")
                
                # Use the same function as runtime
                temp_gdf_for_islands = gdf_assets.copy()
                temp_gdf_for_islands, dissolved_roads = match_island_ids_assets(
                    temp_gdf_for_islands, 
                    hazard_threshold=threshold, 
                    hazard_column=haz_col_str,
                    config=_config
                )
                island_ids = temp_gdf_for_islands['island_id'].values
                
                # Store with EXACT runtime format
                island_data = {
                    'hazard_map': str(hazard_map),
                    'threshold': threshold,
                    'island_ids': island_ids,
                    'dissolved_roads': dissolved_roads,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'precomputed',
                    'method': 'match_island_ids_assets'
                }
                island_cache[cache_key] = island_data
                computed_count += 1
                
                # Save intermediate cache silently 
                if computed_count % 5 == 0:
                    save_island_cache_silent(island_cache, cache_dir, hazard_dir)
                    print(f"    Saved intermediate cache after {computed_count} computations")
                    
            except Exception as e:
                print(f"    Error: {e}")
                computed_count += 1
    
    # Final save with message
    save_island_cache(island_cache, cache_dir, hazard_dir)
    
    print(f"Island assignment pre-computation complete:")
    print(f"  Total combinations: {len(hazard_maps) * len(flood_thresholds)}")
    print(f"  Newly computed: {computed_count}")
    print(f"  Skipped (cached): {skipped_count}")
    
    return island_cache

def create_spatial_index(dissolved_roads):
    """
    Create R-tree spatial index for fast spatial queries
    """
    # Build R-tree index
    idx = index.Index()
    
    # Insert each island's bounding box into the index
    for i, row in dissolved_roads.iterrows():
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        idx.insert(i, bounds, obj=row)
    
    return idx

def _optimized_overlap_calculation(current_islands, previous_islands, buffer_distance=1):
    """
    Geometric intersection computation with R-tree spatial indexing and bounds checking
    """
    # Pre-buffer all geometries once
    current_buffered = current_islands.copy()
    current_buffered['geometry'] = current_islands.geometry.buffer(buffer_distance)
    
    previous_buffered = previous_islands.copy()  
    previous_buffered['geometry'] = previous_islands.geometry.buffer(buffer_distance)
    
    spatial_idx = create_spatial_index(current_buffered)
    
    overlaps_by_prev_island = {}
    
    for _, prev_island in previous_buffered.iterrows():
        prev_geom = prev_island.geometry
        prev_area = prev_geom.area  
        prev_bounds = prev_geom.bounds  
        
        overlaps = {}
        
        # Use R-tree to get candidates (prevents checking all)
        try:
            # Get candidate islands from spatial index
            candidate_indices = list(spatial_idx.intersection(prev_bounds))
            
            # If no spatial candidates found, skip this island
            if not candidate_indices:
                overlaps_by_prev_island[prev_island['island_id']] = overlaps
                continue
                
        except Exception as e:
            # Fallback to all islands if R-tree fails
            print(f"Warning: R-tree query failed, falling back to full scan: {e}")
            candidate_indices = list(current_buffered.index)
        
        # Process only candidate islands
        for candidate_idx in candidate_indices:
            try:
                current_island = current_buffered.iloc[candidate_idx]
                current_geom = current_island.geometry
                current_bounds = current_geom.bounds
                
                # Check if within bounds
                if not (prev_bounds[2] >= current_bounds[0] and  # prev_maxx >= curr_minx
                        prev_bounds[0] <= current_bounds[2] and  # prev_minx <= curr_maxx
                        prev_bounds[3] >= current_bounds[1] and  # prev_maxy >= curr_miny
                        prev_bounds[1] <= current_bounds[3]):    # prev_miny <= curr_maxy
                    continue
                    
                # Geometric intersection check
                if prev_geom.intersects(current_geom):  # Boolean check first
                    intersection = prev_geom.intersection(current_geom)
                    if not intersection.is_empty:
                        overlap_pct = (intersection.area / prev_area)
                        overlaps[current_island['island_id']] = overlap_pct
                        
            except Exception:
                continue
        
        overlaps_by_prev_island[prev_island['island_id']] = overlaps
    
    return overlaps_by_prev_island

def update_repair_crew_islands_with_overlap_cached(
    available_repair_crews, island_ids, dissolved_roads, 
    previous_dissolved_roads=None, buffer_distance=1,
    current_day=None, previous_day=None, hazard_threshold=None, 
    overlap_cache=None, hazard_dir=None, verbose=False
):
    """
    Update repair crew distribution with cached overlap percentages.
    """
    present_islands = dissolved_roads.copy()
    unique_islands = np.unique(island_ids)
    unique_islands = unique_islands[~pd.isna(unique_islands)]
    
    print(f"Processing {len(unique_islands)} islands with crew distribution...")
    
    if isinstance(available_repair_crews, int):
        # Initial distribution logic when crews are given as an integer
        print(f"Initial distribution of {available_repair_crews} crews across {len(unique_islands)} islands")
        
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
        
        if (overlap_cache is not None and current_day is not None and 
            previous_day is not None and hazard_threshold is not None):
            
            overlap_cache_key = create_overlap_cache_key(previous_day, current_day, hazard_threshold, hazard_dir)
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
            
            # Cache the result (only store percentages)
            if overlap_cache is not None and overlap_cache_key is not None:
                overlap_cache[overlap_cache_key] = overlaps_by_prev_island
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
                    print(f"Assigned {crew_count} crews to nearest island {nearest_island_id}")
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
                        print(f"Redistributed {crew_count} crews from previous island {prev_island_id} based on cached overlaps")
        
        if total_redistributed_crews != input_total_crews:
            print(f"Crew redistribution mismatch. Input: {input_total_crews}, Redistributed: {total_redistributed_crews}")
        
        if verbose: print(f"Overlap-based crew redistribution complete: {[(island, crews) for (island, crews) in new_crew_distribution.items() if crews > 0]}")
        
        return new_crew_distribution

    # Handle other cases
    elif isinstance(available_repair_crews, dict):
        print("No previous dissolved roads provided, treating as initial distribution")
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
            current_day=current_day, previous_day=previous_day,
            hazard_threshold=hazard_threshold, overlap_cache=overlap_cache,
            hazard_dir=hazard_dir
        )


def match_island_ids_assets(temp_gdf, hazard_threshold=0.2, hazard_column='EV1_ma', config=_config):
    """
    Get islands wrapping polygons for a specific day and assign island IDs to assets.
    
    Parameters:
    temp_gdf (gpd.GeoDataFrame): GeoDataFrame with asset geometries
    hazard_threshold (float): Threshold for hazard analysis
    hazard_column (str): Column name for hazard data
    config (dict): Configuration dictionary with 'root_dir' key
    
    Returns:
    tuple: (temp_gdf_with_island_ids, dissolved_roads) where temp_gdf_with_island_ids 
           has island_id column and dissolved_roads is the islands wrapping polygons
    """
    try:
        # Fetch graph with grid-based accessibility hex with islands
        hazard_graph_path = config['root_dir'] / 'data' / 'static' / 'output_graph' / f'base_graph_hazard_editted.p'
        print(f"Loading graph from: {hazard_graph_path}")
        print(f"Using hazard_threshold={hazard_threshold}, hazard_column={hazard_column}")
        
        islands_gdf = compute_island_geodataframe_from_graph(hazard_graph_path, hazard_threshold=hazard_threshold, hazard_column=hazard_column)
        print(f"Loaded {len(islands_gdf)} island features")
        
        islands_gdf = islands_gdf.drop_duplicates(subset=["geometry", "island_id"])
        print(f"After deduplication: {len(islands_gdf)} island features")

        # Dissolve roads by island_id to get road network per island
        dissolved_roads = islands_gdf.dissolve(by='island_id', as_index=False)
        print(f"Created {len(dissolved_roads)} dissolved road islands")
        
        # Initialize island_id column with -1 for all assets
        temp_gdf = temp_gdf.copy()  # Make sure we're working with a copy
        temp_gdf['island_id'] = -1  # Initialize island_id column with -1 for all assets

        projected_crs = 'epsg:28992'  # Use a projected CRS for accurate distance calculations
        dissolved_roads = dissolved_roads.to_crs(projected_crs)
        temp_gdf = temp_gdf.to_crs(projected_crs)  # Ensure temp_gdf is in the same CRS

        # Assign each asset to the nearest island
        for idx, asset_row in temp_gdf.iterrows():
            asset_geom = asset_row.geometry
            
            # Calculate distance to island boundaries
            distances = dissolved_roads.geometry.distance(asset_geom)
            nearest_island_idx = distances.idxmin()
            nearest_island_id = dissolved_roads.loc[nearest_island_idx, 'island_id']
            
            # Update the island_id in temp_gdf 
            temp_gdf.loc[idx, 'island_id'] = nearest_island_id

        print(f"Successfully assigned {len(temp_gdf)} assets to islands")
        return temp_gdf, dissolved_roads
        
    except Exception as e:
        print(f"Error in match_island_ids_assets: {e}")
        print(f"Hazard graph path: {hazard_graph_path}")
        print(f"Hazard threshold: {hazard_threshold}")
        print(f"Hazard column: {hazard_column}")
        import traceback
        traceback.print_exc()
        # Fallback: assign all assets to island 0
        temp_gdf_copy = temp_gdf.copy()
        temp_gdf_copy['island_id'] = 0
        return temp_gdf_copy, None