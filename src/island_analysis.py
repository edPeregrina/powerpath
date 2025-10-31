
from tabnanny import verbose
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
from src.caching import load_island_cache, create_island_cache_key, save_island_cache, create_overlap_cache_key, save_overlap_cache, get_asset_centroid_hash

# Import hazard extraction method from config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

# #progress apply 
from tqdm import tqdm
tqdm.pandas()

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
        idx.insert(i, bounds)#, obj=row) #use case for obj=row is only when the rest of the row should be directly queried
    
    return idx

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

def match_assets_access(temp_gdf, hazard_threshold=0.2, hazard_column='EV0_ma',
                       config=None, island_cache=None, cache_dir=None, hazard_dir=None):
    """
    Assign each asset in temp_gdf to the closest road section in islands_gdf using spatial index. 
    This step is executed at initialization only since the access rfid is an attribute of the graph that does not change.
    """
    if config is None:
        _config = get_config()
    else:
        _config = config    
    verbose = _config['simulation_config']['verbose'] 
    boundary_assets_cache_path = _config['interim_dir'] / 'boundary_assets.pkl'
    boundary_islands_cache_path = _config['interim_dir'] / 'boundary_islands.pkl'
    asset_access_cache_path = _config['interim_dir'] / 'asset_access_rfid.pkl'
    road_segment_lengths_cache_path = _config['interim_dir'] / 'road_segment_lengths.pkl'
    asset_hash = get_asset_centroid_hash(temp_gdf)

    # Check if cached or initialize empty dictionaries
    def load_or_init(path):
        return pickle.load(open(path, 'rb')) if path.exists() else {}

    boundary_assets_dict = load_or_init(boundary_assets_cache_path)
    access_assets_dict = load_or_init(asset_access_cache_path)

    # Load boundary islands and road segment lengths (these are not per asset set)
    boundary_islands_rfids = pickle.load(open(boundary_islands_cache_path, 'rb')) if boundary_islands_cache_path.exists() else None
    rfids_lengths = pickle.load(open(road_segment_lengths_cache_path, 'rb')) if road_segment_lengths_cache_path.exists() else None

    # Check if all required cached values exist for this asset set
    has_boundary = asset_hash in boundary_assets_dict
    has_access = asset_hash in access_assets_dict
    has_islands = boundary_islands_rfids is not None
    has_lengths = rfids_lengths is not None

    if has_boundary and has_access and has_islands and has_lengths:
        boundary_asset_indices = boundary_assets_dict[asset_hash]
        access_rfids = access_assets_dict[asset_hash]
        return access_rfids, boundary_asset_indices, boundary_islands_rfids, rfids_lengths

    # If not cached, compute from scratch
    try:
        hazard_graph_path = _config['hazard_dir'].parent / 'static' / 'output_graph' / f'base_graph_hazard_editted.p'
        # Subgraphs are computed on the fly from the filtered hazard graph. each subgraph is an "island".
        islands_gdf = compute_island_geodataframe_from_graph(
            hazard_graph_path, hazard_threshold=hazard_threshold, 
            hazard_column=hazard_column, buffer_distance=20, verbose=verbose
        )  
        rfids_lengths = dict(zip(islands_gdf['rfid'], islands_gdf['length_m']))

        temp_gdf['access_rfid'] = -1

        projected_crs = 'epsg:28992'
        islands_gdf = islands_gdf.to_crs(projected_crs)
        temp_gdf = temp_gdf.to_crs(projected_crs) 

        # Build spatial index for road sections
        spatial_idx = create_spatial_index(islands_gdf)

        # Find main island id (largest by road length)
        main_island_id = islands_gdf.groupby('island_id')['length_m'].sum().idxmax()

        # For each asset, find nearest road section within a reasonable search radius
        search_radius = 100  # meters

        # Precompute centroids and buffers
        temp_gdf['centroid'] = temp_gdf.geometry.centroid
        temp_gdf['buffer'] = temp_gdf['centroid'].apply(lambda c: c.buffer(search_radius))

        # For each asset, get candidate road indices and assign closest
        def find_nearest_road(asset_row, spatial_idx, islands_gdf):
            asset_centroid = asset_row['centroid']
            asset_buffer = asset_row['buffer']
            candidate_idxs = list(spatial_idx.intersection(asset_buffer.bounds))
            if candidate_idxs:
                candidates = islands_gdf.iloc[candidate_idxs]
                if not candidates.empty:
                    distances = candidates.geometry.distance(asset_centroid)
                    if not distances.empty:
                        nearest_idx = distances.idxmin()
                        return candidates.loc[nearest_idx, 'rfid']
            return -1
        
        temp_gdf['access_rfid'] = temp_gdf.apply(
            lambda row: find_nearest_road(row, spatial_idx, islands_gdf), axis=1
        )

        def safe_island_lookup(rfid, islands_gdf):
            if rfid == -1:
                return -1
            matches = islands_gdf[islands_gdf['rfid'] == rfid]['island_id'].values
            return matches[0] if len(matches) > 0 else -1

        temp_gdf['island_id'] = [safe_island_lookup(rfid, islands_gdf) for rfid in temp_gdf['access_rfid']]

        # Now, boundary assets are those with island_id != main_island_id or still -1
        boundary_asset_mask = (temp_gdf['island_id'] != main_island_id) | (temp_gdf['island_id'] == -1)
        boundary_asset_indices = temp_gdf[boundary_asset_mask].index.tolist()
        print(f"Identified {len(boundary_asset_indices)} boundary assets out of {len(temp_gdf)} total assets.")

        # Identify boundary islands (non-main islands that exist in baseline)
        boundary_islands_rfids = islands_gdf[islands_gdf['island_id'] != main_island_id]['rfid'].to_list()
        
        # Pickle for future use
        boundary_assets_dict[asset_hash] = boundary_asset_indices
        with open(boundary_assets_cache_path, 'wb') as f:
            pickle.dump(boundary_assets_dict, f)

        with open(boundary_islands_cache_path, 'wb') as f:
            pickle.dump(boundary_islands_rfids, f)

        access_assets_dict[asset_hash] = temp_gdf['access_rfid']
        with open(asset_access_cache_path, 'wb') as f:
            pickle.dump(access_assets_dict, f)

        with open(road_segment_lengths_cache_path, 'wb') as f:
            pickle.dump(rfids_lengths, f)

        if verbose:
            print(f"Cached {len(boundary_asset_indices)} boundary assets out of {len(temp_gdf)} total assets.")
            print(f"Cached {len(boundary_islands_rfids)} boundary island geographic features from {hazard_column}")
            print(f"Cached asset access rfids for {len(temp_gdf)} assets.")

        return temp_gdf['access_rfid'], boundary_asset_indices, boundary_islands_rfids, rfids_lengths

    except Exception as e:
        print(f"Error in computing islands from graph: {e}")
        return None, None, None, None

def match_island_ids_assets(temp_gdf, boundary_asset_indices=None, boundary_islands_rfids=None, hazard_threshold=0.2, hazard_column='EV1_ma', config=None,
                            island_cache=None, cache_dir=None, hazard_dir=None):
   
    if config is None:
        _config = get_config()
    else:
        _config = config    
    verbose = _config['simulation_config']['verbose']
    asset_hash = get_asset_centroid_hash(temp_gdf)
    # Create cache key for this computation
    if island_cache is not None and cache_dir is not None: 
        cache_key = create_island_cache_key(hazard_column, hazard_threshold, asset_hash)
        
        # Check if this computation is already cached
        if cache_key in island_cache:
            if verbose:
                print(f"Using cached island assignment for {cache_key}")
            return island_cache[cache_key]['island_ids'], island_cache[cache_key]['rfids_islands']

    if boundary_asset_indices is None or boundary_islands_rfids is None: 
        boundary_assets_cache_path = _config['interim_dir'] / 'boundary_assets.pkl'
        boundary_islands_cache_path = _config['interim_dir'] / 'boundary_islands.pkl'
        # Load or initialize cache dicts
        with open(boundary_assets_cache_path, 'rb') as f:
            boundary_assets_dict = pickle.load(f)
        if asset_hash in boundary_assets_dict:
            boundary_asset_indices = boundary_assets_dict[asset_hash]
        else:
            print(f"Boundary asset indices not provided and not found in cache for asset hash {asset_hash}.")
            return None, None

        with open(boundary_islands_cache_path, 'rb') as f:
            boundary_islands_rfids = pickle.load(f)

    try:
        hazard_graph_path = _config['hazard_dir'].parent / 'static' / 'output_graph' / f'base_graph_hazard_editted.p'
        print(f"Loading hazard graph from {hazard_graph_path}")
        islands_gdf = compute_island_geodataframe_from_graph(
            hazard_graph_path, hazard_threshold=hazard_threshold, 
            hazard_column=hazard_column, buffer_distance=20, verbose=verbose
        )  

        # Drop the boundary rfids and find the main island
        islands_gdf = islands_gdf[~islands_gdf['rfid'].isin(boundary_islands_rfids)].copy()
        main_island_id = islands_gdf.groupby('island_id')['length_m'].sum().idxmax()

        # Map assets to island ids
        rfid_to_island = dict(zip(islands_gdf['rfid'], islands_gdf['island_id']))
        asset_island_ids = [rfid_to_island.get(rfid, -1) for rfid in temp_gdf['access_rfid']]
        asset_island_ids = [asset_island_ids[i] if i not in boundary_asset_indices else main_island_id
                            for i in range(len(asset_island_ids))]
        asset_island_ids = np.array(asset_island_ids, dtype=int)

        rfids_islands = dict(zip(islands_gdf['rfid'], islands_gdf['island_id']))

        # Cache the results
        if island_cache is not None and cache_dir is not None:
            # Store the computed results in the cache
            island_cache[cache_key] = {
                'island_ids': asset_island_ids,
                'rfids_islands': rfids_islands
            }
            
            # Save the updated cache using the standardized function
            save_island_cache(island_cache, cache_dir, hazard_dir)
            
            if verbose:
                print(f"Saved island assignment to cache with key {cache_key}")

    except Exception as e:
        print(f"Error in computing islands from graph: {e}")
        return None

    return asset_island_ids, rfids_islands

def update_repair_crew_islands(available_repair_crews, previous_rfids_islands, current_rfids_islands, rfids_lengths, 
                              verbose=False, overlap_cache=None, current_map=None, previous_map=None, 
                              hazard_threshold=None, hazard_dir=None, _config=None, cache_updated=None):
    """
    Distribute repair crews by island, based on road feature lengths.

    Arguments:
    - available_repair_crews: int (initial round) or dict (subsequent rounds) of available repair crews
    - previous_rfids_islands: dict mapping road feature ids (rfids) to island ids from previous timestep (None if initial round)
    - current_rfids_islands: dict mapping rfids to island ids from current timestep 
    - rfids_lengths: dict mapping rfids to their lengths
    - verbose: bool, whether to print detailed logs
    - overlap_cache: dict for caching overlap computations  
    - current_map: identifier for current hazard map (e.g., filename or timestamp)
    - previous_map: identifier for previous hazard map (None if initial round)
    - hazard_threshold: float, threshold used for hazard impact
    - hazard_dir: directory path for hazard data (used in caching)
    - _config: configuration dictionary (used in caching)
    - cache_updated: dict to track if cache was updated (used in caching)
    """
    try: # Exceptions are handled by returning input crews as fallback, but a stack trace is printed for debugging
        # First round: crews as int
        if isinstance(available_repair_crews, int):
            # Check cache first for initial distribution
            initial_probabilities = None
            overlap_cache_key = None
            
            if (overlap_cache is not None and current_map is not None and 
                hazard_threshold is not None):
                
                # Special cache key for initial distribution
                overlap_cache_key = create_overlap_cache_key("initial", current_map, hazard_threshold, hazard_dir)
                
                if overlap_cache_key in overlap_cache:
                    if verbose:
                        print(f"Using cached initial distribution for {overlap_cache_key}")
                    initial_probabilities = overlap_cache[overlap_cache_key]
            
            # If cache hit, use cached probabilities directly
            if initial_probabilities is not None:
                unique_islands = list(initial_probabilities.keys())
                probabilities = np.array([initial_probabilities[i] for i in unique_islands])
            else:
                # Cache miss - compute island lengths
                print("Computing island lengths for initial distribution...")
                curr_island_lengths = {}
                for rfid, island_id in current_rfids_islands.items():
                    curr_island_lengths.setdefault(island_id, 0)
                    curr_island_lengths[island_id] += rfids_lengths.get(rfid, 0)
                
                unique_islands = list(curr_island_lengths.keys())
                lengths = np.array([curr_island_lengths[i] for i in unique_islands])
                probabilities = lengths / lengths.sum() if lengths.sum() > 0 else np.ones_like(lengths)/len(lengths)
                
                # Cache computed probabilities
                if overlap_cache is not None and overlap_cache_key is not None:
                    initial_probabilities = dict(zip(unique_islands, probabilities))
                    overlap_cache[overlap_cache_key] = initial_probabilities
                    
                    # Save to disk
                    if _config is not None:
                        cache_dir = _config['interim_dir']
                        save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                        if verbose:
                            print(f"Cached initial distribution for {overlap_cache_key}")
                    
                    # Update cache_updated
                    if cache_updated is not None:
                        cache_updated['overlap_cache'] = overlap_cache
            
            assigned = np.random.choice(unique_islands, size=available_repair_crews, p=probabilities, replace=True)
            crew_counts = dict(zip(*np.unique(assigned, return_counts=True)))
            available_repair_crews_by_island = {i: crew_counts.get(i, 0) for i in unique_islands}
            
            if verbose:
                print(f"Initial crew distribution: {available_repair_crews_by_island}")
            return available_repair_crews_by_island

        # Subsequent rounds: crews as dict
        elif isinstance(available_repair_crews, dict):
            # Check cache first for transition probabilities
            transition_probabilities = None
            
            if (overlap_cache is not None and current_map is not None and 
                previous_map is not None and hazard_threshold is not None):
                
                overlap_cache_key = create_overlap_cache_key(previous_map, current_map, hazard_threshold, hazard_dir)
                
                if overlap_cache_key in overlap_cache:
                    if verbose:
                        print(f"Using cached transition probabilities for {overlap_cache_key}")
                    transition_probabilities = overlap_cache[overlap_cache_key]
            
            # If cache miss, compute transition probabilities
            if transition_probabilities is None:
                if verbose:
                    print("Computing transition probabilities (cache miss)")
                
                # Now we need to compute island lengths
                print("Computing island lengths...")
                curr_island_lengths = {}
                prev_island_lengths = {}
                
                for rfid, island_id in current_rfids_islands.items():
                    curr_island_lengths.setdefault(island_id, 0)
                    curr_island_lengths[island_id] += rfids_lengths.get(rfid, 0)
                
                if previous_rfids_islands is not None:
                    for rfid, island_id in previous_rfids_islands.items():
                        prev_island_lengths.setdefault(island_id, 0)
                        prev_island_lengths[island_id] += rfids_lengths.get(rfid, 0)
                
                # Build transition probability dictionary
                transition_probabilities = {}
                
                for prev_island in set(previous_rfids_islands.values()):
                    # Find rfids in previous island
                    rfids_in_prev = [rfid for rfid, island in previous_rfids_islands.items() if island == prev_island]
                    
                    # Map these rfids to current islands and sum lengths
                    curr_lengths = {}
                    for rfid in rfids_in_prev:
                        curr_island = current_rfids_islands.get(rfid, None)
                        if curr_island is not None:
                            curr_lengths.setdefault(curr_island, 0)
                            curr_lengths[curr_island] += rfids_lengths.get(rfid, 0)
                    
                    # Calculate probabilities for this previous island
                    total_length = sum(curr_lengths.values())
                    if total_length > 0:
                        transition_probabilities[prev_island] = {
                            curr_island: length / total_length 
                            for curr_island, length in curr_lengths.items()
                        }
                
                # Cache the computed transition probabilities
                if (overlap_cache is not None and overlap_cache_key is not None):
                    overlap_cache[overlap_cache_key] = transition_probabilities
                    
                    # Save to disk
                    if _config is not None:
                        cache_dir = _config['interim_dir']
                        save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                        if verbose:
                            print(f"Cached transition probabilities for {overlap_cache_key}")
                    
                    # Update cache_updated
                    if cache_updated is not None:
                        cache_updated['overlap_cache'] = overlap_cache
            
            # Use transition probabilities to redistribute crews
            available_repair_crews_by_island = {}
            
            for prev_island, crew_count in available_repair_crews.items():
                if prev_island in transition_probabilities and transition_probabilities[prev_island]:
                    # Get transition probabilities for this island
                    island_transitions = transition_probabilities[prev_island]
                    
                    # Extract islands and probabilities
                    curr_islands = list(island_transitions.keys())
                    probabilities = [island_transitions[i] for i in curr_islands]
                    
                    if verbose:
                        print(f"Probability distribution from/to island {prev_island}: {dict(zip(curr_islands, probabilities))}")
                    
                    # Assign crews based on probabilities
                    assigned = np.random.choice(curr_islands, size=crew_count, p=probabilities, replace=True)
                    crew_counts = dict(zip(*np.unique(assigned, return_counts=True)))
                    
                    for i in curr_islands:
                        available_repair_crews_by_island[i] = available_repair_crews_by_island.get(i, 0) + crew_counts.get(i, 0)
                else:
                    # Fallback: assign all crews to previous island
                    available_repair_crews_by_island[prev_island] = available_repair_crews_by_island.get(prev_island, 0) + crew_count
            
            if verbose:
                print(f"Redistributed crew distribution: {available_repair_crews_by_island}")
            
            return available_repair_crews_by_island
        
        else:
            raise ValueError("available_repair_crews must be int or dict")
            
    except Exception as e:
        print(f"Error in crew redistribution: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return input crews
        if isinstance(available_repair_crews, dict):
            return available_repair_crews
        elif isinstance(available_repair_crews, int):
            # Just put all crews on first island as fallback
            first_island = list(current_rfids_islands.values())[0] if current_rfids_islands else 0
            return {first_island: available_repair_crews}
        else:
            return {}
    