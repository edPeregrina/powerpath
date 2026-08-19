
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
import pickle

# Import hazard extraction method from config
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# Actor-by-island redistribution primitives
# ---------------------------------------------------------------------------

ActorCountsByIsland = dict[int, int]
ActorDistribution = dict[str, ActorCountsByIsland]


def _is_nested_actor_distribution(value) -> bool:
    return isinstance(value, dict) and all(isinstance(v, dict) for v in value.values())


def _extract_actor_counts(
    available_actors,
    actor_type: str,
) -> "tuple[ActorCountsByIsland | int, str, ActorDistribution]":
    """Extract a single actor type from flat/nested compatibility inputs."""
    if isinstance(available_actors, int):
        return available_actors, "int", {actor_type: {}}

    if not isinstance(available_actors, dict):
        raise TypeError("available_actors must be int, dict[island->count], or dict[type->dict]")

    if _is_nested_actor_distribution(available_actors):
        nested_distribution = {k: dict(v) for k, v in available_actors.items()}
        actor_counts = dict(nested_distribution.get(actor_type, {}))
        return actor_counts, "nested", nested_distribution

    flat_distribution = {int(k): int(v) for k, v in available_actors.items()}
    return flat_distribution, "flat", {actor_type: flat_distribution.copy()}


def _pack_actor_counts(
    *,
    updated_actor_counts: "ActorCountsByIsland",
    input_shape: str,
    nested_distribution: "ActorDistribution",
    actor_type: str,
):
    """Pack updated actor counts back to the requested compatibility shape."""
    if input_shape == "nested":
        nested_distribution[actor_type] = updated_actor_counts
        return nested_distribution

    # For int or flat legacy inputs, keep legacy output shape.
    return updated_actor_counts


def _compute_initial_distribution(
    total_actor_count: int,
    current_rfids_islands: "dict[int, int]",
    rfids_lengths: "dict[int, float]",
    *,
    skip_unassigned_islands: bool = False,
):
    curr_island_lengths: dict[int, float] = {}
    for rfid, island_id in current_rfids_islands.items():
        if skip_unassigned_islands and island_id == -1:
            continue
        curr_island_lengths.setdefault(island_id, 0.0)
        curr_island_lengths[island_id] += rfids_lengths.get(rfid, 0.0)

    unique_islands = list(curr_island_lengths.keys())
    if not unique_islands:
        unique_islands = [0]
        probabilities = np.array([1.0], dtype=np.float64)
    else:
        lengths = np.array([curr_island_lengths[i] for i in unique_islands], dtype=np.float64)
        total_length = lengths.sum()
        probabilities = lengths / total_length if total_length > 0 else np.ones_like(lengths) / len(lengths)

    assigned = np.random.choice(unique_islands, size=total_actor_count, p=probabilities, replace=True)
    actor_counts = dict(zip(*np.unique(assigned, return_counts=True))) if total_actor_count > 0 else {}
    return {island: int(actor_counts.get(island, 0)) for island in unique_islands}, dict(zip(unique_islands, probabilities))


def _compute_transition_probabilities(
    previous_rfids_islands: "dict[int, int]",
    current_rfids_islands: "dict[int, int]",
    rfids_lengths: "dict[int, float]",
):
    transition_probabilities: dict = {}

    for prev_island in set(previous_rfids_islands.values()):
        if prev_island == -1:
            continue

        rfids_in_prev = [rfid for rfid, island in previous_rfids_islands.items() if island == prev_island]
        curr_lengths: dict[int, float] = {}
        for rfid in rfids_in_prev:
            curr_island = current_rfids_islands.get(rfid, None)
            if curr_island is None or curr_island == -1:
                continue
            curr_lengths.setdefault(curr_island, 0.0)
            curr_lengths[curr_island] += rfids_lengths.get(rfid, 0.0)

        total_length = sum(curr_lengths.values())
        if total_length > 0:
            transition_probabilities[prev_island] = {
                curr_island: length / total_length
                for curr_island, length in curr_lengths.items()
            }

    return transition_probabilities


def _redistribute_by_transition_probabilities(
    actor_counts: "ActorCountsByIsland",
    transition_probabilities,
    *,
    verbose: bool = False,
):
    redistributed: dict[int, int] = {}

    for prev_island, count in actor_counts.items():
        if prev_island == -1:
            if verbose:
                print(f"WARNING: Skipping actor redistribution from island_id = -1 ({count} actors lost)")
            continue

        if transition_probabilities.get(prev_island):
            island_transitions = transition_probabilities[prev_island]
            curr_islands = list(island_transitions.keys())
            probabilities = [island_transitions[i] for i in curr_islands]

            if verbose:
                print(f"Probability distribution from/to island {prev_island}: {dict(zip(curr_islands, probabilities))}")

            assigned = np.random.choice(curr_islands, size=count, p=probabilities, replace=True)
            assigned_counts = dict(zip(*np.unique(assigned, return_counts=True))) if count > 0 else {}
            for island in curr_islands:
                redistributed[island] = redistributed.get(island, 0) + int(assigned_counts.get(island, 0))
        else:
            redistributed[prev_island] = redistributed.get(prev_island, 0) + int(count)

    return redistributed
from src.caching import (
    create_island_cache_key,
    create_overlap_cache_key,
    get_asset_centroid_hash,
    save_island_cache,
    save_overlap_cache,
)
from src.utils import create_spatial_index, filter_hazard_graph, project_graph_coords

sys.path.append(str(Path(__file__).parent.parent))
# #progress apply 
from tqdm import tqdm

from config import get_config

tqdm.pandas()


def _load_pickle_cache(path):
    if not path.exists():
        return {}
    try:
        with open(path, 'rb') as cache_file:
            return pickle.load(cache_file)
    except Exception as exc:
        print(f"Ignoring unreadable cache {path}: {exc}")
        return {}


def compute_island_geodataframe_from_graph(
    graph_pickle_path: str, 
    hazard_threshold: float, 
    hazard_column: str, 
    buffer_distance: float = 2.5, 
    verbose: bool = False,
    l1_area_geojson=None,
    l2_asset_geojson=None,
) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from graph with buffered road geometries.
    Applies active L1/L2 road adaptations before filtering if provided.
    """
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)

    G = project_graph_coords(G, from_crs="EPSG:4326", to_crs="EPSG:28992")
    
    G = filter_hazard_graph(
        G, hazard_threshold, hazard_column, 
        l1_area_geojson=l1_area_geojson,
        l2_asset_geojson=l2_asset_geojson,
        verbose=verbose
    )

    # # DEBUG 1: Check graph structure after filtering
    # print(f"DEBUG 1: Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Identify strongly connected components
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    # # DEBUG 2: Check component distribution
    # print(f"DEBUG 2: Found {len(components)} connected components")
    # component_sizes = sorted([len(comp) for comp in components], reverse=True)
    # print(f"DEBUG 2: Component sizes: {component_sizes[:10]}")  # Top 10
    
    fid_to_island = {}

    # Assign island_id to each fid
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            fid_to_island[u] = i
            fid_to_island[v] = i

    # # DEBUG 3: Check fid_to_island mapping
    # print(f"DEBUG 3: fid_to_island has {len(fid_to_island)} entries")
    # print(f"DEBUG 3: Island IDs in fid_to_island: {sorted(set(fid_to_island.values()))[:20]}")  # First 20

    # Create transformer for projecting geometries
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    
    # Build edge records with geometry and length 
    records = []
    # missing_island_count = 0  # DEBUG counter

    for u, v, data in G.edges(data=True):
        # Project the actual edge geometry
        original_geom = data['geometry']
        if hasattr(original_geom, 'coords'):
            projected_coords = [transformer.transform(x, y) for x, y in original_geom.coords]
            projected_geom = LineString(projected_coords)
        else:
            projected_geom = LineString([(G.nodes[u]["x_m"], G.nodes[u]["y_m"]),
                                       (G.nodes[v]["x_m"], G.nodes[v]["y_m"])])
        
        length_m = data.get("length", None)
        if length_m is None:
            length_m = projected_geom.length

        island_id = fid_to_island.get(v, -1)
        
        # # DEBUG 4: Track edges that get island_id = -1
        # if island_id == -1:
        #     missing_island_count += 1
        #     if missing_island_count <= 5:  # Print first 5 cases
        #         print(f"DEBUG 4: Edge ({u}, {v}) has island_id = -1")
        #         print(f"         u in fid_to_island: {u in fid_to_island}")
        #         print(f"         v in fid_to_island: {v in fid_to_island}")

        record = data.copy()
        record["geometry"] = projected_geom
        record["length_m"] = length_m
        record["island_id"] = island_id
        records.append(record)

    # # DEBUG 5: Check how many edges got -1
    # print(f"DEBUG 5: {missing_island_count} edges assigned island_id = -1 out of {len(records)}")

    # Create initial GeoDataFrame with linestring geometries
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")

    # # DEBUG 6: Check GeoDataFrame island distribution BEFORE deduplication
    # print(f"DEBUG 6: BEFORE dedup - island_id distribution:")
    # print(gdf['island_id'].value_counts().sort_index())

    # Deduplication and buffering
    gdf = gdf.drop_duplicates(subset=["geometry", "island_id"])
    
    # # DEBUG 7: Check GeoDataFrame island distribution AFTER deduplication
    # print(f"DEBUG 7: AFTER dedup - island_id distribution:")
    # print(gdf['island_id'].value_counts().sort_index())
    
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
                       config=None, island_cache=None, cache_dir=None, hazard_dir=None,
                       l1_area_geojson=None, l1_active_timesteps=None,
                       l2_asset_geojson=None, l2_active_timesteps=None):
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
    road_state_key = create_island_cache_key(
        hazard_column,
        hazard_threshold,
        asset_hash,
        l1_area_geojson=l1_area_geojson,
        l1_active_timesteps=l1_active_timesteps,
        l2_asset_geojson=l2_asset_geojson,
        l2_active_timesteps=l2_active_timesteps,
    )

    # Check if cached or initialize empty dictionaries
    def load_or_init(path):
        return _load_pickle_cache(path)

    boundary_assets_dict = load_or_init(boundary_assets_cache_path)
    access_assets_dict = load_or_init(asset_access_cache_path)
    boundary_islands_dict = load_or_init(boundary_islands_cache_path)
    rfids_lengths_dict = load_or_init(road_segment_lengths_cache_path)
    if not isinstance(boundary_islands_dict, dict):
        boundary_islands_dict = {}
    if rfids_lengths_dict and not isinstance(next(iter(rfids_lengths_dict.values())), dict):
        rfids_lengths_dict = {}

    # Check if all required cached values exist for this asset set
    has_boundary = road_state_key in boundary_assets_dict
    has_access = road_state_key in access_assets_dict
    has_islands = road_state_key in boundary_islands_dict
    has_lengths = road_state_key in rfids_lengths_dict

    if has_boundary and has_access and has_islands and has_lengths:
        boundary_asset_indices = boundary_assets_dict[road_state_key]
        access_rfids = access_assets_dict[road_state_key]
        boundary_islands_rfids = boundary_islands_dict[road_state_key]
        rfids_lengths = rfids_lengths_dict[road_state_key]
        return access_rfids, boundary_asset_indices, boundary_islands_rfids, rfids_lengths

    # If not cached, compute from scratch
    try:
        hazard_graph_path = _config['hazard_dir'].parent / 'static' / 'output_graph' / 'base_graph_hazard_editted.p'
        
        islands_gdf = compute_island_geodataframe_from_graph(
            hazard_graph_path, 
            hazard_threshold=hazard_threshold, 
            hazard_column=hazard_column, 
            buffer_distance=20, 
            verbose=verbose,
            l1_area_geojson=l1_area_geojson,
            l2_asset_geojson=l2_asset_geojson,
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
        # # DEBUG 8: Check rfid assignments
        # print(f"DEBUG 8: access_rfid distribution:")
        # rfid_counts = temp_gdf['access_rfid'].value_counts()
        # print(f"         Assets with rfid = -1: {(temp_gdf['access_rfid'] == -1).sum()}")
        # print(f"         Assets with valid rfid: {(temp_gdf['access_rfid'] != -1).sum()}")

        def safe_island_lookup(rfid, islands_gdf):
            if rfid == -1:
                return -1
            matches = islands_gdf[islands_gdf['rfid'] == rfid]['island_id'].values
            
            # # DEBUG 9: Track failed lookups
            # if len(matches) == 0:
            #     print(f"DEBUG 9: rfid {rfid} not found in islands_gdf")
            #     return -1
            
            island_id = matches[0]
            
            # # DEBUG 10: Track if matched island_id is -1
            # if island_id == -1:
            #     print(f"DEBUG 10: rfid {rfid} has island_id = -1 in islands_gdf")
            
            return island_id

        temp_gdf['island_id'] = [safe_island_lookup(rfid, islands_gdf) for rfid in temp_gdf['access_rfid']]

        # # DEBUG 11: Final asset island distribution
        # print(f"DEBUG 11: Final asset island_id distribution:")
        # print(temp_gdf['island_id'].value_counts().sort_index())
        # print(f"          Assets on island -1: {(temp_gdf['island_id'] == -1).sum()}")
        
        # # DEBUG 12: Check specific asset 66 if it exists
        # if 66 in temp_gdf.index:
        #     asset_66_rfid = temp_gdf.loc[66, 'access_rfid']
        #     asset_66_island = temp_gdf.loc[66, 'island_id']
        #     print(f"DEBUG 12: Asset 66 -> rfid {asset_66_rfid} -> island {asset_66_island}")
            
        #     if asset_66_rfid != -1:
        #         matching_roads = islands_gdf[islands_gdf['rfid'] == asset_66_rfid]
        #         if len(matching_roads) > 0:
        #             print(f"          Road segment for asset 66 has island_id: {matching_roads['island_id'].iloc[0]}")
        #         else:
        #             print(f"          WARNING: rfid {asset_66_rfid} not found in islands_gdf!")

        # Now, boundary assets are those with island_id != main_island_id or still -1
        boundary_asset_mask = (temp_gdf['island_id'] != main_island_id) | (temp_gdf['island_id'] == -1)
        boundary_asset_indices = temp_gdf[boundary_asset_mask].index.tolist()
        print(f"Identified {len(boundary_asset_indices)} boundary assets out of {len(temp_gdf)} total assets.")

        # Identify boundary islands (non-main islands that exist in baseline)
        boundary_islands_rfids = islands_gdf[islands_gdf['island_id'] != main_island_id]['rfid'].to_list()
        
        # Pickle for future use
        boundary_assets_dict[road_state_key] = boundary_asset_indices
        with open(boundary_assets_cache_path, 'wb') as f:
            pickle.dump(boundary_assets_dict, f)

        boundary_islands_dict[road_state_key] = boundary_islands_rfids
        with open(boundary_islands_cache_path, 'wb') as f:
            pickle.dump(boundary_islands_dict, f)

        access_assets_dict[road_state_key] = temp_gdf['access_rfid']
        with open(asset_access_cache_path, 'wb') as f:
            pickle.dump(access_assets_dict, f)

        rfids_lengths_dict[road_state_key] = rfids_lengths
        with open(road_segment_lengths_cache_path, 'wb') as f:
            pickle.dump(rfids_lengths_dict, f)

        if verbose:
            print(f"Cached {len(boundary_asset_indices)} boundary assets out of {len(temp_gdf)} total assets.")
            print(f"Cached {len(boundary_islands_rfids)} boundary island geographic features from {hazard_column}")
            print(f"Cached asset access rfids for {len(temp_gdf)} assets.")

        return temp_gdf['access_rfid'], boundary_asset_indices, boundary_islands_rfids, rfids_lengths

    except Exception as e:
        print(f"Error in computing islands from graph: {e}")
        return None, None, None, None

def match_island_ids_assets(temp_gdf, boundary_asset_indices=None, boundary_islands_rfids=None, 
                            hazard_threshold=0.2, hazard_column='EV1_ma', config=None,
                            island_cache=None, cache_dir=None, hazard_dir=None,
                            l1_area_geojson=None, l1_active_timesteps=None,
                            l2_asset_geojson=None, l2_active_timesteps=None,
                            societal_allocation_cache=None, pop_grid_gdf=None,
                            cell_id_column='cell_id', island_id_column='island_id',
                            nearest_max_distance=200.0):
       
    """
    Match assets to island IDs based on spatial intersection with road network islands.
    Cache key now includes L1 adaptation hash if L1 is active.
    """
    if config is None:
        _config = get_config()
    else:
        _config = config    
    verbose = _config['simulation_config']['verbose']
    asset_hash = get_asset_centroid_hash(temp_gdf)
    cache_key = create_island_cache_key(
        hazard_column,
        hazard_threshold,
        asset_hash,
        l1_area_geojson=l1_area_geojson,
        l1_active_timesteps=l1_active_timesteps,
        l2_asset_geojson=l2_asset_geojson,
        l2_active_timesteps=l2_active_timesteps,
    )

    cache_hit = (
        island_cache is not None
        and cache_dir is not None
        and cache_key in island_cache
    )
    if cache_hit and verbose:
        print(f"Using cached island assignment for {cache_key}")

    if boundary_asset_indices is None or boundary_islands_rfids is None: 
        boundary_assets_cache_path = _config['interim_dir'] / 'boundary_assets.pkl'
        boundary_islands_cache_path = _config['interim_dir'] / 'boundary_islands.pkl'
        # Load or initialize cache dicts
        boundary_assets_dict = _load_pickle_cache(boundary_assets_cache_path)
        if cache_key in boundary_assets_dict:
            boundary_asset_indices = boundary_assets_dict[cache_key]
        else:
            print(f"Boundary asset indices not provided and not found in cache for asset hash {asset_hash}.")
            return None, None

        boundary_islands_dict = _load_pickle_cache(boundary_islands_cache_path)
        if not isinstance(boundary_islands_dict, dict) or cache_key not in boundary_islands_dict:
            print(f"Boundary islands not found in cache for road state {cache_key}.")
            return None, None
        boundary_islands_rfids = boundary_islands_dict[cache_key]

    asset_island_ids = None
    rfids_islands = None
    if cache_hit:
        asset_island_ids = island_cache[cache_key]['island_ids']
        rfids_islands = island_cache[cache_key]['rfids_islands']

    needs_transient_islands = (
        societal_allocation_cache is not None
        and pop_grid_gdf is not None
    )
    if cache_hit and not needs_transient_islands:
        return asset_island_ids, rfids_islands

    try:
        hazard_graph_path = _config['hazard_dir'].parent / 'static' / 'output_graph' / 'base_graph_hazard_editted.p'
        print(f"Loading hazard graph from {hazard_graph_path}")
        islands_gdf = compute_island_geodataframe_from_graph(
            hazard_graph_path, 
            hazard_threshold=hazard_threshold, 
            hazard_column=hazard_column, 
            buffer_distance=20, 
            verbose=verbose,
            l1_area_geojson=l1_area_geojson,
            l2_asset_geojson=l2_asset_geojson,
        )

        # Drop the boundary rfids and find the main island
        islands_gdf = islands_gdf[~islands_gdf['rfid'].isin(boundary_islands_rfids)].copy()
        main_island_id = islands_gdf.groupby('island_id')['length_m'].sum().idxmax()

        if asset_island_ids is None or rfids_islands is None:
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
                'rfids_islands': rfids_islands,
            }
            
            # Save the updated cache using the standardized function
            save_island_cache(island_cache, cache_dir, hazard_dir)
            
            if verbose:
                print(f"Saved island assignment to cache with key {cache_key}")

        if needs_transient_islands:
            from src.societal_access import get_or_build_allocation

            get_or_build_allocation(
                societal_allocation_cache,
                pop_grid_gdf,
                cell_id_column,
                islands_gdf,
                island_id_column=island_id_column,
                nearest_max_distance=nearest_max_distance,
                road_state_key=cache_key,
            )

    except Exception as e:
        print(f"Error in computing islands from graph: {e}")
        return None, None

    return asset_island_ids, rfids_islands

def update_repair_crew_islands(
    available_repair_crews,      
    previous_rfids_islands,      
    current_rfids_islands,       
    rfids_lengths,               
    verbose=False, 
    overlap_cache=None, 
    current_map=None, 
    previous_map=None, 
    hazard_threshold=None, 
    hazard_dir=None, 
    _config=None, 
    cache_updated=None,
    l1_area_geojson=None,
    l1_active_timesteps=None
):
    """
    Distribute repair crews by island while keeping the island logic local.
    """
    try:
        return update_actor_islands(
            available_repair_crews,
            previous_rfids_islands,
            current_rfids_islands,
            rfids_lengths,
            actor_type="repair_crews",
            verbose=verbose,
            overlap_cache=overlap_cache,
            current_map=current_map,
            previous_map=previous_map,
            hazard_threshold=hazard_threshold,
            hazard_dir=hazard_dir,
            _config=_config,
            cache_updated=cache_updated,
            l1_area_geojson=l1_area_geojson,
            l1_active_timesteps=l1_active_timesteps,
        )
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


def update_actor_islands(
    available_actors,
    previous_rfids_islands,
    current_rfids_islands,
    rfids_lengths,
    *,
    actor_type="repair_crews",
    verbose=False,
    overlap_cache=None,
    current_map=None,
    previous_map=None,
    hazard_threshold=None,
    hazard_dir=None,
    _config=None,
    cache_updated=None,
    l1_area_geojson=None,
    l1_active_timesteps=None,
):
    """Redistribute actors across the current island partition.

    This public implementation stays in the island-analysis layer so the repair
    crew workflow, cache semantics, and notebook/API behavior remain attached to
    the original domain module.
    """
    actor_counts, input_shape, nested_distribution = _extract_actor_counts(
        available_actors,
        actor_type,
    )

    if isinstance(actor_counts, int):
        initial_probabilities = None
        overlap_cache_key = None

        if overlap_cache is not None and current_map is not None and hazard_threshold is not None:
            overlap_cache_key = create_overlap_cache_key(
                "initial",
                current_map,
                hazard_threshold,
                hazard_dir,
                l1_area_geojson=l1_area_geojson,
            )
            if overlap_cache_key in overlap_cache:
                if verbose:
                    print(f"Using cached initial distribution for {overlap_cache_key}")
                initial_probabilities = overlap_cache[overlap_cache_key]

        if initial_probabilities is not None:
            unique_islands = list(initial_probabilities.keys())
            probabilities = np.array(
                [initial_probabilities[i] for i in unique_islands],
                dtype=np.float64,
            )
            assigned = np.random.choice(
                unique_islands,
                size=actor_counts,
                p=probabilities,
                replace=True,
            )
            sampled_counts = (
                dict(zip(*np.unique(assigned, return_counts=True)))
                if actor_counts > 0
                else {}
            )
            redistributed_counts = {
                island_id: int(sampled_counts.get(island_id, 0))
                for island_id in unique_islands
            }
        else:
            redistributed_counts, computed_probabilities = _compute_initial_distribution(
                actor_counts,
                current_rfids_islands,
                rfids_lengths,
            )
            if overlap_cache is not None and overlap_cache_key is not None:
                overlap_cache[overlap_cache_key] = computed_probabilities
                if _config is not None:
                    cache_dir = _config["interim_dir"]
                    save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                    if verbose:
                        print(f"Cached initial distribution for {overlap_cache_key}")
                if cache_updated is not None:
                    cache_updated["overlap_cache"] = overlap_cache

        if verbose:
            print(f"Initial {actor_type} distribution: {redistributed_counts}")

        return _pack_actor_counts(
            updated_actor_counts=redistributed_counts,
            input_shape=input_shape,
            nested_distribution=nested_distribution,
            actor_type=actor_type,
        )

    if previous_rfids_islands is None:
        if verbose:
            print(f"First timestep with dict {actor_type}: redistributing from aggregate count")
        total_count = int(sum(actor_counts.values()))
        redistributed_counts, _ = _compute_initial_distribution(
            total_count,
            current_rfids_islands,
            rfids_lengths,
            skip_unassigned_islands=True,
        )
        if verbose:
            print(f"Initial {actor_type} distribution (from dict): {redistributed_counts}")
        return _pack_actor_counts(
            updated_actor_counts=redistributed_counts,
            input_shape=input_shape,
            nested_distribution=nested_distribution,
            actor_type=actor_type,
        )

    transition_probabilities = None
    overlap_cache_key = None
    if (
        overlap_cache is not None
        and current_map is not None
        and previous_map is not None
        and hazard_threshold is not None
    ):
        overlap_cache_key = create_overlap_cache_key(
            previous_map,
            current_map,
            hazard_threshold,
            hazard_dir,
            l1_area_geojson=l1_area_geojson,
            l1_active_timesteps=l1_active_timesteps,
        )

        if overlap_cache_key in overlap_cache:
            if verbose:
                print(f"Using cached transition probabilities for {overlap_cache_key}")
            transition_probabilities = overlap_cache[overlap_cache_key]

    if transition_probabilities is None:
        if verbose:
            print("Computing transition probabilities (cache miss)")
        transition_probabilities = _compute_transition_probabilities(
            previous_rfids_islands,
            current_rfids_islands,
            rfids_lengths,
        )

        if overlap_cache is not None and overlap_cache_key is not None:
            overlap_cache[overlap_cache_key] = transition_probabilities
            if _config is not None:
                cache_dir = _config["interim_dir"]
                save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
                if verbose:
                    print(f"Cached transition probabilities for {overlap_cache_key}")
            if cache_updated is not None:
                cache_updated["overlap_cache"] = overlap_cache

    redistributed_counts = _redistribute_by_transition_probabilities(
        actor_counts,
        transition_probabilities,
        verbose=verbose,
    )

    if verbose:
        print(f"Redistributed {actor_type} distribution: {redistributed_counts}")

    return _pack_actor_counts(
        updated_actor_counts=redistributed_counts,
        input_shape=input_shape,
        nested_distribution=nested_distribution,
        actor_type=actor_type,
    )