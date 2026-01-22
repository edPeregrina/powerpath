"""
Functions to run the damage and recovery simulation.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.caching import load_accessibility_cache, load_island_cache, load_overlap_cache, load_hazard_extraction_cache, save_accessibility_cache, save_overlap_cache, save_hazard_extraction_cache
from src.island_analysis import match_island_ids_assets, match_assets_access, update_repair_crew_islands, compute_island_geodataframe_from_graph
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.damage_recovery import default_damage_ratio_function, default_repair_time_function, vectorized_damage_ratio_solver, default_fragility_function
from src.caching import create_accessibility_cache_key, create_island_cache_key, get_asset_centroid_hash
from src.adaptation import build_l1_l2_reduction_array, _build_adaptation_arrays_cached

# Import hazard extraction method from config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
from shutil import copyfile

class SimulationState:
    def __init__(self, gdf_assets, num_assets):
        self.previous_map_counter = None
        self.damage_ratio = np.zeros(num_assets, dtype=np.float64)
        self.repair_time = np.zeros(num_assets, dtype=np.float64)
        self.accessible = np.ones(num_assets, dtype=bool)
        self.unreachable = np.zeros(num_assets, dtype=bool)
        self.operational = np.ones(num_assets, dtype=bool)
        self.repair_crews_assigned = np.zeros(num_assets, dtype=bool)
        self.current_hazard_values = np.zeros(num_assets, dtype=np.float64)
        self.island_ids = np.zeros(num_assets, dtype=int)
        # self.temp_gdf = gdf_assets[['type', 'geometry']].copy()

def _update_hazard_map_states(
    state, gdf_assets, rfids_lengths, timestep, major_timestep, hazard_maps, haz_dir_name, 
    flood_threshold, repair_crew_assignment_method, _config, accessibility_cache, 
    hazard_extraction_cache, overlap_cache, island_cache, boundary_asset_indices, 
    boundary_islands_rfids, interim_dir, hazard_dir, available_repair_crews, 
    previous_rfids_islands, previous_map_counter, asset_type, num_assets, verbose, 
    fragility_param_k=None, depth_reductions=None, l1_area_geojson=None
):
    """
    Update the simulation states that depend on the hazard map (only on major timesteps)

    Args:
        state (SimulationState): Current state of the simulation
        gdf_assets (GeoDataFrame): GeoDataFrame of assets with geometries and asset types
        timestep (int): Current timestep in the simulation
        major_timestep (int): Major timestep interval for hazard map updates
        hazard_maps (list): List of paths to hazard map files (raster format)
        haz_dir_name (str): Name of the hazard directory (for caching)
        flood_threshold (float): Hazard value threshold for flooding
        _config (dict): Configuration settings for the simulation
        accessibility_cache (dict): Cache for accessibility results
        hazard_extraction_cache (dict): Cache for hazard extraction results
        overlap_cache (dict): Cache for overlap results
        island_cache (dict): Cache for island analysis results
        hazard_dir (Path): Path to the directory containing hazard maps
        available_repair_crews (int or dict): Number of available repair crews or a dictionary with island IDs as keys
        previous_rfids_islands (dict or None): Previous mapping of road RFIDs to island IDs, if any
        previous_map_counter (int or None): Previous map counter, if any
        asset_type (np.ndarray): Array of asset types corresponding to each asset
        num_assets (int): Total number of assets in the simulation
        verbose (bool): If True, print detailed simulation information
        fragility_param_k (float or None): Parameter k for the fragility function, if applicable
        depth_reductions (np.ndarray or None): 2D array of flood depth reductions

    Returns:
        tuple: Updated available_repair_crews, previous_rfids_islands, previous_map_counter, cache_updated (dictionary of updated caches)
    """

    repair_threshold = _config['recovery_parameters'].get('repair_threshold', 2.0)
    damage_ratio_coefficients = _config['recovery_parameters'].get('damage_ratio_coefficients', (0.0468, 0.0077))
    repair_time_coefficients = _config['recovery_parameters'].get('repair_time_coefficients', [702.72, 3.14, 1.9891])
    cache_updated = {}

    map_counter = int(timestep / major_timestep)
    if map_counter >= len(hazard_maps):
        print(f"No more hazard maps available at timestep {timestep}, ending simulation.")
        return available_repair_crews, previous_map_counter, cache_updated

    hazard_map = hazard_maps[map_counter]
    haz_col_str = f'EV{map_counter}_ma'

    if verbose:
        print(f"\n=== Processing timestep {timestep} (map {map_counter}) ===")
    
    # Update hazard values
    temp_gdf = gdf_assets[['type', 'geometry', 'access_rfid']].copy()
    temp_gdf = find_hazard_value_at_points_optimized(
        hazard_map,
        temp_gdf,
        map_counter,
        extraction_method=_config['analysis_config']['hazard_extraction_method'],
        hazard_cache=hazard_extraction_cache,
        hazard_dir=hazard_dir, 
        _config=_config
    )
    cache_updated['hazard_extraction_cache'] = hazard_extraction_cache
    haz_val_str = f'hazard_value_{map_counter}'

    if haz_val_str in temp_gdf.columns:
        state.current_hazard_values = temp_gdf[haz_val_str].fillna(0.0).values
    else:
        state.current_hazard_values = temp_gdf[haz_col_str].fillna(0.0).values

    # Apply L1/L2 adaptations (once per map, not accumulated per hour)
    if depth_reductions is not None:
        if timestep < depth_reductions.shape[0]:
            state.current_hazard_values = np.maximum(
                0.0, 
                state.current_hazard_values - depth_reductions[timestep, :]
            )

    # Island-based crew management
    if 'island' in repair_crew_assignment_method:
        assert isinstance(available_repair_crews, dict), f"Island method requires dict of recpair crews at this point, got {type(available_repair_crews)}"
        asset_hash = get_asset_centroid_hash(temp_gdf)
        cache_key = create_island_cache_key(
            haz_col_str, 
            flood_threshold, 
            asset_hash,
            l1_area_geojson=l1_area_geojson  
        )
        if cache_key in island_cache:
            island_data = island_cache[cache_key]
            state.island_ids = island_data['island_ids']
            rfids_islands = island_data['rfids_islands']
            if verbose:
                print(f"Using cached islands for {cache_key}")
        else:
            print(f"Cache miss for {cache_key}, computing islands on the fly...")
            try:
                temp_gdf_for_islands = temp_gdf
                
                asset_island_ids, rfids_islands = match_island_ids_assets(
                    temp_gdf_for_islands, 
                    boundary_asset_indices=boundary_asset_indices, 
                    boundary_islands_rfids=boundary_islands_rfids, 
                    hazard_threshold=flood_threshold, 
                    hazard_column=haz_col_str, 
                    config=_config,
                    island_cache=island_cache,
                    cache_dir=interim_dir,
                    hazard_dir=hazard_dir,
                    l1_area_geojson=l1_area_geojson  
                )
                state.island_ids = asset_island_ids
                # Assign island for each asset for the current state
                cache_updated['island_cache'] = island_cache
                print(f"Successfully computed and cached islands for {cache_key}")
            except Exception as e:
                print(f"Error computing islands for {cache_key}: {e}")
                print("Falling back to simple island assignment")
                state.island_ids = np.ones(num_assets, dtype=int)
                rfids_islands = None
       
        if rfids_islands is not None:
            # Determine previous map string if available
            previous_map_str = f'EV{previous_map_counter}_ma' if previous_map_counter is not None else None
            current_map_str = haz_col_str  # Already defined as f'EV{map_counter}_ma'
            
            # Call with all caching parameters
            available_repair_crews = update_repair_crew_islands(
                available_repair_crews,
                previous_rfids_islands, 
                rfids_islands, 
                rfids_lengths,
                verbose=verbose,
                overlap_cache=overlap_cache,
                current_map=current_map_str,
                previous_map=previous_map_str,
                hazard_threshold=flood_threshold,
                hazard_dir=hazard_dir,
                _config=_config,
                cache_updated=cache_updated,
                l1_area_geojson=l1_area_geojson
            )

            previous_map_counter = map_counter
            previous_rfids_islands = rfids_islands.copy()
            temp_gdf['island_id'] = state.island_ids                

        else:
            print(f"Reachability not available for {cache_key}, using global crew assignment")

    # Mask of assets flooded above threshold
    flooded_mask = state.current_hazard_values > flood_threshold

    # Apply fragility to assets that are not currently under repair
    assets_to_evaluate = flooded_mask & ~state.repair_crews_assigned & state.operational

    # Update operational status based on fragility for assets above threshold
    if np.any(assets_to_evaluate):
        fragility_operational = np.ones_like(state.operational, dtype=bool)
        hazard_subset = state.current_hazard_values[assets_to_evaluate]
        asset_type_subset = asset_type[assets_to_evaluate]
        fragility_result = default_fragility_function(hazard_subset, asset_type_subset, k=fragility_param_k, major_timestep=major_timestep)
        fragility_operational[assets_to_evaluate] = fragility_result.astype(bool)
        state.operational = np.minimum(state.operational, fragility_operational)

    # Update damage ratio and repair time for assets flooded above threshold this timestep
    if np.any(flooded_mask):
        dr_new = default_damage_ratio_function(state.current_hazard_values[flooded_mask], damage_ratio_coefficients)
        newly_damaged_mask = np.zeros_like(flooded_mask, dtype=bool)
        flooded_indices = np.where(flooded_mask)[0]
        new_damage_check = dr_new > state.damage_ratio[flooded_mask]
        newly_damaged_mask[flooded_indices] = new_damage_check
        state.damage_ratio[flooded_mask] = np.maximum(state.damage_ratio[flooded_mask], dr_new)
        state.repair_time[flooded_mask] = default_repair_time_function(
            state.damage_ratio[flooded_mask], repair_time_coefficients
        )
        if verbose:
            try:
                damage_count = newly_damaged_mask.sum()
                print(f"  New damage at timestep {timestep}: {damage_count} assets")
                if damage_count > 0:
                    print(f"  Damage ratios: {state.damage_ratio[newly_damaged_mask].min():.3f} to {state.damage_ratio[newly_damaged_mask].max():.3f}")
                    print(f"  Repair times: {state.repair_time[newly_damaged_mask].min():.1f} to {state.repair_time[newly_damaged_mask].max():.1f} hours")
            except Exception as e:
                print(f"  Error occurred while logging damage information: {e}, {timestep}")

    # For assets needing repair, solve for current damage ratio excluding assets under repair threshold
    recalc_repair_mask = (state.repair_time > repair_threshold)
    if np.any(recalc_repair_mask):
        repair_times_under_repair = state.repair_time[recalc_repair_mask]
        damage_ratios_from_repair = vectorized_damage_ratio_solver(
            repair_times_under_repair, repair_time_coefficients
        )
        state.damage_ratio[recalc_repair_mask] = damage_ratios_from_repair

    accessibility_model = _config['simulation_config']['accessibility_model']
    if accessibility_model is not None:
        # Daily accessibility update
        accessibility_cache_key = create_accessibility_cache_key(
            map_counter, flood_threshold, hazard_dir,
            accessibility_model=accessibility_model
        )
        if accessibility_cache_key in accessibility_cache:
            state.accessible = accessibility_cache[accessibility_cache_key]
            if verbose:
                print(f"Using cached accessibility for map {map_counter} (hazard dir: {haz_dir_name})")
        else:
            try:
                assets_copy = temp_gdf.copy()
                accessibility_result = grid_hex.accessibility_model(
                    assets_copy.geometry, 
                    hazard_map, 
                    state.current_hazard_values,
                    verbose=verbose,
                    day_string=str(state.day_counter).zfill(2),
                    project_root=_config['paths']['root_dir'],
                )
                # accessibility_result = state.accessible  # Defaulting to accessible, to use only islands logic
                state.accessible = np.array(accessibility_result, dtype=bool)
                accessibility_cache[accessibility_cache_key] = state.accessible
                cache_updated['accessibility_cache'] = accessibility_cache

                if verbose:
                    print(f"Accessibility updated for timestep {timestep} (map {map_counter})")
                    print(f"Accessible assets: {state.accessible.sum()} out of {num_assets}")
            except Exception as e:
                print(f"Warning: Accessibility model failed: {e}")
                print("Keeping current accessibility status")
    else:
        pass

    return available_repair_crews, previous_rfids_islands, previous_map_counter, cache_updated

def update_repair_crew_assignment_optimized(timestep, available_repair_crews, repair_crews_assigned, 
                                           accessible, flooded_mask, repair_time, island_ids=None, method=None, verbose=False, asset_impact_map=None):    
    """
    Assign repair crews to assets based on accessibility, flooding status, repair time, and assignment method.

    Args:
        timestep (int): Current timestep in the simulation.
        available_repair_crews (int or dict): Number of available repair crews (global int or per-island dict).
        repair_crews_assigned (np.ndarray): Boolean array indicating which assets have crews assigned.
        accessible (np.ndarray): Boolean array indicating which assets are accessible.
        flooded_mask (np.ndarray): Boolean array indicating which assets are flooded.
        repair_time (np.ndarray): Array of remaining repair times for each asset.
        island_ids (np.ndarray, optional): Array of island IDs for each asset (for island-based assignment).
        method (str, optional): Assignment strategy:
            - 'random': Assign randomly
            - 'lowest repair time': Assign to assets with lowest repair time
            - 'highest repair time': Assign to assets with highest repair time
            - 'island': Assign by island (requires available_repair_crews as dict)
        verbose (bool, optional): If True, print assignment details.
        asset_impact_map (dict, optional): Mapping of asset indices to their impact values.

    Returns:
        tuple: (updated available_repair_crews, updated repair_crews_assigned)
            available_repair_crews: updated int or dict after assignment
            repair_crews_assigned: updated boolean array
    """
    
    # Check if available repair crews is None, meaning no constraints
    if available_repair_crews is None:
        print("No constraints on repair crews, all assets are assigned for repair.")
        # If no constraints, assign all repairable assets
        repair_crews_assigned[:] = True
        return available_repair_crews, repair_crews_assigned 
    
    # If available_repair_crews is a dictionary, it means we have island-based constraints
    if isinstance(available_repair_crews, dict):
        # Assign repair crews based on island IDs
        for island_id, crew_count in available_repair_crews.items():
            if crew_count > 0:  # for each island with available crews
                if island_ids is not None: # If assets have island IDs, only assign to assets in that island
                    island_mask = (np.atleast_1d(np.array(island_ids)) == island_id)
                    repairable_assets = accessible & ~flooded_mask & (repair_time > 0) & island_mask & ~repair_crews_assigned
                else:
                    repairable_assets = accessible & ~flooded_mask & (repair_time > 0) & ~repair_crews_assigned

                if repairable_assets.sum() <= crew_count: # More crews than assets needing assignment
                    newly_assigned_crews = repairable_assets.sum()
                    repair_crews_assigned[repairable_assets] = True
                    # Bounds checking: ensure we don't assign more crews than available
                    newly_assigned_crews = min(newly_assigned_crews, available_repair_crews[island_id])
                    available_repair_crews[island_id] -= newly_assigned_crews
                    
                    if verbose and newly_assigned_crews > 0:
                        print(f"Assigned {newly_assigned_crews} repair crews to island {island_id}")
                else: # If there are more repairable assets than available crews, assign based on method
                    if verbose:
                        print(f"Assigning repair crews to island {island_id} with {crew_count} available crews and {repairable_assets.sum()} repairable assets")
                    
                    repairable_assets_indices = np.where(repairable_assets)[0]
                    if method is None or method == 'random' or method == 'islands':
                        np.random.shuffle(repairable_assets_indices)
                        repair_crews_assigned[repairable_assets_indices[:crew_count]] = True
                    elif 'lowest repair time' in method:
                        sorted_indices = np.argsort(repair_time[repairable_assets])
                        repair_crews_assigned[repairable_assets_indices[sorted_indices[:crew_count]]] = True
                    elif 'highest repair time' in method:
                        sorted_indices = np.argsort(-repair_time[repairable_assets])
                        repair_crews_assigned[repairable_assets_indices[sorted_indices[:crew_count]]] = True
                    elif 'impact' in method:
                        sorted_indices = np.argsort(-np.array([asset_impact_map.get(idx, 0) for idx in repairable_assets_indices]))
                        repair_crews_assigned[repairable_assets_indices[sorted_indices[:crew_count]]] = True

                    newly_assigned_crews = crew_count
                    # Bounds checking: ensure we don't assign more crews than available
                    newly_assigned_crews = min(newly_assigned_crews, available_repair_crews[island_id])
                    available_repair_crews[island_id] -= newly_assigned_crews
                    
                    if verbose:
                        print(f"Assigned {newly_assigned_crews} repair crews to island {island_id} based on method '{method}'")

        # Return the updated assignment
        return available_repair_crews, repair_crews_assigned

    # If available_repair_crews is an int, we have a global constraint
    if available_repair_crews == 0:
        return available_repair_crews, repair_crews_assigned 
    
    if available_repair_crews > 0:       
        repairable_assets = accessible & ~flooded_mask & (repair_time > 0) & ~repair_crews_assigned

        if not 'islands' in str(method):  # Convert method to string to handle None case
            # If there are more repair crews than assets needing assignment, assign all repairable assets
            if repairable_assets.sum() <= available_repair_crews:
                newly_assigned_crews = repairable_assets.sum()
                repair_crews_assigned[repairable_assets] = True
                available_repair_crews -= newly_assigned_crews
                
                return available_repair_crews, repair_crews_assigned
            
            # If there are fewer repair crews than assets needing assignment
            repairable_assets_indices = np.where(repairable_assets)[0]
            if method is None or method == 'random':
                np.random.shuffle(repairable_assets_indices)
                repair_crews_assigned[repairable_assets_indices[:available_repair_crews]] = True
            elif method == 'lowest repair time':
                sorted_indices = np.argsort(repair_time[repairable_assets])
                repair_crews_assigned[repairable_assets_indices[sorted_indices[:available_repair_crews]]] = True
            elif method == 'highest repair time':
                sorted_indices = np.argsort(-repair_time[repairable_assets])
                repair_crews_assigned[repairable_assets_indices[sorted_indices[:available_repair_crews]]] = True
            
            newly_assigned_crews = available_repair_crews
            available_repair_crews -= newly_assigned_crews
                
            if verbose:
                print(f"Assigned {newly_assigned_crews} repair crews to assets based on method '{method}'")
                print(f"->there remain {repairable_assets.sum() - newly_assigned_crews} repairable assets with no crews assigned")

            return available_repair_crews, repair_crews_assigned

        else:
            # Handle island-based methods that weren't caught above
            if 'islands' in str(method) or 'island' in str(method):
                print(f"Island-based method '{method}' detected, but no valid island constraints provided")
                print("Falling back to global assignment with the base method")
                
                # Extract the base method from island-based methods
                base_method = method
                if 'islands' in str(method):
                    base_method = method.replace('islands', '').strip()
                elif 'island' in str(method):
                    base_method = method.replace('island', '').strip()
                
                # Apply the base method globally
                repairable_assets_indices = np.where(repairable_assets)[0]
                crews_to_assign = min(available_repair_crews, len(repairable_assets_indices))
                
                if base_method == 'lowest repair time' or 'lowest' in base_method:
                    sorted_indices = np.argsort(repair_time[repairable_assets])
                    repair_crews_assigned[repairable_assets_indices[sorted_indices[:crews_to_assign]]] = True
                elif base_method == 'highest repair time' or 'highest' in base_method:
                    sorted_indices = np.argsort(-repair_time[repairable_assets])
                    repair_crews_assigned[repairable_assets_indices[sorted_indices[:crews_to_assign]]] = True
                else:
                    # Default to random for unknown base methods
                    np.random.shuffle(repairable_assets_indices)
                    repair_crews_assigned[repairable_assets_indices[:crews_to_assign]] = True
                
                available_repair_crews -= crews_to_assign
                
                if verbose:
                    print(f"Applied base method '{base_method}' globally, assigned {crews_to_assign} repair crews")
                
                return available_repair_crews, repair_crews_assigned
            else:
                print(f"Method '{method}' not implemented yet, returning current assignment.")
                return available_repair_crews, repair_crews_assigned
    
    return available_repair_crews, repair_crews_assigned

_DEPTH_REDUCTION_CACHE = {}

def _initialize_simulation(
    gdf_assets, hazard_maps, recovery_parameters, root_dir, config, repair_crew_assignment_method,
    accessibility_cache=None, hazard_extraction_cache=None, overlap_cache=None, island_cache=None, 
    fragility_param_k=None, major_timestep=24,
    l1_area_geojson=None, l1_active_timesteps=None, 
    l2_asset_geojson=None, l2_active_timesteps=None, 
    verbose=False  
):
    """
    Handles all setup: paths, directories, config, caches, adaptation arrays, etc.
    
    Args:
        gdf_assets (GeoDataFrame): Asset geometries and types
        hazard_maps (list): List of hazard map file paths
        recovery_parameters (dict): Recovery model parameters
        root_dir (str or Path): Root directory for data storage
        config (dict): Simulation configuration dictionary
        repair_crew_assignment_method (str): Method for assigning repair crews
        accessibility_cache (dict, optional): Existing accessibility cache
        hazard_extraction_cache (dict, optional): Existing hazard extraction cache
        overlap_cache (dict, optional): Existing overlap cache
        island_cache (dict, optional): Existing island cache
        fragility_param_k (float, optional): Fragility parameter for operational status modeling
        major_timestep (int): Hours per hazard map update
        l1_area_geojson (str or Path, optional): GeoJSON file path for L1 adaptation areas
        l1_active_timesteps (list, optional): List of timesteps when L1 is active
        l2_asset_geojson (str or Path, optional): GeoJSON file path for L2 protected assets
        l2_active_timesteps (list, optional): List of timesteps when L2 is active
        l2_asset_depth_red (dict, optional): Legacy L2 format {timestep: [(asset_idx, depth_red), ...]}
        
    Returns:
        dict: Initialized variables including loaded caches and adaptation arrays
    """
    # Use provided config or load default
    if config is None:
        _config = get_config()
    else:
        _config = config

    # Set root directory
    if root_dir is None:
        root_dir = Path.cwd().parent
    else:
        root_dir = Path(root_dir)

    # Create interim directory
    interim_dir = _config['interim_dir']
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Determine hazard directory
    if hazard_maps:
        hazard_dir = Path(hazard_maps[0]).parent
        hazard_dir_name = hazard_dir.name
    else:
        hazard_dir = None
        hazard_dir_name = "unknown"

    # Create output directory
    output_dir = root_dir / 'data' / 'output' / f"output_{hazard_dir_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set default recovery parameters if not provided
    if recovery_parameters is None:
        recovery_parameters = {
            'repair_time_coefficients': [702.72, 3.14, 1.9891],
            'damage_ratio_coefficients': (0.0468, 0.0077),
            'time_step_hours': 1,
            'damage_threshold': 0.001,
            'repair_threshold': 2.0
        }
    elif not isinstance(recovery_parameters, dict):
        raise TypeError("recovery_parameters must be a dictionary or None")

    # Initialize grid-based accessibility if configured
    if _config['simulation_config']['accessibility_model'] is not None:
        print("Initializing grid-based accessibility analysis...")
        grid_hex.initialize_grid_analysis(root_dir)
        
    # Load caches if not provided
    if accessibility_cache is None:
        print("\nLoading accessibility cache...")
        accessibility_cache = load_accessibility_cache(interim_dir, hazard_dir)
        
    if hazard_extraction_cache is None:
        print("Loading hazard extraction cache...")
        hazard_extraction_cache = load_hazard_extraction_cache(interim_dir, hazard_dir)
        
    if overlap_cache is None:
        print("Loading overlap cache...")
        overlap_cache = load_overlap_cache(interim_dir, hazard_dir)
        
    if island_cache is None:
        if 'island' in repair_crew_assignment_method:
            print("Loading island cache...")
            island_cache = load_island_cache(interim_dir, hazard_dir_name)
        else:
            island_cache = {}

    damage_threshold = recovery_parameters['damage_threshold']
    repair_threshold = recovery_parameters['repair_threshold']
    num_assets = len(gdf_assets)
    asset_type = gdf_assets['type'].values

    # Build L1/L2 depth reduction array (with caching)
    depth_reductions = None
    if l1_area_geojson is not None or l2_asset_geojson is not None:
        # Create cache key
        cache_key = (
            str(l1_area_geojson) if l1_area_geojson else 'no_l1',
            tuple(l1_active_timesteps) if l1_active_timesteps else (),
            str(l2_asset_geojson) if l2_asset_geojson else 'no_l2',
            tuple(l2_active_timesteps) if l2_active_timesteps else (),
            len(hazard_maps),
            major_timestep,
            len(gdf_assets)
        )
        
        # Check cache first
        if cache_key in _DEPTH_REDUCTION_CACHE:
            depth_reductions = _DEPTH_REDUCTION_CACHE[cache_key]
            if verbose:
                print("Using cached depth reduction array")
        else:
            if verbose:
                print("\nBuilding adaptation depth reduction arrays...")
            
            n_timesteps = len(hazard_maps) * major_timestep
            
            # Build reduction array using existing function (NO CHANGES)
            depth_reductions_3d = build_l1_l2_reduction_array(
                gdf_assets=gdf_assets,
                l1_area_geojson=l1_area_geojson,
                l1_active_timesteps=l1_active_timesteps,
                l2_asset_geojson=l2_asset_geojson,
                l2_active_timesteps=l2_active_timesteps,
                hazard_maps=hazard_maps,
                major_timestep=major_timestep,
                config=_config,
                verbose=False
            )
            
            if depth_reductions_3d is not None:
                if verbose:
                    print(f"Built adaptation reduction array: shape {depth_reductions_3d.shape}")
                    l1_active = np.count_nonzero(depth_reductions_3d[:, :, 0])
                    l2_active = np.count_nonzero(depth_reductions_3d[:, :, 1])
                    print(f"  L1 active entries: {l1_active}")
                    print(f"  L2 active entries: {l2_active}")
                
                depth_reductions = depth_reductions_3d.sum(axis=2)
                
                if verbose:
                    print(f"Combined into 2D array: shape {depth_reductions.shape}")
                    print(f"  Max combined reduction: {depth_reductions.max():.2f}m")
            else:
                depth_reductions = None
            
            # Store in cache (the 2D version)
            _DEPTH_REDUCTION_CACHE[cache_key] = depth_reductions


    return {
        'config': _config,
        'root_dir': root_dir,
        'interim_dir': interim_dir,
        'hazard_dir': hazard_dir,
        'hazard_dir_name': hazard_dir_name,
        'output_dir': output_dir,
        'recovery_parameters': recovery_parameters,
        'damage_threshold': damage_threshold,
        'repair_threshold': repair_threshold,
        'num_assets': num_assets,
        'asset_type': asset_type,
        'accessibility_cache': accessibility_cache,
        'hazard_extraction_cache': hazard_extraction_cache,
        'overlap_cache': overlap_cache,
        'island_cache': island_cache,
        'fragility_param_k': fragility_param_k,
        'depth_reductions': depth_reductions,
        'l1_area_geojson': l1_area_geojson
    }
def _process_timestep(
    state, gdf_assets, rfids_lengths, timestep, major_timestep, hazard_maps, hazard_dir_name, config,
    accessibility_cache, hazard_extraction_cache, overlap_cache, island_cache, 
    boundary_asset_indices, boundary_islands_rfids, interim_dir, hazard_dir, 
    available_repair_crews, previous_rfids_islands, previous_map_counter, asset_type, 
    num_assets, verbose, flood_threshold, repair_crew_assignment_method, fragility_param_k,
    depth_reductions=None, l1_area_geojson=None  
):
    """Process hazard map and update state/caches if on major timestep."""
    cache_updated = {}
    if timestep % major_timestep == 0:
        available_repair_crews, previous_rfids_islands, previous_map_counter, timestep_cache_updated = _update_hazard_map_states(
            state, gdf_assets, rfids_lengths, timestep, major_timestep, hazard_maps, hazard_dir_name, 
            flood_threshold, repair_crew_assignment_method, config, accessibility_cache, 
            hazard_extraction_cache, overlap_cache, island_cache, boundary_asset_indices, 
            boundary_islands_rfids, interim_dir, hazard_dir, available_repair_crews, 
            previous_rfids_islands, previous_map_counter, asset_type, num_assets, verbose, 
            fragility_param_k=fragility_param_k, 
            depth_reductions=depth_reductions,  
            l1_area_geojson=l1_area_geojson  
        )
        flooded_mask = state.current_hazard_values > flood_threshold
        cache_updated = timestep_cache_updated
    else:
        flooded_mask = state.current_hazard_values > flood_threshold
    return available_repair_crews, previous_rfids_islands, previous_map_counter, flooded_mask, cache_updated

def _assign_repair_crews(
    timestep, available_repair_crews, repair_crews_assigned, accessible, flooded_mask,
    repair_time, island_ids, method, verbose, asset_impact_map=None
):
    """Assign repair crews using the assignment method."""
    return update_repair_crew_assignment_optimized(
        timestep, available_repair_crews, repair_crews_assigned, accessible, flooded_mask,
        repair_time, island_ids, method=method, verbose=verbose, asset_impact_map=asset_impact_map
    )

def _update_repair_progress(state, flooded_mask):
    """Decrement repair_time for assets being repaired."""
    can_repair_mask = state.accessible & ~flooded_mask & state.repair_crews_assigned
    state.repair_time[can_repair_mask] -= 1.0
    state.repair_time[can_repair_mask] = np.maximum(state.repair_time[can_repair_mask], 0.0, out=state.repair_time[can_repair_mask])

def _handle_completed_repairs(state, available_repair_crews, verbose, timestep):
    """Update operational status and release crews for completed repairs."""
    completed_repairs = (state.repair_time == 0.0) & state.repair_crews_assigned
    if np.any(completed_repairs):
        non_operational_completed = completed_repairs & (~state.operational)
        if np.any(non_operational_completed):
            state.operational[non_operational_completed] = True
        state.damage_ratio[completed_repairs] = 0.0
        state.repair_time[completed_repairs] = 0.0
        num_completed_repairs = completed_repairs.sum()
        if available_repair_crews is not None:
            if isinstance(available_repair_crews, dict):
                for asset_idx in np.where(completed_repairs)[0]:
                    asset_island_id = state.island_ids[asset_idx]
                    if asset_island_id in available_repair_crews:
                        available_repair_crews[asset_island_id] += 1
                    else:
                        if available_repair_crews:
                            first_island = list(available_repair_crews.keys())[0]
                            available_repair_crews[first_island] += 1
            else:
                available_repair_crews += num_completed_repairs
        state.repair_crews_assigned[completed_repairs] = False
        if verbose:
            completed_repairs_indices = np.where(completed_repairs)[0]
            print(f"Assets {completed_repairs_indices.tolist()} became operational at timestep {timestep}")

    return available_repair_crews

def _update_unreachable_assets(state, available_repair_crews, flooded_mask, damage_threshold):
    """Update unreachable assets for island-based assignment.
    
    Assets are marked unreachable if they are:
    1. Damaged (damage > threshold), AND
    2. NOT flooded (flooded assets can't be repaired anyway), AND
    3. Either on island_id = -1 OR in an island without crews
    """
    all_island_ids = np.unique(state.island_ids)
    
    # Convert dictionary keys to numpy array
    idle_crew_islands = np.array([island_id for island_id, crew_count in available_repair_crews.items()
                                 if crew_count > 0], dtype=state.island_ids.dtype)
    
    # Get assigned crew islands as numpy array
    if np.any(state.repair_crews_assigned):
        assigned_crew_islands = np.unique(state.island_ids[state.repair_crews_assigned])
    else:
        assigned_crew_islands = np.array([], dtype=state.island_ids.dtype)
    
    # Combine arrays using numpy operations
    if len(idle_crew_islands) > 0 and len(assigned_crew_islands) > 0:
        islands_with_crews = np.unique(np.concatenate([idle_crew_islands, assigned_crew_islands]))
    elif len(idle_crew_islands) > 0:
        islands_with_crews = idle_crew_islands
    elif len(assigned_crew_islands) > 0:
        islands_with_crews = assigned_crew_islands
    else:
        islands_with_crews = np.array([], dtype=state.island_ids.dtype)
    
    # Find islands without crews using numpy's set difference
    islands_without_crews = np.setdiff1d(all_island_ids, islands_with_crews)
    
    in_islands_without_crews = np.isin(state.island_ids, islands_without_crews)
    
    # - It's damaged AND not flooded AND (either on island -1 OR in an island without crews)
    island_minus_one_mask = (state.island_ids == -1)
    
    state.unreachable = (
        (state.damage_ratio > damage_threshold) &
        (~flooded_mask) &
        (island_minus_one_mask | in_islands_without_crews)  # Either -1 OR no crews on island
    )

def _collect_timestep_metrics(
    state, timestep, map_counter, day_counter, num_assets, flooded_mask,
    damage_threshold, repair_threshold
):
    """Collect metrics and asset states for the current timestep."""
    timestep_data = {
        'timestep': timestep,
        'map': map_counter,
        'day': day_counter,
        'asset_id': range(num_assets),
        'damage_ratio': state.damage_ratio.copy(),
        'repair_time': state.repair_time.copy(),
        'operational': state.operational.astype(int).copy(),
        'accessible': state.accessible.astype(int).copy(),
        'unreachable': state.unreachable.astype(int).copy(),
        'flooded': flooded_mask.astype(int).copy(),
        'crew_assigned': state.repair_crews_assigned.astype(int).copy(),
        'hazard_value': state.current_hazard_values.copy(),
        'island_id': state.island_ids.copy() if state.island_ids is not None else np.zeros(num_assets, dtype=int)
    }
    damaged_assets_mask = state.damage_ratio > damage_threshold
    repair_needed_mask = state.repair_time > repair_threshold
    avg_damage_ratio = state.damage_ratio[damaged_assets_mask].mean() if np.any(damaged_assets_mask) else 0.0
    avg_repair_time = state.repair_time[repair_needed_mask].mean() if np.any(repair_needed_mask) else 0.0
    total_repair_backlog = state.repair_time.sum()
    total_damage_ratio = state.damage_ratio.sum()
    metrics = {
        'day': day_counter,
        'map': map_counter,
        'timestep': timestep,
        'operational_count': state.operational.sum(),
        'accessible_count': state.accessible.sum(),
        'unreachable_count': state.unreachable.sum(),
        'flooded_count': flooded_mask.sum(),
        'damaged_count': damaged_assets_mask.sum(),
        'crews_assigned_count': state.repair_crews_assigned.sum(),
        'avg_damage_ratio': avg_damage_ratio,
        'avg_repair_time': avg_repair_time,
        'total_repair_backlog': total_repair_backlog,
        'total_damage_ratio': total_damage_ratio
    }
    return timestep_data, metrics

def _save_config_file(output_dir, root_dir, execution_id):
    """Save configuration file to output directory."""
    try:
        config_output_file = output_dir / f'log_config_{execution_id}.txt' if execution_id else output_dir / 'log_config.txt'
        config_source_file = root_dir / 'config.py'
        copyfile(config_source_file, config_output_file)
        print(f"Saved simulation configuration to {config_output_file}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")

def simulate_asset_damage_recovery_access_breakdown(
    gdf_assets,
    hazard_maps,
    number_repair_crews=5,
    repair_crew_assignment_method='islands',
    flood_threshold=0.2,
    recovery_parameters=None,
    root_dir=None,
    verbose=False,
    timestep_output=False,
    execution_id=None,
    config=None,
    major_timestep=24,
    accessibility_cache=None,
    hazard_extraction_cache=None,
    overlap_cache=None,
    island_cache=None,
    fragility_param_k=None,
    asset_population_map=None,
    asset_to_lu=None,
    l1_area_geojson=None,
    l1_active_timesteps=None,
    l2_asset_geojson=None,
    l2_active_timesteps=None
    ):
    """
    Runs a time-stepped simulation of asset damage and recovery, considering hazard exposure, accessibility, and repair crew assignment.
  
    Args:
        gdf_assets (GeoDataFrame): Asset geometries and types.
        hazard_maps (list[str or Path]): List of hazard map file paths (rasters).
        number_repair_crews (int or dict): Number of available repair crews (global int or per-island dict).
        repair_crew_assignment_method (str): Crew assignment strategy ('random', 'lowest repair time', 'highest repair time', 'island', etc.).
        flood_threshold (float): Hazard value threshold for flooding.
        recovery_parameters (dict, optional): Recovery model parameters (damage/repair coefficients, thresholds).
        root_dir (str or Path, optional): Root directory for data and cache storage.
        verbose (bool): If True, prints detailed simulation progress.
        timestep_output (bool): If True, collects detailed asset states at each timestep.
        execution_id (str, optional): Unique identifier for output file naming.
        config (dict, optional): Simulation configuration dictionary.
        major_timestep (int): Number of hours per hazard map update.
        accessibility_cache (dict): Cache for accessibility results.
        hazard_extraction_cache (dict): Cache for hazard extraction results.
        overlap_cache (dict): Cache for overlap results.
        island_cache (dict): Cache for island analysis results.
        fragility_param_k (float, optional): Fragility parameter for operational status modeling.
        asset_population_map (dict, optional): Mapping of asset indices to population impact values.
        asset_to_lu (dict, optional): Mapping of asset indices to land-use impact values.
        l1_area_geojson (GeoDataFrame or str, optional): GeoJSON defining L1 adaptation areas.
        l1_active_timesteps (list[int], optional): Timesteps when L1 adaptation is
            active (in hours).
        l2_asset_depth_red (dict, optional): Mapping of timesteps to lists of tuples
            (asset_index, depth_reduction) for L2 adaptation measures.

    Returns:
        tuple:
            - list: Simulation results. Contains a tuple of (simulation ID, summary metrics per timestep, detailed asset states per timestep).
            - dict: Final asset states, with keys:
                'operational': np.ndarray, operational status of each asset.
                'hazard_value': np.ndarray, last hazard value for each asset.
                'damage_ratio': np.ndarray, last damage ratio for each asset.
                'repair_time': np.ndarray, last remaining repair time for each asset.
                'accessible': np.ndarray, last accessibility status for each asset.
                'repair_crews_assigned': np.ndarray, last crew assignment status for each asset.
            - dict: Updated caches for the caller to save.
    """
    # --- Initialization ---
    init = _initialize_simulation(
        gdf_assets, hazard_maps, recovery_parameters, root_dir, config, repair_crew_assignment_method,
        accessibility_cache, hazard_extraction_cache, overlap_cache, island_cache, 
        fragility_param_k, major_timestep,
        l1_area_geojson, l1_active_timesteps, 
        l2_asset_geojson, l2_active_timesteps,
        verbose=verbose  
    )
    _config = init['config']
    root_dir = init['root_dir']
    interim_dir = init['interim_dir']
    hazard_dir = init['hazard_dir']
    hazard_dir_name = init['hazard_dir_name']
    output_dir = init['output_dir']
    damage_threshold = init['damage_threshold']
    repair_threshold = init['repair_threshold']
    num_assets = init['num_assets']
    asset_type = init['asset_type']
    fragility_param_k = init['fragility_param_k']
    depth_reductions = init['depth_reductions']  
    l1_area_geojson = init['l1_area_geojson']  

    # Get caches from initialization
    accessibility_cache = init['accessibility_cache']
    hazard_extraction_cache = init['hazard_extraction_cache']
    overlap_cache = init['overlap_cache']
    island_cache = init['island_cache']

    # Results tracking for this simulation
    results = []
    timestep_results = []    

    # Initialize blank state variables
    state = SimulationState(gdf_assets, num_assets)
    # previous_islands = None
    previous_rfids_islands = None
    previous_map_counter = None
    available_repair_crews = number_repair_crews
    
    island_method_active = 'island' in repair_crew_assignment_method
    if island_method_active:
        access_rfids, boundary_asset_indices, boundary_islands_rfids, rfids_lengths = match_assets_access(
            gdf_assets, 
            hazard_threshold=flood_threshold, 
            hazard_column='EV0_ma',
            config=_config, 
            island_cache=island_cache, 
            cache_dir=interim_dir, 
            hazard_dir=hazard_dir,
            l1_area_geojson=l1_area_geojson  
        )
        gdf_assets['access_rfid'] = access_rfids
        if isinstance(number_repair_crews, int):
            available_repair_crews = {0: number_repair_crews}
            # print(f"Initialized island method with {number_repair_crews} crews in temporary island 0")
        else:
            available_repair_crews = number_repair_crews

    else:
        available_repair_crews = number_repair_crews

    if 'monetary' in repair_crew_assignment_method and asset_to_lu is not None:
        asset_impact_map = {
            aid: sum(area * rate for (lu, area, rate) in asset_to_lu[aid])
            for aid in asset_to_lu
        }
    elif 'population' in repair_crew_assignment_method and asset_population_map is not None:
        asset_impact_map = asset_population_map
    else:
        asset_impact_map = None

    timesteps = np.arange(0, len(hazard_maps) * major_timestep)
    cache_updated = {}  # Track cache updates throughout the simulation

    for timestep in timesteps:
        day_counter = timestep // 24
        map_counter = int(timestep / major_timestep)
                
        # 1. Process hazard map and update state if on major timestep
        available_repair_crews, previous_rfids_islands, previous_map_counter, flooded_mask, timestep_cache_updated = _process_timestep(
            state, gdf_assets, rfids_lengths, timestep, major_timestep, hazard_maps, hazard_dir_name, _config,
            accessibility_cache, hazard_extraction_cache, overlap_cache, island_cache, 
            boundary_asset_indices, boundary_islands_rfids, interim_dir, hazard_dir, 
            available_repair_crews, previous_rfids_islands, previous_map_counter, asset_type, 
            num_assets, verbose, flood_threshold, repair_crew_assignment_method, fragility_param_k,
            depth_reductions=depth_reductions,
            l1_area_geojson=l1_area_geojson
        )
        
        # Merge cache updates
        for cache_name, cache_content in timestep_cache_updated.items():
            cache_updated[cache_name] = cache_content

        # 2. Repair crew assignment
        available_repair_crews, state.repair_crews_assigned = _assign_repair_crews(
            timestep, available_repair_crews, state.repair_crews_assigned, state.accessible,
            flooded_mask, state.repair_time, state.island_ids, repair_crew_assignment_method, verbose, asset_impact_map=asset_impact_map
        )
        
        # 3. Update repair progress
        _update_repair_progress(state, flooded_mask)
        
        # 4. Handle completed repairs
        available_repair_crews = _handle_completed_repairs(
            state, available_repair_crews, verbose, timestep
        )

        # 5. Update unreachable assets (island method)
        if island_method_active:
            _update_unreachable_assets(state, available_repair_crews, flooded_mask, damage_threshold)

        # 6. Collect timestep metrics
        if timestep_output:
            timestep_data, metrics = _collect_timestep_metrics(
                state, timestep, map_counter, day_counter, num_assets, flooded_mask,
                damage_threshold, repair_threshold
            )
            timestep_results.append(timestep_data)
            results.append(metrics)
        else:
            # If not collecting detailed timestep output, still need metrics
            damaged_assets_mask = state.damage_ratio > damage_threshold
            repair_needed_mask = state.repair_time > repair_threshold
            avg_damage_ratio = state.damage_ratio[damaged_assets_mask].mean() if np.any(damaged_assets_mask) else 0.0
            avg_repair_time = state.repair_time[repair_needed_mask].mean() if np.any(repair_needed_mask) else 0.0
            total_repair_backlog = state.repair_time.sum()
            total_damage_ratio = state.damage_ratio.sum()
            
            results.append({
                'day': day_counter,
                'map': map_counter,
                'timestep': timestep,
                'operational_count': state.operational.sum(),
                'accessible_count': state.accessible.sum(),
                'unreachable_count': state.unreachable.sum(),
                'flooded_count': (state.current_hazard_values > flood_threshold).sum(),
                'damaged_count': damaged_assets_mask.sum(),
                'crews_assigned_count': state.repair_crews_assigned.sum(),
                'avg_damage_ratio': avg_damage_ratio,
                'avg_repair_time': avg_repair_time,
                'total_repair_backlog': total_repair_backlog,  
                'total_damage_ratio': total_damage_ratio       
            })

        # 7. Print end-of-day summary if verbose
        if timestep % 24 == 23 and verbose:
            print(f"Day {day_counter} summary: {state.operational.sum()}/{num_assets} operational, "
                  f"{state.accessible.sum()} accessible, {state.unreachable.sum()} unreachable damaged assets, {flooded_mask.sum()} flooded")

    # 8. Save config (optional)
    #_save_config_file(output_dir, root_dir, execution_id)

    # Create output list format consistent with original function
    all_results = [(1, results, timestep_results)]

    # Return results, final state, and updated caches
    return all_results, {
        'operational': state.operational,
        'hazard_value': state.current_hazard_values,
        'damage_ratio': state.damage_ratio,
        'repair_time': state.repair_time,
        'accessible': state.accessible,
        'repair_crews_assigned': state.repair_crews_assigned
    }, cache_updated

