"""
Functions to run the damage and recovery simulation.
"""

# Import hazard extraction method from config
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.adaptation import build_l1_l2_reduction_array
from src.caching import (
    create_accessibility_cache_key,
    create_island_cache_key,
    get_asset_centroid_hash,
    load_accessibility_cache,
    load_hazard_extraction_cache,
    load_island_cache,
    load_overlap_cache,
)
from src.damage_recovery import (
    default_damage_ratio_function,
    default_fragility_function,
    default_repair_time_function,
    vectorized_damage_ratio_solver,
)
from src.dependency_evaluator import (
    activate_delayed_trigger_waits,
    clear_completed_delayed_triggers,
    evaluate_dependencies,
    evaluate_dependencies_from_graph,
)
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.island_analysis import (
    match_assets_access,
    match_island_ids_assets,
    update_repair_crew_islands,
)
from src.recovery_scheduler import (
    all_selected_waits_cleared,
    decrement_recovery_wait_vectors,
    initialize_recovery_wait_vectors,
)
from src.utils import build_service_area_map_from_rules

sys.path.append(str(Path(__file__).parent.parent))
from shutil import copyfile

from config import get_config


class SimulationState:
    def __init__(self, gdf_assets, num_assets):
        self.previous_map_counter = None
        self.damage_ratio = np.zeros(num_assets, dtype=np.float64)
        self.accessible = np.ones(num_assets, dtype=bool)
        self.unreachable = np.zeros(num_assets, dtype=bool)
        self.operational = np.ones(num_assets, dtype=bool)
        self.repair_crews_assigned = np.zeros(num_assets, dtype=bool)
        self.current_hazard_values = np.zeros(num_assets, dtype=np.float64)
        self.island_ids = np.zeros(num_assets, dtype=int)
        self.recovery_wait_vectors = initialize_recovery_wait_vectors(num_assets)
        self.recovery_delay_active = {}
        self.dependency_report = {}
        self.simulation_warnings = []
        # self.temp_gdf = gdf_assets[['type', 'geometry']].copy()

def _update_hazard_map_states(
    state, gdf_assets, rfids_lengths, timestep, major_timestep, hazard_maps, haz_dir_name, 
    flood_threshold, repair_crew_assignment_method, _config, accessibility_cache, 
    hazard_extraction_cache, overlap_cache, island_cache, boundary_asset_indices, 
    boundary_islands_rfids, interim_dir, hazard_dir, available_repair_crews, 
    previous_rfids_islands, previous_map_counter, asset_type, num_assets, verbose, 
    fragility_param_k=None, depth_reductions=None, l1_area_geojson=None, l1_active_timesteps=None,
    repair_crews_by_asset_type=None,
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
        l1_area_geojson (str or GeoDataFrame or None): Path to L1 adaptation GeoJSON or GeoDataFrame, if applicable
        l1_active_timesteps (list or None): List of timesteps when L1 adaptation is active, or None for all timesteps

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
            l1_area_geojson=l1_area_geojson,
            l1_active_timesteps=l1_active_timesteps
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
                    l1_area_geojson=l1_area_geojson,
                    l1_active_timesteps=l1_active_timesteps  
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
                l1_area_geojson=l1_area_geojson,
                l1_active_timesteps=l1_active_timesteps
            )
            if repair_crews_by_asset_type is not None:
                for pool in repair_crews_by_asset_type["pools"]:
                    pool["available"] = update_repair_crew_islands(
                        pool["available"],
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
                        l1_area_geojson=l1_area_geojson,
                        l1_active_timesteps=l1_active_timesteps,
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
        fragility_result = default_fragility_function(
            hazard_subset,
            asset_type_subset,
            k=fragility_param_k,
            major_timestep=major_timestep,
            fragility_models=_config['recovery_parameters'].get('fragility_models'),
        )
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
        repair_time = state.recovery_wait_vectors["repair_time"]
        repair_time[flooded_mask] = default_repair_time_function(
            state.damage_ratio[flooded_mask], repair_time_coefficients
        )
        if verbose:
            try:
                damage_count = newly_damaged_mask.sum()
                print(f"  New damage at timestep {timestep}: {damage_count} assets")
                if damage_count > 0:
                    print(f"  Damage ratios: {state.damage_ratio[newly_damaged_mask].min():.3f} to {state.damage_ratio[newly_damaged_mask].max():.3f}")
                    print(f"  Repair times: {repair_time[newly_damaged_mask].min():.1f} to {repair_time[newly_damaged_mask].max():.1f} hours")
            except Exception as e:
                print(f"  Error occurred while logging damage information: {e}, {timestep}")

    # For assets needing repair, solve for current damage ratio excluding assets under repair threshold
    repair_time = state.recovery_wait_vectors["repair_time"]
    recalc_repair_mask = (repair_time > repair_threshold)
    if np.any(recalc_repair_mask):
        repair_times_under_repair = repair_time[recalc_repair_mask]
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
                    newly_assigned_crews = min(newly_assigned_crews, crew_count)
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
            elif 'impact' in method:
                sorted_indices = np.argsort(-np.array([asset_impact_map.get(idx, 0) for idx in repairable_assets_indices]))
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


def _normalize_asset_type_group_key(group_key):
    """Normalize an asset-type group key to a frozenset of strings."""
    if isinstance(group_key, str):
        normalized = [group_key]
    elif isinstance(group_key, (tuple, list, set, frozenset, np.ndarray, pd.Index)):
        normalized = [str(v) for v in group_key if v is not None]
    else:
        raise TypeError(
            "repair_crews_by_asset_type keys must be a string asset type or an iterable of asset types."
        )

    normalized = [v.strip() for v in normalized if str(v).strip()]
    if not normalized:
        raise ValueError("Asset-type group keys cannot be empty.")
    return frozenset(normalized)


def _normalize_repair_crews_by_asset_type_config(repair_crews_by_asset_type):
    """Normalize repair-crew pool config to an internal grouped-pool structure.

    Supported inputs:
      - ``{'hospital': 2}`` (single-type pool)
      - ``{('ls', 'msls'): 5, 'hospital': 2}`` (grouped + single pools)
      - ``[({'ls', 'msls'}, 5), ('hospital', 2)]`` (list of (group, crews))
      - ``[{'asset_types': ['ls', 'msls'], 'count': 5}, {'asset_types': ['hospital'], 'count': 2}]``
    """
    if not repair_crews_by_asset_type:
        return None

    if isinstance(repair_crews_by_asset_type, dict) and "pools" in repair_crews_by_asset_type:
        # Already normalized internal state.
        return repair_crews_by_asset_type

    if isinstance(repair_crews_by_asset_type, dict):
        raw_items = list(repair_crews_by_asset_type.items())
    elif isinstance(repair_crews_by_asset_type, (list, tuple)):
        raw_items = []
        for entry in repair_crews_by_asset_type:
            if isinstance(entry, dict):
                if "asset_types" not in entry:
                    raise ValueError("Pool dictionaries must contain an 'asset_types' key.")
                if "count" in entry:
                    crew_count = entry["count"]
                elif "crews" in entry:
                    crew_count = entry["crews"]
                else:
                    raise ValueError("Pool dictionaries must contain 'count' (or 'crews').")
                raw_items.append((entry["asset_types"], crew_count))
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                raw_items.append((entry[0], entry[1]))
            else:
                raise TypeError(
                    "List-style repair_crews_by_asset_type entries must be (asset_types, crews) pairs "
                    "or dictionaries with asset_types/count."
                )
    else:
        raise TypeError(
            "repair_crews_by_asset_type must be a dict, list of pairs, or list of pool dictionaries."
        )

    pools = []
    asset_type_to_pool = {}
    for raw_key, raw_count in raw_items:
        asset_type_group = _normalize_asset_type_group_key(raw_key)
        crew_count = int(raw_count)
        if crew_count < 0:
            raise ValueError("Crew counts in repair_crews_by_asset_type must be non-negative.")

        pool_index = len(pools)
        pools.append(
            {
                "asset_types": asset_type_group,
                "available": crew_count,
            }
        )

        for atype in asset_type_group:
            if atype in asset_type_to_pool:
                prev_group = pools[asset_type_to_pool[atype]]["asset_types"]
                raise ValueError(
                    f"Asset type '{atype}' appears in multiple crew pools: {set(prev_group)} and {set(asset_type_group)}."
                )
            asset_type_to_pool[atype] = pool_index

    return {
        "pools": pools,
        "asset_type_to_pool": asset_type_to_pool,
    }

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

    dependency_config = _config.get('dependency_parameters', {})
    knowledge_graph_rules = dependency_config.get('knowledge_graph', None) or []
    if knowledge_graph_rules and dependency_config.get('service_area_map', None) is None:
        dependency_config['service_area_map'] = build_service_area_map_from_rules(
            gdf_assets,
            knowledge_graph_rules,
        )

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
    depth_reductions=None, l1_area_geojson=None, l1_active_timesteps=None,
    repair_crews_by_asset_type=None,
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
            l1_area_geojson=l1_area_geojson, 
            l1_active_timesteps=l1_active_timesteps,
            repair_crews_by_asset_type=repair_crews_by_asset_type,
        )
        flooded_mask = state.current_hazard_values > flood_threshold
        cache_updated = timestep_cache_updated
    else:
        flooded_mask = state.current_hazard_values > flood_threshold
    return available_repair_crews, previous_rfids_islands, previous_map_counter, flooded_mask, cache_updated

def _assign_repair_crews(
    timestep, available_repair_crews, repair_crews_assigned, accessible, flooded_mask,
    repair_time, island_ids, method, verbose, asset_impact_map=None,
    asset_type=None, repair_crews_by_asset_type=None
):
    """Assign repair crews using the assignment method.

    When *repair_crews_by_asset_type* is provided the function runs two
    integrated crew-assignment passes:

    1. **Grouped pass** – for each configured asset-type group in
       *repair_crews_by_asset_type* the assignment is run only against assets
       in that group, drawing from that group's dedicated crew pool.
    2. **Default pass** – for all remaining asset types, the existing
       *available_repair_crews* mechanism is used unchanged.
    """
    if repair_crews_by_asset_type is not None and asset_type is not None:
        pool_state = _normalize_repair_crews_by_asset_type_config(repair_crews_by_asset_type)
        if pool_state is not None:
            # Grouped dedicated pools retain the same island-aware availability
            # representation as the default pool.
            for pool in pool_state["pools"]:
                pool_asset_types = tuple(pool["asset_types"])
                type_mask = np.isin(asset_type, list(pool_asset_types))
                if not np.any(type_mask):
                    continue
                pool["available"], repair_crews_assigned = (
                    update_repair_crew_assignment_optimized(
                        timestep,
                        pool["available"],
                        repair_crews_assigned,
                        accessible & type_mask,
                        flooded_mask,
                        repair_time,
                        island_ids,
                        method=method,
                        verbose=verbose,
                        asset_impact_map=asset_impact_map,
                    )
                )

            # Run default pool assignment for asset types not covered by any grouped pool.
            handled_types = set(pool_state["asset_type_to_pool"].keys())
            if asset_type is not None and handled_types:
                default_mask = ~np.isin(asset_type, list(handled_types))
            else:
                default_mask = np.ones(len(repair_crews_assigned), dtype=bool)
            if np.any(default_mask):
                available_repair_crews, repair_crews_assigned = update_repair_crew_assignment_optimized(
                    timestep, available_repair_crews, repair_crews_assigned,
                    accessible & default_mask, flooded_mask, repair_time, island_ids,
                    method=method, verbose=verbose, asset_impact_map=asset_impact_map,
                )
            return available_repair_crews, repair_crews_assigned

        # fall through if config normalizes to None
        if asset_type is not None:
            default_mask = np.ones(len(repair_crews_assigned), dtype=bool)
        else:
            default_mask = np.ones(len(repair_crews_assigned), dtype=bool)
        if np.any(default_mask):
            available_repair_crews, repair_crews_assigned = update_repair_crew_assignment_optimized(
                timestep, available_repair_crews, repair_crews_assigned,
                accessible & default_mask, flooded_mask, repair_time, island_ids,
                method=method, verbose=verbose, asset_impact_map=asset_impact_map,
            )
        return available_repair_crews, repair_crews_assigned

    return update_repair_crew_assignment_optimized(
        timestep, available_repair_crews, repair_crews_assigned, accessible, flooded_mask,
        repair_time, island_ids, method=method, verbose=verbose, asset_impact_map=asset_impact_map
    )

def _handle_completed_repairs(state, available_repair_crews, verbose, timestep,
                              asset_type=None, repair_crews_by_asset_type=None):
    """Clear completed repair states and release crews for completed repairs."""
    completed_repairs = (
        all_selected_waits_cleared(
            state.recovery_wait_vectors,
            wait_vector_names=("repair_time",),
        )
        & state.repair_crews_assigned
    )
    if np.any(completed_repairs):
        state.damage_ratio[completed_repairs] = 0.0
        state.recovery_wait_vectors["repair_time"][completed_repairs] = 0.0
        num_completed_repairs = completed_repairs.sum()

        # Return crews to the appropriate pool(s).
        if repair_crews_by_asset_type is not None and asset_type is not None:
            pool_state = _normalize_repair_crews_by_asset_type_config(repair_crews_by_asset_type)
            type_to_pool = pool_state["asset_type_to_pool"] if pool_state else {}
            pools = pool_state["pools"] if pool_state else []
            # Per-type (or grouped-type) pool: return each completed asset's crew to its pool.
            for asset_idx in np.where(completed_repairs)[0]:
                atype = str(asset_type[asset_idx])
                pool_idx = type_to_pool.get(atype)
                if pool_idx is not None:
                    pool_available = pools[pool_idx]["available"]
                    if isinstance(pool_available, dict):
                        asset_island_id = int(state.island_ids[asset_idx])
                        pool_available[asset_island_id] = (
                            pool_available.get(asset_island_id, 0) + 1
                        )
                    else:
                        pools[pool_idx]["available"] += 1
                    continue
                # Fallback to the default island/global pool for types not in grouped pools.
                if available_repair_crews is not None:
                    if isinstance(available_repair_crews, dict):
                        asset_island_id = int(state.island_ids[asset_idx])
                        available_repair_crews[asset_island_id] = (
                            available_repair_crews.get(asset_island_id, 0) + 1
                        )
                    else:
                        available_repair_crews += 1
        elif available_repair_crews is not None:
            if isinstance(available_repair_crews, dict):
                for asset_idx in np.where(completed_repairs)[0]:
                    asset_island_id = int(state.island_ids[asset_idx])
                    available_repair_crews[asset_island_id] = (
                        available_repair_crews.get(asset_island_id, 0) + 1
                    )
            else:
                available_repair_crews += num_completed_repairs
        state.operational[completed_repairs] = True
        state.repair_crews_assigned[completed_repairs] = False
        if verbose:
            completed_repairs_indices = np.where(completed_repairs)[0]
            print(f"Assets {completed_repairs_indices.tolist()} completed repair at timestep {timestep}")

    return available_repair_crews

def _update_repair_progress(state, flooded_mask, elapsed_time=1.0):
    """Advance crew-based repair and crew-independent recovery countdowns."""
    can_repair_mask = state.accessible & ~flooded_mask & state.repair_crews_assigned
    decrement_recovery_wait_vectors(
        state.recovery_wait_vectors,
        elapsed_time=elapsed_time,
        active_masks={"repair_time": can_repair_mask},
    )

def _update_operational_state(state, asset_type, flooded_mask, config, repair_threshold, knowledge_graph=None):
    """Evaluate dependency rules each timestep using current state vectors.

    When a knowledge graph is configured this function performs two passes:

    1. **Restore pass** – re-enables assets whose return-to-operational
       trigger is now satisfied (no longer flooded and repair condition met).
    2. **Block pass** – suppresses assets that still do not meet conditions
       (currently flooded or repair not yet complete per the rule trigger).

    The two-pass design means the knowledge graph is the single authority for
    both directions of operational-state change.  The ``state.operational``
    array is updated in-place via assignment.

    Args:
        state: Current :class:`SimulationState`.
        asset_type: String array of asset types.
        flooded_mask: Boolean array; ``True`` where hazard exceeds threshold.
        config: Simulation configuration dict.
        repair_threshold: Repair-time value below which an asset is considered
            not in need of formal repair (used on the legacy path only).
        knowledge_graph: Pre-built :class:`DependencyKnowledgeGraph` instance,
            or ``None`` to build from ``config['dependency_parameters']`` each
            call (legacy behaviour; use the pre-built instance for performance).
    """
    dependency_config = config.get('dependency_parameters', {})
    kg_config = dependency_config.get('knowledge_graph', None) or []
    state.dependency_report = {}

    if kg_config:
        # Graph-aware path: per-pair rules from the knowledge graph.
        if knowledge_graph is None:
            from src.dependency_knowledge_graph import DependencyKnowledgeGraph
            knowledge_graph = DependencyKnowledgeGraph.from_config(kg_config)
        hazard_type = dependency_config.get('hazard_type', 'flooding')
        service_area_map = dependency_config.get('service_area_map', None)
        # evaluate_dependencies_from_graph internally calls restore_operational_from_graph
        # first (restore pass) then applies the block pass, so state.operational is
        # updated correctly in both directions.
        state.operational, state.dependency_report = evaluate_dependencies_from_graph(
            state.operational,
            asset_type,
            hazard_type,
            knowledge_graph,
            flooded_mask=flooded_mask,
            repair_time=state.recovery_wait_vectors["repair_time"],
            wait_vectors=state.recovery_wait_vectors,
            service_area_map=service_area_map,
            return_report=True,
        )
        clear_completed_delayed_triggers(
            state.operational,
            state.recovery_wait_vectors,
            state.recovery_delay_active,
        )
    else:
        # Legacy flat-flags path (backwards compatible).
        state.operational, state.dependency_report = evaluate_dependencies(
            state.operational,
            asset_type,
            hazard_values=state.current_hazard_values,
            flooded_mask=flooded_mask,
            repair_time=state.recovery_wait_vectors["repair_time"],
            repair_threshold=repair_threshold,
            dependency_map=dependency_config.get('dependency_map'),
            area_dependencies=dependency_config.get('area_dependencies'),
            pairwise_dependencies=dependency_config.get('pairwise_dependencies'),
            enable_default_rules=dependency_config.get('enable_default_rules', True),
            require_repair_for_operational=dependency_config.get('require_repair_for_operational', False),
            return_report=True,
        )

    warning = state.dependency_report.get("warning")
    if warning and warning not in state.simulation_warnings:
        state.simulation_warnings.append(warning)

def _update_unreachable_assets(
    state,
    available_repair_crews,
    flooded_mask,
    damage_threshold,
    *,
    asset_type=None,
    repair_crews_by_asset_type=None,
):
    """Update unreachable assets for island-based assignment.
    
    Assets are marked unreachable if they are:
    1. Damaged (damage > threshold), AND
    2. NOT flooded (flooded assets can't be repaired anyway), AND
    3. Either on island_id = -1 OR in an island without crews
    """
    has_available_crew = np.zeros(len(state.island_ids), dtype=bool)
    grouped_types = set()

    if repair_crews_by_asset_type is not None and asset_type is not None:
        pool_state = _normalize_repair_crews_by_asset_type_config(
            repair_crews_by_asset_type
        )
        grouped_types = set(pool_state["asset_type_to_pool"])
        for pool in pool_state["pools"]:
            type_mask = np.isin(asset_type, list(pool["asset_types"]))
            pool_available = pool["available"]
            if isinstance(pool_available, dict):
                crew_islands = [
                    island_id
                    for island_id, crew_count in pool_available.items()
                    if crew_count > 0
                ]
                has_available_crew |= type_mask & np.isin(
                    state.island_ids, crew_islands
                )
            elif pool_available > 0:
                has_available_crew |= type_mask

    default_type_mask = (
        ~np.isin(asset_type, list(grouped_types))
        if asset_type is not None and grouped_types
        else np.ones(len(state.island_ids), dtype=bool)
    )
    if isinstance(available_repair_crews, dict):
        default_crew_islands = [
            island_id
            for island_id, crew_count in available_repair_crews.items()
            if crew_count > 0
        ]
        has_available_crew |= default_type_mask & np.isin(
            state.island_ids, default_crew_islands
        )
    elif available_repair_crews is None or available_repair_crews > 0:
        has_available_crew |= default_type_mask

    has_available_crew |= state.repair_crews_assigned
    state.unreachable = (
        (state.damage_ratio > damage_threshold)
        & ~flooded_mask
        & ((state.island_ids == -1) | ~has_available_crew)
    )

def _collect_timestep_metrics(
    state, timestep, map_counter, day_counter, num_assets, flooded_mask,
    damage_threshold, repair_threshold
):
    """Collect metrics and asset states for the current timestep."""
    repair_time = state.recovery_wait_vectors["repair_time"]
    timestep_data = {
        'timestep': timestep,
        'map': map_counter,
        'day': day_counter,
        'asset_id': range(num_assets),
        'damage_ratio': state.damage_ratio.copy(),
        'repair_time': repair_time.copy(),
        'operational': state.operational.astype(int).copy(),
        'accessible': state.accessible.astype(int).copy(),
        'unreachable': state.unreachable.astype(int).copy(),
        'flooded': flooded_mask.astype(int).copy(),
        'crew_assigned': state.repair_crews_assigned.astype(int).copy(),
        'hazard_value': state.current_hazard_values.copy(),
        'island_id': state.island_ids.copy() if state.island_ids is not None else np.zeros(num_assets, dtype=int)
    }
    damaged_assets_mask = state.damage_ratio > damage_threshold
    repair_needed_mask = repair_time > repair_threshold
    avg_damage_ratio = state.damage_ratio[damaged_assets_mask].mean() if np.any(damaged_assets_mask) else 0.0
    avg_repair_time = repair_time[repair_needed_mask].mean() if np.any(repair_needed_mask) else 0.0
    total_repair_backlog = repair_time.sum()
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
    metrics.update({
        'dependency_blocked_count': state.dependency_report.get('blocked_count', 0),
        'dependency_active_rule_count': state.dependency_report.get('active_rule_count', len(state.dependency_report.get('active_rules', []))),
        'dependency_warning': state.dependency_report.get('warning'),
    })
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
    l2_active_timesteps=None,
    repair_crews_by_asset_type=None
    ):
    """
    Runs a time-stepped simulation of asset damage and recovery, considering hazard exposure, accessibility, and repair crew assignment.
  
    Args:
        gdf_assets (GeoDataFrame): Asset geometries and types.
        hazard_maps (list[str or Path]): List of hazard map file paths (rasters).
        number_repair_crews (int or dict): Default repair-crew pool (global int or per-island dict),
            used for asset types not covered by ``repair_crews_by_asset_type``.
        repair_crew_assignment_method (str): Crew assignment strategy ('random', 'lowest repair time', 'highest repair time', 'island', etc.).
        flood_threshold (float): Hazard value threshold for flooding.
        recovery_parameters (dict, optional): Recovery model parameters (damage/repair coefficients, thresholds).
        root_dir (str or Path, optional): Root directory for data and cache storage.
        verbose (bool): If True, prints detailed simulation progress.
        repair_crews_by_asset_type (dict/list, optional): Dedicated grouped crew pools by
            asset type. Supports:
            ``{'hospital': 2}``,
            ``{('ls', 'msls'): 5, 'hospital': 2}``,
            or list-style entries such as
            ``[({'ls', 'msls'}, 5), ('hospital', 2)]``.
            Any asset types not covered by these grouped pools fall back to
            *number_repair_crews*.
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
        rfids_lengths = None
        boundary_asset_indices = None
        boundary_islands_rfids = None

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

    # Normalize grouped type-pool configuration to mutable internal state.
    repair_crews_by_asset_type = _normalize_repair_crews_by_asset_type_config(repair_crews_by_asset_type)

    # Pre-build the knowledge graph once to avoid reconstructing it every timestep.
    _knowledge_graph = None
    _dep_config = _config.get('dependency_parameters', {})
    _kg_config = _dep_config.get('knowledge_graph', None) or []
    if _kg_config:
        from src.dependency_knowledge_graph import DependencyKnowledgeGraph
        _knowledge_graph = DependencyKnowledgeGraph.from_config(_kg_config)

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
            l1_area_geojson=l1_area_geojson,
            l1_active_timesteps=l1_active_timesteps,
            repair_crews_by_asset_type=repair_crews_by_asset_type,
        )
        
        # Merge cache updates
        for cache_name, cache_content in timestep_cache_updated.items():
            cache_updated[cache_name] = cache_content

        if _knowledge_graph is not None:
            activate_delayed_trigger_waits(
                state.operational,
                asset_type,
                _dep_config.get('hazard_type', 'flooding'),
                _knowledge_graph,
                flooded_mask=flooded_mask,
                wait_vectors=state.recovery_wait_vectors,
                active_masks=state.recovery_delay_active,
            )

        # 2. Repair crew assignment
        available_repair_crews, state.repair_crews_assigned = _assign_repair_crews(
            timestep, available_repair_crews, state.repair_crews_assigned, state.accessible,
            flooded_mask, state.recovery_wait_vectors["repair_time"], state.island_ids, repair_crew_assignment_method, verbose, asset_impact_map=asset_impact_map,
            asset_type=asset_type, repair_crews_by_asset_type=repair_crews_by_asset_type
        )
        
        # 3. Update repair progress
        _update_repair_progress(
            state,
            flooded_mask,
            elapsed_time=_config['recovery_parameters'].get(
                'time_step_hours', 1.0
            ),
        )
        
        # 4. Handle completed repairs
        available_repair_crews = _handle_completed_repairs(
            state, available_repair_crews, verbose, timestep,
            asset_type=asset_type, repair_crews_by_asset_type=repair_crews_by_asset_type
        )

        # 5. Evaluate dependencies using current repair/hazard state
        _update_operational_state(state, asset_type, flooded_mask, _config, repair_threshold, knowledge_graph=_knowledge_graph)

        # 6. Update unreachable assets (island method)
        if island_method_active:
            _update_unreachable_assets(
                state,
                available_repair_crews,
                flooded_mask,
                damage_threshold,
                asset_type=asset_type,
                repair_crews_by_asset_type=repair_crews_by_asset_type,
            )

        # 7. Collect timestep metrics
        if timestep_output:
            timestep_data, metrics = _collect_timestep_metrics(
                state, timestep, map_counter, day_counter, num_assets, flooded_mask,
                damage_threshold, repair_threshold
            )
            timestep_results.append(timestep_data)
            results.append(metrics)
        else:
            # If not collecting detailed timestep output, still need metrics
            repair_time = state.recovery_wait_vectors["repair_time"]
            damaged_assets_mask = state.damage_ratio > damage_threshold
            repair_needed_mask = repair_time > repair_threshold
            avg_damage_ratio = state.damage_ratio[damaged_assets_mask].mean() if np.any(damaged_assets_mask) else 0.0
            avg_repair_time = repair_time[repair_needed_mask].mean() if np.any(repair_needed_mask) else 0.0
            total_repair_backlog = repair_time.sum()
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
                'total_damage_ratio': total_damage_ratio,
                'dependency_blocked_count': state.dependency_report.get('blocked_count', 0),
                'dependency_active_rule_count': state.dependency_report.get('active_rule_count', len(state.dependency_report.get('active_rules', []))),
                'dependency_warning': state.dependency_report.get('warning'),
            })

        # 8. Print end-of-day summary if verbose
        if timestep % 24 == 23 and verbose:
            print(f"Day {day_counter} summary: {state.operational.sum()}/{num_assets} operational, "
                  f"{state.accessible.sum()} accessible, {state.unreachable.sum()} unreachable damaged assets, {flooded_mask.sum()} flooded")

    # 9. Save config (optional)
    #_save_config_file(output_dir, root_dir, execution_id)

    # Create output list format consistent with original function
    if results and state.simulation_warnings:
        results[-1]['simulation_warnings'] = list(state.simulation_warnings)
    all_results = [(1, results, timestep_results)]

    # Return results, final state, and updated caches
    return all_results, {
        'operational': state.operational,
        'hazard_value': state.current_hazard_values,
        'damage_ratio': state.damage_ratio,
        'repair_time': state.recovery_wait_vectors["repair_time"].copy(),
        'recovery_wait_vectors': {
            name: values.copy()
            for name, values in state.recovery_wait_vectors.items()
        },
        'accessible': state.accessible,
        'repair_crews_assigned': state.repair_crews_assigned,
        'dependency_report': state.dependency_report,
        'simulation_warnings': list(state.simulation_warnings),
    }, cache_updated
