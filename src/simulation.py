"""
Functions to run the damage and recovery simulation.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.caching import load_accessibility_cache, load_overlap_cache, load_hazard_extraction_cache, save_accessibility_cache, save_overlap_cache, save_hazard_extraction_cache
from src.island_analysis import initialize_island_cache, match_island_ids_assets, update_repair_crew_islands_with_overlap_cached
from src.hazard_analysis_electricity import find_hazard_value_at_points_optimized
from src.damage_recovery import default_damage_ratio_function, default_repair_time_function, vectorized_damage_ratio_solver, default_fragility_function
from src.caching import create_accessibility_cache_key
import src.grid_based_accessibility_hex as grid_hex

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
        self.temp_gdf = gdf_assets.copy()

def _update_hazard_map_states(state, timestep, major_timestep, hazard_maps, haz_dir_name,
                              _config, accessibility_cache, hazard_extraction_cache, overlap_cache, island_cache,
                              hazard_dir, available_repair_crews, previous_islands, previous_map_counter,
                              asset_type, num_assets, verbose):
    """
    Update the simulation states that depend on the hazard map (only on major timesteps)

    Args:
        state (SimulationState): Current state of the simulation
        timestep (int): Current timestep in the simulation
        major_timestep (int): Major timestep interval for hazard map updates
        hazard_maps (list): List of paths to hazard map files (raster format)
        haz_dir_name (str): Name of the hazard directory (for caching)
        _config (dict): Configuration settings for the simulation
        accessibility_cache (dict): Cache for accessibility results
        hazard_extraction_cache (dict): Cache for hazard extraction results
        overlap_cache (dict): Cache for overlap results
        island_cache (dict): Cache for island analysis results
        hazard_dir (Path): Path to the directory containing hazard maps
        available_repair_crews (int or dict): Number of available repair crews or a dictionary with island IDs as keys
        previous_islands (GeoDataFrame or None): Previous island assignments, if any
        previous_map_counter (int or None): Previous map counter, if any
        asset_type (np.ndarray): Array of asset types corresponding to each asset
        num_assets (int): Total number of assets in the simulation
        verbose (bool): If True, print detailed simulation information

    Returns:
        tuple: Updated available_repair_crews, previous_islands, previous_map_counter, cache_updated (dictionary of updated caches)
    """

    repair_threshold = _config['recovery_parameters'].get('repair_threshold', 2.0)
    flood_threshold = _config['simulation_config'].get('flood_threshold', 0.2)
    repair_crew_assignment_method = _config['simulation_config'].get('repair_crew_assignment_method', 'islands')
    damage_ratio_coefficients = _config['recovery_parameters'].get('damage_ratio_coefficients', (0.0468, 0.0077))
    repair_time_coefficients = _config['recovery_parameters'].get('repair_time_coefficients', [702.72, 3.14, 1.9891])
    cache_updated = {}

    map_counter = int(timestep / major_timestep)
    if map_counter >= len(hazard_maps):
        print(f"No more hazard maps available at timestep {timestep}, ending simulation.")
        return available_repair_crews, previous_islands, previous_map_counter, cache_updated

    hazard_map = hazard_maps[map_counter]
    haz_col_str = f'EV{map_counter}_ma'

    if verbose:
        print(f"\n=== Processing timestep {timestep} (map {map_counter}) ===")

    # Update hazard values
    state.temp_gdf = find_hazard_value_at_points_optimized(
        hazard_map,
        state.temp_gdf,
        map_counter,
        extraction_method=_config['analysis_config']['hazard_extraction_method'],
        hazard_cache=hazard_extraction_cache,
        hazard_dir=hazard_dir, 
        _config=_config
    )
    haz_val_str = f'hazard_value_{map_counter}'

    if haz_val_str in state.temp_gdf.columns:
        state.current_hazard_values = state.temp_gdf[haz_val_str].fillna(0.0).values
    else:
        state.current_hazard_values = state.temp_gdf[haz_col_str].fillna(0.0).values

    # Island-based crew management
    if 'island' in repair_crew_assignment_method:
        cache_key = f"{flood_threshold}_{haz_col_str}"
        if cache_key in island_cache:
            island_data = island_cache[cache_key]
            state.island_ids = island_data['island_ids']
            dissolved_roads = island_data['dissolved_roads']
            if verbose:
                print(f"Using cached islands for {cache_key}")
        else:
            print(f"Cache miss for {cache_key}, computing islands on the fly...")
            try:
                temp_gdf_for_islands = state.temp_gdf.copy()
                temp_gdf_for_islands, dissolved_roads = match_island_ids_assets(
                    temp_gdf_for_islands,
                    hazard_threshold=flood_threshold,
                    hazard_column=haz_col_str,
                    config=_config,
                )
                state.island_ids = temp_gdf_for_islands['island_id'].values
                island_data = {
                    'hazard_map': str(hazard_map),
                    'threshold': flood_threshold,
                    'island_ids': state.island_ids,
                    'dissolved_roads': dissolved_roads,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'computed_on_demand',
                    'method': 'match_island_ids_assets'
                }
                island_cache[cache_key] = island_data
                cache_updated['island_cache'] = island_cache
                print(f"Successfully computed and cached islands for {cache_key}")

            except Exception as e:
                print(f"Error computing islands for {cache_key}: {e}")
                print("Falling back to simple island assignment")
                state.island_ids = np.ones(num_assets, dtype=int)
                dissolved_roads = None

        if dissolved_roads is not None:
            available_repair_crews = update_repair_crew_islands_with_overlap_cached(
                available_repair_crews,
                state.island_ids,
                dissolved_roads,
                previous_islands,
                current_map=map_counter,
                previous_map=previous_map_counter,
                hazard_threshold=flood_threshold,
                overlap_cache=overlap_cache,
                hazard_dir=hazard_dir,
                _config=_config,
            )
            previous_map_counter = map_counter
            previous_islands = dissolved_roads.copy()
            state.temp_gdf['island_id'] = state.island_ids
        else:
            print(f"No dissolved roads available for {cache_key}, using global crew assignment")

    # Mask of assets flooded above threshold
    flooded_mask = state.current_hazard_values > flood_threshold

    # Apply fragility to assets that are not currently under repair
    not_under_repair_mask = ~state.repair_crews_assigned
    assets_to_evaluate = flooded_mask & not_under_repair_mask & state.operational

    # Update operational status based on fragility for assets above threshold
    if np.any(assets_to_evaluate):
        fragility_operational = np.ones_like(state.operational, dtype=bool)
        hazard_subset = state.current_hazard_values[assets_to_evaluate]
        asset_type_subset = asset_type[assets_to_evaluate]
        fragility_result = default_fragility_function(hazard_subset, asset_type_subset)
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
                print(f"  New damage at timestep {timestep}: {newly_damaged_mask.sum()} assets")
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
                assets_copy = state.temp_gdf.copy()
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

    return available_repair_crews, previous_islands, previous_map_counter, cache_updated


def simulate_asset_damage_recovery_access_optimized(
    gdf_assets, 
    hazard_maps, 
    number_repair_crews=15, 
    repair_crew_assignment_method='lowest repair time',
    flood_threshold=0.2, 
    recovery_parameters=None,
    root_dir=None,
    verbose=False,
    timestep_output=True, 
    execution_id=None,
    config=None,
    major_timestep=24,
    iterations=1
):
    """
    Simulate the damage and recovery of assets over time, considering hazard maps, accessibility, and repair crew assignment.

    Args:
        gdf_assets (GeoDataFrame): Asset geometries and types.
        hazard_maps (list): List of hazard map file paths.
        number_repair_crews (int): Number of available repair crews (global or per island).
        repair_crew_assignment_method (str): Crew assignment strategy ('random', 'lowest repair time', 'highest repair time', 'island', etc.).
        flood_threshold (float): Hazard value threshold for flooding.
        recovery_parameters (dict, optional): Recovery model parameters (damage/repair coefficients, thresholds).
        root_dir (str or Path, optional): Root directory for data and cache storage.
        verbose (bool): If True, prints detailed simulation progress.
        timestep_output (bool): If True, collects detailed asset states at each timestep.
        execution_id (str, optional): Unique identifier for output file naming.
        config (dict, optional): Simulation configuration dictionary.
        major_timestep (int): Number of hours per hazard map update.
        iterations (int): Number of simulation runs (for stochastic analysis).

    Returns:
        list: Simulation results for each iteration (summary and timestep details).
        dict: Final asset states (operational, hazard_value, damage_ratio, repair_time, accessible, repair_crews_assigned).
    """
    if config is None:
        _config = get_config()
    else:
        _config = config

    # Initialize paths and caching
    if root_dir is None:
        root_dir = Path.cwd().parent
    else:
        root_dir = Path(root_dir)
    
    # interim_dir = root_dir / 'data' / 'interim'
    interim_dir = _config['interim_dir']
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine hazard directory from first hazard map for cache naming
    if hazard_maps:
        hazard_dir = Path(hazard_maps[0]).parent
        hazard_dir_name = hazard_dir.name
        print(f"Using hazard directory for cache naming: {hazard_dir_name}")
    else:
        hazard_dir = None
        hazard_dir_name = "unknown"

    # Setup output directory
    output_dir = root_dir / 'data' / 'output' / f"output_{hazard_dir_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up default recovery parameters if not provided
    if recovery_parameters is None:
        recovery_parameters = {
            'repair_time_coefficients': [702.72, 3.14, 1.9891],
            'damage_ratio_coefficients': (0.0468, 0.0077),  # Added damage ratio coefficients
            'time_step_hours': 1,
            'damage_threshold': 0.001,
            'repair_threshold': 2.0  # Default threshold for repairable damage ratio
        }    

    # Initialize accessibility model if specified
    if _config['simulation_config']['accessibility_model'] is not None:
        print("Initializing grid-based accessibility analysis...")
        grid_hex.initialize_grid_analysis(root_dir)

    # Load caches with hazard directory context
    print("\nLoading simulation caches...")
    accessibility_cache = load_accessibility_cache(interim_dir, hazard_dir)
    hazard_extraction_cache = load_hazard_extraction_cache(interim_dir, hazard_dir)
    overlap_cache = load_overlap_cache(interim_dir, hazard_dir)
    if 'island' in repair_crew_assignment_method:
        # Add hazard directory name to island cache filename
        island_cache = initialize_island_cache(interim_dir, hazard_dir_name)
    else:
        island_cache = {}    
        
    damage_threshold = recovery_parameters['damage_threshold']
    repair_threshold = recovery_parameters['repair_threshold']

    num_assets = len(gdf_assets)
    asset_type = gdf_assets['type'].values

    # Initialize results storage
    all_results = []

    # Run simulation for specified number of iterations
    for i in tqdm(range(iterations), desc="Simulation Iterations", total = iterations):
        if verbose:
            print(f"\n--- Starting simulation iteration {i+1} of {iterations} ---")

        # Results tracking for this iteration
        results = []
        timestep_results = []    

        # Initialize blank state variables
        state = SimulationState(gdf_assets, num_assets)
        previous_islands = None
        previous_map_counter = None
        available_repair_crews = number_repair_crews
        
        island_method_active = 'island' in repair_crew_assignment_method
        if verbose and island_method_active:
            print(f"Island-based method '{repair_crew_assignment_method}' will be used for crew assignment")
        
        timesteps = np.arange(0, len(hazard_maps) * major_timestep)  # if 4 maps per day, major timestep is 6; if 1 map per day, major timestep is 24. Total hours is number of maps times hours covered per map

        for timestep in timesteps:
            day_counter = timestep // 24 # Day counter depends on the timestep
            map_counter = int(timestep / major_timestep)
                    
            # Every x (24 by default) hour-timesteps, process the hazard map for that period
            if timestep % major_timestep == 0:

                available_repair_crews, previous_islands, previous_map_counter, cache_updated = _update_hazard_map_states(state, timestep, major_timestep, hazard_maps, hazard_dir_name,
                                _config, accessibility_cache, hazard_extraction_cache, overlap_cache, island_cache,
                                hazard_dir, available_repair_crews, previous_islands, previous_map_counter,
                                asset_type, num_assets, verbose)
                flooded_mask = state.current_hazard_values > flood_threshold


            # Repair crew assignment
            available_repair_crews, state.repair_crews_assigned = update_repair_crew_assignment_optimized(
                timestep, 
                available_repair_crews, 
                state.repair_crews_assigned, 
                state.accessible, 
                state.current_hazard_values > flood_threshold,  # flooded_mask
                state.repair_time, 
                state.island_ids, 
                method=repair_crew_assignment_method, 
                verbose=verbose
            )
            
            # For each timestep, decrement repair_time if accessible, not flooded, and with repair crews assigned
            can_repair_mask = state.accessible & ~flooded_mask & state.repair_crews_assigned
            #state.repair_time[can_repair_mask] = np.maximum(state.repair_time[can_repair_mask] - 1.0, 0.0)
            state.repair_time[can_repair_mask] -= 1.0
            state.repair_time[can_repair_mask] = np.maximum(state.repair_time[can_repair_mask], 0.0, out=state.repair_time[can_repair_mask])
            
            # Check for completed repairs
            completed_repairs = (state.repair_time == 0.0) & state.repair_crews_assigned

            if np.any(completed_repairs):
                # Only force operational=True for assets that are non-operational but completed repair
                non_operational_completed = completed_repairs & (~state.operational)
                
                if np.any(non_operational_completed):
                    state.operational[non_operational_completed] = True

                # For ALL completed repairs, reset damage and release crews
                state.damage_ratio[completed_repairs] = 0.0
                state.repair_time[completed_repairs] = 0.0

                # Release repair crews
                num_completed_repairs = completed_repairs.sum()
                if available_repair_crews is not None:
                    if isinstance(available_repair_crews, dict):
                        # Island-based crew management
                        for asset_idx in np.where(completed_repairs)[0]:
                            asset_island_id = state.island_ids[asset_idx]
                            if asset_island_id in available_repair_crews:
                                available_repair_crews[asset_island_id] += 1
                            else:
                                # Fallback to first available island
                                if available_repair_crews:
                                    first_island = list(available_repair_crews.keys())[0]
                                    available_repair_crews[first_island] += 1
                    else:
                        # Simple integer crew management
                        available_repair_crews += num_completed_repairs

                state.repair_crews_assigned[completed_repairs] = False

                if verbose:
                    completed_repairs_indices = np.where(completed_repairs)[0]
                    print(f"Assets {completed_repairs_indices.tolist()} became operational at timestep {timestep}")

            if island_method_active:
                # Islands with zero available crews and no crews assigned
                all_island_ids = np.unique(state.island_ids)
                idle_crew_islands = [island_id for island_id, crew_count in available_repair_crews.items() if crew_count > 0]
                assigned_crew_islands = set(state.island_ids[state.repair_crews_assigned])
                islands_with_crews = set(idle_crew_islands) | assigned_crew_islands
                islands_without_crews = set(all_island_ids) - islands_with_crews
                in_islands_without_crews = np.isin(state.island_ids, list(islands_without_crews))
                state.unreachable = (
                    (state.damage_ratio > damage_threshold) &
                    (~flooded_mask) &
                    in_islands_without_crews
                )
                # unreachable_count = unreachable_mask.sum()

            if timestep_output:
                timestep_data = {
                    'timestep': timestep,
                    'map': map_counter,
                    'day': day_counter,
                    'asset_id': range(num_assets),
                    'damage_ratio': state.damage_ratio.copy(),
                    'repair_time': state.repair_time.copy(),
                    'operational': state.operational.astype(int),
                    'accessible': state.accessible.astype(int),
                    'unreachable': state.unreachable.astype(int),
                    'flooded': flooded_mask.astype(int),
                    'crew_assigned': state.repair_crews_assigned.astype(int),
                    'hazard_value': state.current_hazard_values.copy(),
                    'island_id': state.island_ids.copy() if state.island_ids is not None else np.zeros(num_assets, dtype=int)
                }
                
                timestep_results.append(timestep_data)

            # Calculate output metrics            
            # Only calculate averages for assets that actually have damage/repair time 
            damaged_assets_mask = state.damage_ratio > damage_threshold
            repair_needed_mask = state.repair_time > repair_threshold

            if np.any(damaged_assets_mask):
                avg_damage_ratio = state.damage_ratio[damaged_assets_mask].mean()
            else:
                avg_damage_ratio = 0.0
                
            if np.any(repair_needed_mask):
                avg_repair_time = state.repair_time[repair_needed_mask].mean()
            else:
                avg_repair_time = 0.0

            total_repair_backlog = state.repair_time.sum()  # Sum of all repair hours remaining
            total_damage_ratio = state.damage_ratio.sum()   # Sum of all damage ratios
            
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

            if timestep % 24 == 23:  # End of day
                if verbose:
                    print(f"Day {day_counter} summary: {state.operational.sum()}/{num_assets} operational, "
                        f"{state.accessible.sum()} accessible, {state.unreachable.sum()} unreachable damaged assets, {(state.current_hazard_values > flood_threshold).sum()} flooded")

        all_results.append((i+1, results, timestep_results))

        # Update caches if they were modified during the simulation
        if 'accessibility_cache' in cache_updated.keys():
            accessibility_cache = cache_updated['accessibility_cache']
            print("\nSaving optimization caches...")
            save_accessibility_cache(accessibility_cache, interim_dir, hazard_dir)            
        if 'island_cache' in cache_updated.keys():
            island_cache = cache_updated['island_cache']
            try:
                cache_dir = interim_dir / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Create island cache filename with hazard directory context
                hazard_dir_name = Path(hazard_dir).name if hazard_dir else "unknown"
                
                island_cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
                
                with open(island_cache_file, 'wb') as f:
                    pickle.dump(cache_updated['island_cache'], f)

                print(f"Saved updated island cache with {len(cache_updated['island_cache'])} entries to {island_cache_file}")

            except Exception as e:
                print(f"Warning: Could not save island cache: {e}")

    try:
        config_output_file = output_dir / f'log_config_{execution_id}.txt' if execution_id else output_dir / 'log_config.txt'
        config_source_file = root_dir / 'config.py'
        copyfile(config_source_file, config_output_file)
        print(f"Saved simulation configuration to {config_output_file}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.columns = ['Simulation ID', 'Results summary', 'Timestep Results']

    return all_results, {
        'operational': state.operational,
        'hazard_value': state.current_hazard_values,
        'damage_ratio': state.damage_ratio,
        'repair_time': state.repair_time,
        'accessible': state.accessible,
        'repair_crews_assigned': state.repair_crews_assigned
    }

def update_repair_crew_assignment_optimized(timestep, available_repair_crews, repair_crews_assigned, 
                                           accessible, flooded_mask, repair_time, island_ids=None, method=None, verbose=False):    
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

    Returns:
        tuple: (updated available_repair_crews, updated repair_crews_assigned)
            available_repair_crews: updated int or dict after assignment
            repair_crews_assigned: updated boolean array
    """
    
    # Check if available repair crews is None, meaning no constraints
    if available_repair_crews is None:
        # If no constraints, assign all repairable assets
        repair_crews_assigned[:] = True
        if verbose:
            print("No constraints on repair crews, all assets are assigned for repair.")
        return available_repair_crews, repair_crews_assigned 
    
    # If available_repair_crews is a dictionary, it means we have island-based constraints
    if isinstance(available_repair_crews, dict):
        # Assign repair crews based on island IDs
        for island_id, crew_count in available_repair_crews.items():
            if crew_count > 0:  # for each island with available crews
                if island_ids is not None: # If assets have island IDs, only assign to assets in that island
                    island_mask = (island_ids == island_id)
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
        print("No available repair crews, cannot assign any assets for repair.")
        return available_repair_crews, repair_crews_assigned 
    
    if available_repair_crews > 0:
        print(f"Available repair crews: {available_repair_crews}, proceeding with assignment...")
        
        repairable_assets = accessible & ~flooded_mask & (repair_time > 0) & ~repair_crews_assigned

        if not 'islands' in str(method):  # Convert method to string to handle None case
            # If there are more repair crews than assets needing assignment, assign all repairable assets
            if repairable_assets.sum() <= available_repair_crews:
                newly_assigned_crews = repairable_assets.sum()
                repair_crews_assigned[repairable_assets] = True
                available_repair_crews -= newly_assigned_crews

                print(f"Assigned {newly_assigned_crews} repair crews to all repairable assets; There remain {available_repair_crews} available repair crews")
                
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