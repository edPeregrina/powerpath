"""
Functions to run the damage and recovery simulation.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

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

# Get the hazard extraction method from config
# _config = get_config()
# HAZARD_EXTRACTION_METHOD = _config['analysis_config']['hazard_extraction_method']

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

    #TODO: add repair threshold and any other parameters that come from config and take them out of parameters
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

    # Only apply fragility to assets that are NOT currently under repair
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
    Run the asset damage recovery simulation with accessibility and repair crew assignment.
    
    This version includes optimizations to prevent computational spikes in visualizations:
    - State change tracking with appropriate tolerances
    - Vectorized operations to reduce calculation peaks
    - Efficient masking and conditional logic
    - Smart averaging calculations only for relevant assets

    Args:
        gdf_assets (GeoDataFrame): GeoDataFrame containing asset geometries and initial states
        hazard_maps (list): List of paths to hazard map files (raster format)
        number_repair_crews (int): Total number of repair crews available for the simulation
        repair_crew_assignment_method (str): Method for assigning repair crews:
            - 'random': Random assignment
            - 'lowest repair time': Assign to assets with lowest remaining repair time
            - 'highest repair time': Assign to assets with highest remaining repair time
            - 'island': Assign based on pre-computed island assignments
            - 'islands lowest repair time': Assign to assets with lowest remaining repair time within islands
            - 'islands highest repair time': Assign to assets with highest remaining repair time within islands
        flood_threshold (float): Threshold for flooding to determine asset accessibility
        recovery_parameters (dict, optional): Dictionary containing recovery parameters:
            - 'repair_time_coefficients': Coefficients for calculating repair time based on damage ratio
            - 'damage_ratio_coefficients': Coefficients for calculating damage ratio based on hazard values
            - 'time_step_hours': Time step in hours for the simulation
            - 'damage_threshold': Threshold for damage ratio to consider asset damaged
            - 'repair_threshold': Threshold for repairable damage ratio
        root_dir (str or Path, optional): Root directory for data storage and caching
        verbose (bool): If True, print detailed simulation information
        timestep_output (bool): If True, output detailed asset states at each timestep to Parquet file
        execution_id (str, optional): Unique identifier for this simulation run. Used for output file naming.
        config (dict, optional): Configuration settings for the simulation

        
    Returns:
        dataframe: DataFrame containing simulation results by timestep
        dict: Simulation results including asset states, repair crew assignments, and accessibility
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

    if _config['simulation_config']['accessibility_model'] is not None:
        # Initialize accessibility model analysis once if using any model
        print("Initializing grid-based accessibility analysis...")
        grid_hex.initialize_grid_analysis(root_dir)

    # Load caches with hazard directory context
    print("\nLoading simulation caches...")
    accessibility_cache = load_accessibility_cache(interim_dir, hazard_dir)
    hazard_extraction_cache = load_hazard_extraction_cache(interim_dir, hazard_dir)
    overlap_cache = load_overlap_cache(interim_dir, hazard_dir)
    # island_cache = {}
    if 'island' in repair_crew_assignment_method:
        # Add hazard directory name to island cache filename
        island_cache = initialize_island_cache(interim_dir, hazard_dir_name)
    else:
        island_cache = {}    
        
    # repair_time_coefficients = recovery_parameters['repair_time_coefficients']
    # damage_ratio_coefficients = recovery_parameters.get('damage_ratio_coefficients', (0.0468, 0.0077))
    damage_threshold = recovery_parameters['damage_threshold']
    repair_threshold = recovery_parameters['repair_threshold']

    num_assets = len(gdf_assets)
    asset_type = gdf_assets['type'].values

    # Initialize results storage
    all_results = []

    # Run simulation for specified number of iterations
    for i in range(iterations):
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

                # map_counter = int(timestep / major_timestep)
                # if map_counter >= len(hazard_maps):
                #     print(f"No more hazard maps available at timestep {timestep}, ending simulation.")
                #     break

                # hazard_map = hazard_maps[map_counter]
                # haz_col_str = f'EV{map_counter}_ma'

                # if verbose:
                #     print(f"\n=== Processing timestep {timestep} (day {day_counter}, map {map_counter}) ===")

                # # Update hazard values
                # temp_gdf = find_hazard_value_at_points_optimized(
                #     hazard_map,
                #     temp_gdf,
                #     map_counter,
                #     extraction_method=_config['analysis_config']['hazard_extraction_method'],
                #     hazard_cache=hazard_extraction_cache,
                #     hazard_dir=hazard_dir
                # )
                # haz_val_str = f'hazard_value_{map_counter}'

                # # TODO: get rid of one of the two columns
                # if haz_val_str in temp_gdf.columns:
                #     current_hazard_values = temp_gdf[haz_val_str].fillna(0.0).values
                # else:
                #     current_hazard_values = temp_gdf[haz_col_str].fillna(0.0).values
                
                # # Island-based crew management
                # if 'island' in repair_crew_assignment_method:
                #     cache_key = f"{flood_threshold}_{haz_col_str}"
                    
                #     if cache_key in island_cache:
                #         # Use cached data
                #         island_data = island_cache[cache_key]
                #         island_ids = island_data['island_ids']
                #         dissolved_roads = island_data['dissolved_roads']
                        
                #         if verbose:
                #             print(f"Using cached islands for {cache_key}")
                #     else:
                #         # Cache miss - compute islands on the fly using proper function
                #         print(f"Cache miss for {cache_key}, computing islands on the fly...")
                        
                #         try:
                #             temp_gdf_for_islands = gdf_assets.copy()
                #             temp_gdf_for_islands, dissolved_roads = match_island_ids_assets(
                #                 temp_gdf_for_islands, 
                #                 hazard_threshold=flood_threshold, 
                #                 hazard_column=haz_col_str,
                #                 config=_config,
                    
                #             )
                #             island_ids = temp_gdf_for_islands['island_id'].values
                            
                #             # Cache the computed result for future use
                #             island_data = {
                #                 'hazard_map': str(hazard_map),
                #                 'threshold': flood_threshold,
                #                 'island_ids': island_ids,
                #                 'dissolved_roads': dissolved_roads,
                #                 'timestamp': datetime.now().isoformat(),
                #                 'status': 'computed_on_demand',
                #                 'method': 'match_island_ids_assets'
                #             }
                #             island_cache[cache_key] = island_data
                            
                #             print(f"Successfully computed and cached islands for {cache_key}")
                            
                #         except Exception as e:
                #             print(f"Error computing islands for {cache_key}: {e}")
                #             print("Falling back to simple island assignment")
                #             # Fallback to simple assignment
                #             island_ids = np.ones(len(gdf_assets), dtype=int)
                #             dissolved_roads = None
                    
                #     # Continue with crew distribution logic
                #     if dissolved_roads is not None:
                #         available_repair_crews = update_repair_crew_islands_with_overlap_cached(
                #             available_repair_crews, 
                #             island_ids, 
                #             dissolved_roads, 
                #             previous_islands if 'previous_islands' in locals() else None,
                #             current_map=map_counter,
                #             previous_map=previous_map_counter,
                #             hazard_threshold=flood_threshold,
                #             overlap_cache=overlap_cache,
                #             hazard_dir=hazard_dir
                #         )

                #         # Update previous map counter
                #         previous_map_counter = map_counter

                #         # Store current islands for next iteration
                #         previous_islands = dissolved_roads.copy()
                        
                #         # Assign island_ids to temp_gdf for later use
                #         if 'temp_gdf' not in locals():
                #             temp_gdf = gdf_assets.copy()
                #         temp_gdf['island_id'] = island_ids
                #     else:
                #         print(f"No dissolved roads available for {cache_key}, using global crew assignment")
                
                # # Mask of assets flooded above threshold  
                # flooded_mask = current_hazard_values > flood_threshold 

                # # Only apply fragility to assets that are NOT currently under repair
                # not_under_repair_mask = ~repair_crews_assigned
                # assets_to_evaluate = flooded_mask & not_under_repair_mask & operational
                
                # if np.any(assets_to_evaluate):
                #     fragility_operational = np.ones_like(operational, dtype=bool)  # Start with all operational
                    
                #     # Only evaluate assets not under repair
                #     hazard_subset = current_hazard_values[assets_to_evaluate]
                #     asset_type_subset = asset_type[assets_to_evaluate]
                    
                #     fragility_result = default_fragility_function(hazard_subset, asset_type_subset)
                #     fragility_operational[assets_to_evaluate] = fragility_result.astype(bool)
                    
                #     # Update operational status, but preserve assets under repair
                #     operational = np.minimum(operational, fragility_operational)
                
                # # Update damage ratio for assets flooded above threshold this timestep
                # if np.any(flooded_mask):
                #     dr_new = default_damage_ratio_function(current_hazard_values[flooded_mask], damage_ratio_coefficients)
                    
                #     # Track which assets are getting new damage this timestep
                #     newly_damaged_mask = np.zeros_like(flooded_mask, dtype=bool)
                #     flooded_indices = np.where(flooded_mask)[0]
                    
                #     # Check which flooded assets have increased damage
                #     new_damage_check = dr_new > damage_ratio[flooded_mask]
                #     newly_damaged_mask[flooded_indices] = new_damage_check
                    
                #     # Update damage ratios (keep maximum damage)
                #     damage_ratio[flooded_mask] = np.maximum(damage_ratio[flooded_mask], dr_new)

                #     # Update repair times for assets with latest damage ratios
                #     repair_time[flooded_mask] = default_repair_time_function(
                #         damage_ratio[flooded_mask], repair_time_coefficients
                #     )
                    
                #     if verbose: 
                #         try:
                #             print(f"  New damage at timestep {timestep}: {newly_damaged_mask.sum()} assets")
                #             print(f"  Damage ratios: {damage_ratio[newly_damaged_mask].min():.3f} to {damage_ratio[newly_damaged_mask].max():.3f}")
                #             print(f"  Repair times: {repair_time[newly_damaged_mask].min():.1f} to {repair_time[newly_damaged_mask].max():.1f} hours")
                #         except Exception as e:
                #             print(f"  Error occurred while logging damage information: {e}, {timestep}")

                # # For assets needing repair, solve for current damage ratio excluding assets under repair threshold
                # recalc_repair_mask = (repair_time > repair_threshold)
                # if np.any(recalc_repair_mask):
                #     repair_times_under_repair = repair_time[recalc_repair_mask]
                #     damage_ratios_from_repair = vectorized_damage_ratio_solver(
                #         repair_times_under_repair, repair_time_coefficients
                #     )
                    
                #     damage_ratio[recalc_repair_mask] = damage_ratios_from_repair
                    
                # # Daily accessibility update 
                # accessibility_cache_key = create_accessibility_cache_key(map_counter, flood_threshold, hazard_dir, accessibility_model=_config['simulation_config']['accessibility_model'])

                # if accessibility_cache_key in accessibility_cache:
                #     accessible = accessibility_cache[accessibility_cache_key]
                #     if verbose:
                #         print(f"Using cached accessibility for map {map_counter} (hazard dir: {hazard_dir_name})")
                # else:
                #     try:
                #         # accessibility_result = grid_hex.accessibility_model(
                #         #     gdf_assets.geometry, 
                #         #     hazard_map, 
                #         #     current_hazard_values,
                #         #     verbose=verbose,
                #         #     day_string=day_counter_str,
                #         #     project_root=root_dir
                #         # )
                #         accessibility_result = accessible # Defaulting to accessible, to use only islands logic
                #         accessible = np.array(accessibility_result, dtype=bool)
                #         accessibility_cache[accessibility_cache_key] = accessible
                        
                #         if verbose:
                #             print(f"Accessibility updated for timestep {timestep} (map {map_counter})")
                #             print(f"Accessible assets: {accessible.sum()} out of {num_assets}")
                #     except Exception as e:
                #         print(f"Warning: Accessibility model failed: {e}")
                #         print("Keeping current accessibility status")

            # Island-based crew assignment logic - simplified check since islands are already computed above
            # if island_method_active:
            #     # Check if we still need to initialize island assignments (backup case)
            #     if isinstance(available_repair_crews, int):
            #         if verbose:
            #             print(f"Backup: Initializing island assignments for {available_repair_crews} crews")
                    
            #         temp_gdf, dissolved_roads = match_island_ids_assets(
            #             temp_gdf,  
            #             hazard_threshold=flood_threshold, 
            #             hazard_column=haz_col_str,
            #             config=_config
            #         )
            #         island_ids = temp_gdf['island_id'].values
                    
            #         if dissolved_roads is not None and len(dissolved_roads) > 0:
            #             # Use overlap-based crew redistribution to convert int to dict
            #             available_repair_crews = update_repair_crew_islands_with_overlap_cached(
            #                 available_repair_crews,
            #                 state.island_ids, 
            #                 dissolved_roads, 
            #                 previous_islands if 'previous_islands' in locals() else None,
            #                 current_map=map_counter,
            #                 previous_map=previous_map_counter,
            #                 hazard_threshold=flood_threshold,
            #                 overlap_cache=overlap_cache,
            #                 hazard_dir=hazard_dir,
            #                 verbose=verbose
            #             )
                        
            #             # Store current islands for next iteration
            #             previous_islands = dissolved_roads.copy()
                        
            #             if verbose:
            #                 print(f"Distributed crews across {len(dissolved_roads)} islands: {available_repair_crews}")
            #         else:
            #             if verbose:
            #                 print("No dissolved roads found, using global crew assignment")

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
            state.repair_time[can_repair_mask] = np.maximum(state.repair_time[can_repair_mask] - 1.0, 0.0)
            
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
                unreachable = (
                    (state.damage_ratio > damage_threshold) &
                    (~flooded_mask) &
                    in_islands_without_crews
                )
                # unreachable_count = unreachable_mask.sum()

            # Write timestep data to parquet buffer if requested

            if timestep_output:
                # # Generate output filename based on execution_id
                # if execution_id:
                #     timestep_output_file = output_dir / f'timestep_output_{execution_id}.parquet'
                # else:
                #     timestep_output_file = output_dir / f'timestep_output.parquet'
                
                # # Remove existing file if it exists (for clean start)
                # if timestep_output_file.exists():
                #     print(f"Warning! Timestep output file already exists: {timestep_output_file}")
                
                # if verbose:
                #     print(f"Timestep output will be written to: {timestep_output_file}")

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
                # timestep_df = pd.DataFrame(timestep_data)
            
            # Record results at every timestep 
            # Only calculate averages for assets that actually have damage/repair time to avoid computational spikes
            damaged_assets_mask = state.damage_ratio > damage_threshold
            repair_needed_mask = state.repair_time > repair_threshold

            # Calculate output metrics
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
                # 'iteration': i + 1,
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

    
    # Save caches for next run with hazard directory context
    # if 'accessibility_cache' in cache_updated.keys():
    #     accessibility_cache = cache_updated['accessibility_cache']
    #     print("\nSaving optimization caches...")
    #     save_accessibility_cache(accessibility_cache, interim_dir, hazard_dir)
    
    # if hazard_extraction_cache:
    #     save_hazard_extraction_cache(hazard_extraction_cache, interim_dir, hazard_dir)
    
    # if overlap_cache:
    #     save_overlap_cache(overlap_cache, interim_dir, hazard_dir)
    
    # Save island cache if it was updated
    # if 'island_cache' in cache_updated.keys():
    #     try:
    #         cache_dir = interim_dir / "cache"
    #         cache_dir.mkdir(parents=True, exist_ok=True)
            
    #         # Create island cache filename with hazard directory context
    #         hazard_dir_name = Path(hazard_dir).name if hazard_dir else "unknown"
            
    #         island_cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
            
    #         with open(island_cache_file, 'wb') as f:
    #             pickle.dump(cache_updated['island_cache'], f)

    #         print(f"Saved updated island cache with {len(cache_updated['island_cache'])} entries to {island_cache_file}")

    #     except Exception as e:
    #         print(f"Warning: Could not save island cache: {e}")
    
    # Create results DataFrame
    # results_df = pd.DataFrame(results)
    
    # Copy of config.py file as text to log_config_executionid.txt
    try:
        config_output_file = output_dir / f'log_config_{execution_id}.txt' if execution_id else output_dir / 'log_config.txt'
        config_source_file = root_dir / 'config.py'
        copyfile(config_source_file, config_output_file)
        print(f"Saved simulation configuration to {config_output_file}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")
    
    all_results_df = pd.DataFrame(all_results)
    all_results_df.columns = ['Simulation ID', 'Results summary', 'Timestep Results']
    # all_results_df.to_parquet(output_dir / f'simulation_results_{execution_id}.parquet', index=False)

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
    Update repair crew assignments based on accessibility, operational status, and repair time.

    Args:
        timestep (str or int): Current timestep in the simulation.
        available_repair_crews (int or dict): Number of available repair crews or a dictionary with island IDs as keys.
        repair_crews_assigned (np.ndarray): Boolean array indicating which assets already have repair crews assigned.
        accessible (np.ndarray): Boolean array indicating which assets are accessible.
        flooded_mask (np.ndarray): Boolean array indicating which assets are flooded.
        repair_time (np.ndarray): Array of remaining repair times for each asset.
        island_ids (np.ndarray, optional): Array of island IDs for each asset, if available.
        method (str, optional): Method for assigning repair crews:
            - 'random': Random assignment
            - 'lowest repair time': Assign to assets with lowest remaining repair time
            - 'highest repair time': Assign to assets with highest remaining repair time
        verbose (bool, optional): If True, print detailed assignment information.

    Returns:
        tuple: (updated available repair crews, updated repair crews assigned array)
            updated available repair crews can be an int or a dictionary with island IDs as keys.
            updated repair crews assigned is a boolean array indicating which assets have repair crews assigned.
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
    Performance analysis of the simulation.
    """
    import time
    import psutil
    import functools
    performance_data = {
        'timestep_times': [],
        'daily_operations': [],
        'memory_usage': [],
        'function_calls': {}
    }
    
    # Memory baseline
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Starting performance analysis (max {max_maps} maps)")
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    
    # Monkey patch key functions to track their performance
    original_match_islands = match_island_ids_assets
    original_hazard_extract = find_hazard_value_at_points_optimized
    original_crew_assignment = update_repair_crew_assignment_optimized
    original_island_overlaps = update_repair_crew_islands_with_overlap_cached
    original_grid_accessibility = grid_hex.accessibility_model
    
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
    
    # Replace functions temporarily with timed versions
    globals()['match_island_ids_assets'] = timed_function('match_islands', original_match_islands)
    globals()['find_hazard_value_at_points_optimized'] = timed_function('hazard_extract', original_hazard_extract)
    globals()['update_repair_crew_assignment_optimized'] = timed_function('crew_assignment', original_crew_assignment)
    globals()['update_repair_crew_islands_with_overlap_cached'] = timed_function('island_overlaps', original_island_overlaps) # current bottleneck
    globals()['grid_hex.accessibility_model'] = timed_function('grid_accessibility', original_grid_accessibility)
    
    try:
        # Run simulation with limited maps
        start_time = time.time()
        
        results_df, final_state = simulate_asset_damage_recovery_access_optimized(
            gdf_assets=gdf_assets,
            hazard_maps=hazard_maps[:max_maps],
            number_repair_crews=config['simulation_config']['number_repair_crews'],
            repair_crew_assignment_method=config['simulation_config']['repair_crew_assignment_method'],
            flood_threshold=config['simulation_config']['flood_threshold'],
            recovery_parameters=config['recovery_parameters'],
            root_dir=config['root_dir'],
            verbose=False
        )
        
        total_time = time.time() - start_time
        
        # Memory usage after simulation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate performance report
        print(f"\nPERFORMANCE ANALYSIS REPORT")
        print(f"Total simulation time: {total_time:.2f}s")
        print(f"Memory usage: {baseline_memory:.1f} MB → {final_memory:.1f} MB (Δ{final_memory-baseline_memory:+.1f} MB)")
        print(f"Timesteps processed: {len(results_df)}")
        print(f"Time per timestep: {total_time/len(results_df):.3f}s")
        
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
        
        # Identify bottlenecks
        print(f"\nBOTTLENECK ANALYSIS:")
        bottlenecks = []
        for func_name, times in performance_data['function_calls'].items():
            if times:
                total_func_time = sum(times)
                percentage = total_func_time / total_time * 100
                if percentage > 10:  # Functions taking >10% of total time
                    bottlenecks.append((func_name, percentage, total_func_time))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        if bottlenecks:
            for func_name, percentage, total_func_time in bottlenecks:
                print(f"  🔴 {func_name}: {percentage:.1f}% ({total_func_time:.2f}s)")
        else:
            print("  ✅ All functions <10% of total time")
        
        return performance_data, results_df, total_time
        
    finally:
        # Restore original functions
        globals()['match_island_ids_assets'] = original_match_islands
        globals()['find_hazard_value_at_points_optimized'] = original_hazard_extract
        globals()['update_repair_crew_assignment_optimized'] = original_crew_assignment
        globals()['update_repair_crew_islands_with_overlap_cached'] = original_island_overlaps
        globals()['grid_hex.accessibility_model'] = original_grid_accessibility
