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

def generate_island_timestep_report(timestep, day_counter, available_repair_crews, repair_crews_assigned, 
                                   damage_ratio, repair_time, island_ids, damage_threshold, verbose=False):
    """
    Generate island-by-island report for current timestep
    
    Returns dict with island statistics
    """
    island_report = {
        'timestep': timestep,
        'day': day_counter,
        'islands': {}
    }
    
    # Get unique islands
    unique_islands = np.unique(island_ids)
    
    for island_id in unique_islands:

        # Create mask for this island
        island_mask = (island_ids == island_id)
        
        # Count assets on this island
        total_assets_on_island = island_mask.sum()
        
        # Count damaged assets on this island
        damaged_on_island = island_mask & (damage_ratio > damage_threshold)
        damaged_count = damaged_on_island.sum()
        
        # Count crews assigned on this island
        crews_assigned_on_island = island_mask & repair_crews_assigned
        crews_assigned_count = crews_assigned_on_island.sum()
        
        # Available crews for this island
        available_crews = available_repair_crews.get(island_id, 0)
        
        # Calculate repair backlog for this island
        repair_backlog = repair_time[island_mask].sum()
        
        # Calculate average damage ratio for damaged assets on this island
        if damaged_count > 0:
            avg_damage_ratio = damage_ratio[damaged_on_island].mean()
            avg_repair_time = repair_time[damaged_on_island].mean()
        else:
            avg_damage_ratio = 0.0
            avg_repair_time = 0.0
        
        island_report['islands'][island_id] = {
            'total_assets': int(total_assets_on_island),
            'damaged_assets': int(damaged_count),
            'crews_assigned': int(crews_assigned_count),
            'available_crews': int(available_crews),
            'repair_backlog_hours': float(repair_backlog),
            'avg_damage_ratio': float(avg_damage_ratio),
            'avg_repair_time': float(avg_repair_time)
        }
    
    # Print summary if verbose and at daily intervals
    print(f"\n=== ISLAND REPORT - Day {day_counter} (Timestep {timestep}) ===")
    for island_id, stats in island_report['islands'].items():
        print(f"Island {island_id}: {stats['damaged_assets']}/{stats['total_assets']} damaged, "
                f"{stats['crews_assigned']} crews working, {stats['available_crews']} available, "
                f"{stats['repair_backlog_hours']:.1f}h backlog")
    
    return island_report


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
    config=None
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
    
    interim_dir = root_dir / 'data' / 'interim'
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
    
    # Load caches with hazard directory context
    print("\nLoading optimization caches...")
    accessibility_cache = load_accessibility_cache(interim_dir, hazard_dir)
    overlap_cache = load_overlap_cache(interim_dir, hazard_dir)
    hazard_extraction_cache = load_hazard_extraction_cache(interim_dir, hazard_dir)
    previous_day_counter = None  # Track previous day for overlap caching (considers overlap between previous and current day islands)
    
    # Initialize island cache if using island-based method
    island_cache = {}
    if 'island' in repair_crew_assignment_method:
        # hazard_thresholds = [0.2]  # Expected thresholds
        
        # Add hazard directory name to island cache filename
        island_cache = initialize_island_cache(interim_dir, hazard_dir_name)
        
    # Set up default recovery parameters if not provided
    if recovery_parameters is None:
        recovery_parameters = {
            'repair_time_coefficients': [702.72, 3.14, 1.9891],
            'damage_ratio_coefficients': (0.0468, 0.0077),  # Added damage ratio coefficients
            'time_step_hours': 1,
            'damage_threshold': 0.001,
            'repair_threshold': 2.0  # Default threshold for repairable damage ratio
        }
    
    repair_time_coefficients = recovery_parameters['repair_time_coefficients']
    damage_ratio_coefficients = recovery_parameters.get('damage_ratio_coefficients', (0.0468, 0.0077))
    # time_step_hours = recovery_parameters['time_step_hours'] #TODO remove time_step_hours from config
    damage_threshold = recovery_parameters['damage_threshold']
    repair_threshold = recovery_parameters['repair_threshold']

    # Initialize simulation arrays
    num_assets = len(gdf_assets)
    
    # State arrays - following original function initialization
    asset_type = gdf_assets['type'].values
    damage_ratio = np.zeros(num_assets, dtype=np.float64)
    repair_time = np.zeros(num_assets, dtype=np.float64)
    accessible = np.ones(num_assets, dtype=bool)  # Start as accessible
    unreachable = np.zeros(num_assets, dtype=bool)  # Start as reachable
    operational = np.ones(num_assets, dtype=bool)  # All start operational
    repair_crews_assigned = np.zeros(num_assets, dtype=bool)
    current_hazard_values = np.zeros(num_assets, dtype=np.float64)
    island_ids = np.zeros(num_assets, dtype=int)  # For island if needed
    previous_islands = None
    
    temp_gdf = gdf_assets.copy()  # Reference to original, not a copy
    
    # Results tracking
    results = []
    
    # Initialize timestep output if requested
    timestep_parquet_buffer = []
    timestep_output_file = None
    buffer_size = 100  # Write to disk every 100 timesteps
    
    if timestep_output:
        # Generate output filename based on execution_id
        if execution_id:
            timestep_output_file = output_dir / f'timestep_output_{execution_id}.parquet'
        else:
            timestep_output_file = output_dir / 'timestep_output.parquet'
        
        # Remove existing file if it exists (for clean start)
        if timestep_output_file.exists():
            timestep_output_file.unlink()
        
        if verbose:
            print(f"Timestep output will be written to: {timestep_output_file}")

    if _config['simulation_config']['accessibility_model'] is not None:
        # Initialize accessibility model analysis once if using any model
        print("Initializing grid-based accessibility analysis...")
        grid_hex.initialize_grid_analysis(root_dir)
    
    # Simulation loop
    available_repair_crews = number_repair_crews
    
    # Pre-compute information that only changes daily 
    island_method_active = 'island' in repair_crew_assignment_method
    if verbose and island_method_active:
        print(f"Island-based method '{repair_crew_assignment_method}' will be used for crew assignment")
    
    timesteps = np.arange(0, len(hazard_maps) * 24)  # 24 hours per day

    for timestep in timesteps:
        day_counter = timestep // 24
        day_counter_str = str(day_counter).zfill(2)
                
        # Every 24 hour-timesteps, process the hazard map for that day
        if timestep % 24 == 0:           
            if day_counter >= len(hazard_maps):
                break  # No more hazard maps available
                
            hazard_map = hazard_maps[day_counter]
            haz_col_str = f'EV{day_counter}_ma'
                        
            if verbose:
                print(f"\n=== Processing timestep {timestep} (day {day_counter}) ===")
            
            # Update hazard values
            temp_gdf = find_hazard_value_at_points_optimized(
                hazard_map, 
                temp_gdf, 
                day_counter, 
                extraction_method=_config['analysis_config']['hazard_extraction_method'],
                hazard_cache=hazard_extraction_cache,
                hazard_dir=hazard_dir
            )
            haz_val_str = f'hazard_value_{day_counter_str}'
            if haz_val_str in temp_gdf.columns:
                current_hazard_values = temp_gdf[haz_val_str].fillna(0.0).values
            else:
                current_hazard_values = temp_gdf[haz_col_str].fillna(0.0).values
            
            # Island-based crew management
            if 'island' in repair_crew_assignment_method:
                cache_key = f"{flood_threshold}_{haz_col_str}"
                
                if cache_key in island_cache:
                    # Use cached data
                    island_data = island_cache[cache_key]
                    island_ids = island_data['island_ids']
                    dissolved_roads = island_data['dissolved_roads']
                    
                    if verbose:
                        print(f"Using cached islands for {cache_key}")
                else:
                    # Cache miss - compute islands on the fly using proper function
                    print(f"Cache miss for {cache_key}, computing islands on the fly...")
                    
                    try:
                        temp_gdf_for_islands = gdf_assets.copy()
                        temp_gdf_for_islands, dissolved_roads = match_island_ids_assets(
                            temp_gdf_for_islands, 
                            hazard_threshold=flood_threshold, 
                            hazard_column=haz_col_str,
                            config=_config,
                
                        )
                        island_ids = temp_gdf_for_islands['island_id'].values
                        
                        # Cache the computed result for future use
                        island_data = {
                            'hazard_map': str(hazard_map),
                            'threshold': flood_threshold,
                            'island_ids': island_ids,
                            'dissolved_roads': dissolved_roads,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'computed_on_demand',
                            'method': 'match_island_ids_assets'
                        }
                        island_cache[cache_key] = island_data
                        
                        print(f"Successfully computed and cached islands for {cache_key}")
                        
                    except Exception as e:
                        print(f"Error computing islands for {cache_key}: {e}")
                        print("Falling back to simple island assignment")
                        # Fallback to simple assignment
                        island_ids = np.ones(len(gdf_assets), dtype=int)
                        dissolved_roads = None
                
                # Continue with crew distribution logic
                if dissolved_roads is not None:
                    available_repair_crews = update_repair_crew_islands_with_overlap_cached(
                        available_repair_crews, 
                        island_ids, 
                        dissolved_roads, 
                        previous_islands if 'previous_islands' in locals() else None,
                        current_day=day_counter,
                        previous_day=previous_day_counter,
                        hazard_threshold=flood_threshold,
                        overlap_cache=overlap_cache,
                        hazard_dir=hazard_dir
                    )

                    # Update previous day counter
                    previous_day_counter = day_counter
                    
                    # Store current islands for next iteration
                    previous_islands = dissolved_roads.copy()
                    
                    # Assign island_ids to temp_gdf for later use
                    if 'temp_gdf' not in locals():
                        temp_gdf = gdf_assets.copy()
                    temp_gdf['island_id'] = island_ids
                else:
                    print(f"No dissolved roads available for {cache_key}, using global crew assignment")
            
            # Mask of assets flooded above threshold  
            flooded_mask = current_hazard_values > flood_threshold 

            # Only apply fragility to assets that are NOT currently under repair
            not_under_repair_mask = ~repair_crews_assigned
            assets_to_evaluate = flooded_mask & not_under_repair_mask & operational
            
            if np.any(assets_to_evaluate):
                fragility_operational = np.ones_like(operational, dtype=bool)  # Start with all operational
                
                # Only evaluate assets not under repair
                hazard_subset = current_hazard_values[assets_to_evaluate]
                asset_type_subset = asset_type[assets_to_evaluate]
                
                fragility_result = default_fragility_function(hazard_subset, asset_type_subset)
                fragility_operational[assets_to_evaluate] = fragility_result.astype(bool)
                
                # Update operational status, but preserve assets under repair
                operational = np.minimum(operational, fragility_operational)
            
            # Update damage ratio for assets flooded above threshold this timestep
            if np.any(flooded_mask):
                dr_new = default_damage_ratio_function(current_hazard_values[flooded_mask], damage_ratio_coefficients)
                
                # Track which assets are getting new damage this timestep
                newly_damaged_mask = np.zeros_like(flooded_mask, dtype=bool)
                flooded_indices = np.where(flooded_mask)[0]
                
                # Check which flooded assets have increased damage
                new_damage_check = dr_new > damage_ratio[flooded_mask]
                newly_damaged_mask[flooded_indices] = new_damage_check
                
                # Update damage ratios (keep maximum damage)
                damage_ratio[flooded_mask] = np.maximum(damage_ratio[flooded_mask], dr_new)

                # Update repair times for assets with latest damage ratios
                repair_time[flooded_mask] = default_repair_time_function(
                    damage_ratio[flooded_mask], repair_time_coefficients
                )
                
                if verbose:
                    print(f"  New damage at timestep {timestep}: {newly_damaged_mask.sum()} assets")
                    print(f"  Damage ratios: {damage_ratio[newly_damaged_mask].min():.3f} to {damage_ratio[newly_damaged_mask].max():.3f}")
                    print(f"  Repair times: {repair_time[newly_damaged_mask].min():.1f} to {repair_time[newly_damaged_mask].max():.1f} hours")

            # For assets needing repair, solve for current damage ratio excluding assets under repair threshold
            recalc_repair_mask = (repair_time > repair_threshold)
            if np.any(recalc_repair_mask):
                repair_times_under_repair = repair_time[recalc_repair_mask]
                damage_ratios_from_repair = vectorized_damage_ratio_solver(
                    repair_times_under_repair, repair_time_coefficients
                )
                
                damage_ratio[recalc_repair_mask] = damage_ratios_from_repair
                
            # Daily accessibility update 
            accessibility_cache_key = create_accessibility_cache_key(day_counter, flood_threshold, hazard_dir, accessibility_model=_config['simulation_config']['accessibility_model'])

            if accessibility_cache_key in accessibility_cache:
                accessible = accessibility_cache[accessibility_cache_key]
                if verbose:
                    print(f"Using cached accessibility for day {day_counter} (hazard dir: {hazard_dir_name})")
            else:
                try:
                    # accessibility_result = grid_hex.accessibility_model(
                    #     gdf_assets.geometry, 
                    #     hazard_map, 
                    #     current_hazard_values,
                    #     verbose=verbose,
                    #     day_string=day_counter_str,
                    #     project_root=root_dir
                    # )
                    accessibility_result = accessible # Defaulting to accessible, to use only islands logic
                    accessible = np.array(accessibility_result, dtype=bool)
                    accessibility_cache[accessibility_cache_key] = accessible
                    
                    if verbose:
                        print(f"Accessibility updated for timestep {timestep} (day {day_counter})")
                        print(f"Accessible assets: {accessible.sum()} out of {num_assets}")
                except Exception as e:
                    print(f"Warning: Accessibility model failed: {e}")
                    print("Keeping current accessibility status")

        # Island-based crew assignment logic - simplified check since islands are already computed above
        if island_method_active:
            # Check if we still need to initialize island assignments (backup case)
            if isinstance(available_repair_crews, int):
                if verbose:
                    print(f"Backup: Initializing island assignments for {available_repair_crews} crews")
                
                temp_gdf, dissolved_roads = match_island_ids_assets(
                    temp_gdf,  
                    hazard_threshold=flood_threshold, 
                    hazard_column=haz_col_str,
                    config=_config
                )
                island_ids = temp_gdf['island_id'].values
                
                if dissolved_roads is not None and len(dissolved_roads) > 0:
                    # Use overlap-based crew redistribution to convert int to dict
                    available_repair_crews = update_repair_crew_islands_with_overlap_cached(
                        available_repair_crews,
                        island_ids, 
                        dissolved_roads, 
                        previous_islands if 'previous_islands' in locals() else None,
                        current_day=day_counter,
                        previous_day=previous_day_counter,
                        hazard_threshold=flood_threshold,
                        overlap_cache=overlap_cache,
                        hazard_dir=hazard_dir,
                        verbose=verbose
                    )
                    
                    # Store current islands for next iteration
                    previous_islands = dissolved_roads.copy()
                    
                    if verbose:
                        print(f"Distributed crews across {len(dissolved_roads)} islands: {available_repair_crews}")
                else:
                    if verbose:
                        print("No dissolved roads found, using global crew assignment")

        # Repair crew assignment
        available_repair_crews, repair_crews_assigned = update_repair_crew_assignment_optimized(
            timestep, 
            available_repair_crews, 
            repair_crews_assigned, 
            accessible, 
            current_hazard_values > flood_threshold,  # flooded_mask
            repair_time, 
            island_ids, 
            method=repair_crew_assignment_method, 
            verbose=verbose
        )
        
        # For each timestep, decrement repair_time if accessible, not flooded, and with repair crews assigned
        can_repair_mask = accessible & ~flooded_mask & repair_crews_assigned
        repair_time[can_repair_mask] = np.maximum(repair_time[can_repair_mask] - 1.0, 0.0)
        
        # Check for completed repairs
        completed_repairs = (repair_time == 0.0) & repair_crews_assigned
        
        if np.any(completed_repairs):
            # Only force operational=True for assets that are non-operational but completed repair
            non_operational_completed = completed_repairs & (~operational)
            
            if np.any(non_operational_completed):
                operational[non_operational_completed] = True
            
            # For ALL completed repairs, reset damage and release crews
            damage_ratio[completed_repairs] = 0.0
            repair_time[completed_repairs] = 0.0
            
            # Release repair crews
            num_completed_repairs = completed_repairs.sum()
            if available_repair_crews is not None:
                if isinstance(available_repair_crews, dict):
                    # Island-based crew management
                    for asset_idx in np.where(completed_repairs)[0]:
                        asset_island_id = island_ids[asset_idx]
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
            
            repair_crews_assigned[completed_repairs] = False

            if verbose:
                completed_repairs_indices = np.where(completed_repairs)[0]
                print(f"Assets {completed_repairs_indices.tolist()} became operational at timestep {timestep}")

        if island_method_active:
            # Islands with zero available crews and no crews assigned
            all_island_ids = np.unique(island_ids)
            idle_crew_islands = [island_id for island_id, crew_count in available_repair_crews.items() if crew_count > 0]
            assigned_crew_islands = set(island_ids[repair_crews_assigned])
            islands_with_crews = set(idle_crew_islands) | assigned_crew_islands
            islands_without_crews = set(all_island_ids) - islands_with_crews
            in_islands_without_crews = np.isin(island_ids, list(islands_without_crews))
            unreachable = (
                (damage_ratio > damage_threshold) &
                (~flooded_mask) &
                in_islands_without_crews
            )
            # unreachable_count = unreachable_mask.sum()

        # Write timestep data to parquet buffer if requested
        if timestep_output:
            timestep_data = {
                'timestep': timestep,
                'day': day_counter,
                'asset_id': range(num_assets),
                'damage_ratio': damage_ratio.copy(),
                'repair_time': repair_time.copy(),
                'operational': operational.astype(int),
                'accessible': accessible.astype(int),
                'unreachable': unreachable.astype(int),
                'flooded': flooded_mask.astype(int),
                'crew_assigned': repair_crews_assigned.astype(int),
                'hazard_value': current_hazard_values.copy()
            }
            
            timestep_df = pd.DataFrame(timestep_data)
            timestep_parquet_buffer.append(timestep_df)
            
            # Write buffer to file when it reaches size limit or at end of day
            if len(timestep_parquet_buffer) >= buffer_size or timestep % 24 == 23:
                write_timestep_buffer_to_parquet(timestep_parquet_buffer, timestep_output_file)
                timestep_parquet_buffer.clear()
                
                if verbose and timestep % 24 == 23:
                    print(f"Wrote timestep data to {timestep_output_file} (day {day_counter} complete)")
        
        # Record results at every timestep 
        # Only calculate averages for assets that actually have damage/repair time to avoid computational spikes
        damaged_assets_mask = damage_ratio > damage_threshold
        repair_needed_mask = repair_time > repair_threshold
        
        # Calculate output metrics
        if np.any(damaged_assets_mask):
            avg_damage_ratio = damage_ratio[damaged_assets_mask].mean()
        else:
            avg_damage_ratio = 0.0
            
        if np.any(repair_needed_mask):
            avg_repair_time = repair_time[repair_needed_mask].mean()
        else:
            avg_repair_time = 0.0
        
        total_repair_backlog = repair_time.sum()  # Sum of all repair hours remaining
        total_damage_ratio = damage_ratio.sum()   # Sum of all damage ratios
        
        results.append({
            'day': day_counter,
            'timestep': timestep,
            'operational_count': operational.sum(),
            'accessible_count': accessible.sum(),
            'unreachable_count': unreachable.sum(),
            'flooded_count': (current_hazard_values > flood_threshold).sum(),
            'damaged_count': damaged_assets_mask.sum(),
            'crews_assigned_count': repair_crews_assigned.sum(),
            'avg_damage_ratio': avg_damage_ratio,
            'avg_repair_time': avg_repair_time,
            'total_repair_backlog': total_repair_backlog,  
            'total_damage_ratio': total_damage_ratio       
        })

        if timestep % 24 == 23:  # End of day
            if verbose:
                print(f"Day {day_counter} summary: {operational.sum()}/{num_assets} operational, "
                      f"{accessible.sum()} accessible, {unreachable.sum()} unreachable damaged assets, {(current_hazard_values > flood_threshold).sum()} flooded")

    # Write any remaining data in buffer
    if timestep_output and timestep_parquet_buffer:
        write_timestep_buffer_to_parquet(timestep_parquet_buffer, timestep_output_file)
        print(f"Final timestep data written to {timestep_output_file}")
    
    # Save caches for next run with hazard directory context
    print("\nSaving optimization caches...")
    save_accessibility_cache(accessibility_cache, interim_dir, hazard_dir)
    
    if hazard_extraction_cache:
        save_hazard_extraction_cache(hazard_extraction_cache, interim_dir, hazard_dir)
    
    if overlap_cache:
        save_overlap_cache(overlap_cache, interim_dir, hazard_dir)
    
    # Save island cache if it was updated
    if island_cache:
        try:
            cache_dir = interim_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create island cache filename with hazard directory context
            hazard_dir_name = Path(hazard_dir).name if hazard_dir else "unknown"
            
            island_cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
            
            with open(island_cache_file, 'wb') as f:
                pickle.dump(island_cache, f)
            
            print(f"Saved updated island cache with {len(island_cache)} entries to {island_cache_file}")
            
        except Exception as e:
            print(f"Warning: Could not save island cache: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Copy of config.py file as text to log_config_executionid.txt
    try:
        config_output_file = output_dir / f'log_config_{execution_id}.txt' if execution_id else output_dir / 'log_config.txt'
        config_source_file = root_dir / 'config.py'
        copyfile(config_source_file, config_output_file)
        print(f"Saved simulation configuration to {config_output_file}")
    except Exception as e:
        print(f"Warning: Could not save configuration file: {e}")

    return results_df, {
        'operational': operational,
        'hazard_value': current_hazard_values,
        'damage_ratio': damage_ratio,
        'repair_time': repair_time,
        'accessible': accessible,
        'repair_crews_assigned': repair_crews_assigned
    }




def write_timestep_buffer_to_parquet(buffer, output_file):
    """
    Write buffered timestep data to parquet file, appending if file exists.
    
    Args:
        buffer (list): List of DataFrames to write
        output_file (Path): Path to output parquet file
    """
    if not buffer:
        return
    
    # Combine buffer data
    combined_df = pd.concat(buffer, ignore_index=True)
    
    # Append to existing file or create new one
    if output_file.exists():
        try:
            existing_df = pd.read_parquet(output_file)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not read existing parquet file {output_file}: {e}")
            print("Creating new file instead")
    
    # Write to parquet with compression
    try:
        combined_df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    except ImportError:
        # Fallback to default engine if pyarrow not available
        combined_df.to_parquet(output_file, index=False, compression='gzip')
    except Exception as e:
        print(f"Error writing to parquet file {output_file}: {e}")
        # Fallback to CSV if parquet fails
        csv_file = output_file.with_suffix('.csv')
        combined_df.to_csv(csv_file, index=False)
        print(f"Saved as CSV instead: {csv_file}")


def update_repair_crew_assignment_optimized(timestep, available_repair_crews, repair_crews_assigned, 
                                           accessible, flooded_mask, repair_time, island_ids=None, method=None, verbose=False):    
    """
    Update repair crew assignments based on accessibility, operational status, and repair time.

    Args:
        timestep (str or int): Current timestep/day in the simulation (matches day_counter_str from original).
        available_repair_crews (int or dict): Number of available repair crews or a dictionary with island IDs as keys.
        repair_crews_assigned (np.ndarray): Boolean array indicating which assets already have repair crews assigned.
        accessible (np.ndarray): Boolean array indicating which assets are accessible.
        flooded_mask (np.ndarray): Boolean array indicating which assets are flooded (matches original parameter name).
        repair_time (np.ndarray): Array of remaining repair times for each asset.
        island_ids (np.ndarray, optional): Array of island IDs for each asset, if available.
        method (str, optional): Method for assigning repair crews (matches repair_crew_assignment_method from original):
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


def analyze_simulation_performance(gdf_assets, hazard_maps, config, max_days=3):
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
    
    print(f"Starting performance analysis (max {max_days} days)")
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
        # Run simulation with limited days
        start_time = time.time()
        
        results_df, final_state = simulate_asset_damage_recovery_access_optimized(
            gdf_assets=gdf_assets,
            hazard_maps=hazard_maps[:max_days],
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
