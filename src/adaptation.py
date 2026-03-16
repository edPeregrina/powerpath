"""
Adaptation measures for L1 (area-based depth reduction) and L2 (asset-specific depth reduction).
Uses STRtree for optimal spatial queries (query smaller dataset against larger).
All spatial operations are performed once per simulation run and cached.
"""
import numpy as np
import geopandas as gpd
from pathlib import Path
import shapely
import pickle
import networkx as nx
import hashlib
from typing import Optional, Tuple, List, Dict



# Module-level cache for adaptation arrays
_ADAPTATION_CACHE: Dict[tuple, np.ndarray] = {}

def _build_adaptation_arrays_cached(
    gdf_assets: gpd.GeoDataFrame,
    n_timesteps: int,
    l1_area_geojson: Optional[str] = None,
    l1_active_timesteps: Optional[List[int]] = None,
    l2_asset_geojson: Optional[str] = None,
    l2_active_timesteps: Optional[List[int]] = None,
    config: Optional[Dict] = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Build adaptation depth reduction arrays with memory and disk caching.
    
    Creates a (n_timesteps, n_assets, 2) array where:
    - [..., 0] = L1 adaptation (area-based, e.g., pumping/barriers)
    - [..., 1] = L2 adaptation (asset-based, e.g., individual protection)
    
    Parameters
    ----------
    gdf_assets : gpd.GeoDataFrame
        Asset geodataframe
    n_timesteps : int
        Number of simulation timesteps
    l1_area_geojson : str, optional
        Path to L1 adaptation polygon (area-based protection)
    l1_active_timesteps : list of int, optional
        Timesteps when L1 adaptation is active
    l2_asset_geojson : str, optional
        Path to L2 adaptation polygons (asset-specific protection)
    l2_active_timesteps : list of int, optional
        Timesteps when L2 adaptation is active
    config : dict, optional
        Configuration dict with 'interim_dir' for disk caching
    verbose : bool
        Print diagnostic information
    
    Returns
    -------
    np.ndarray
        Shape (n_timesteps, n_assets, 2) with depth reduction values
    """
    # Create cache key from adaptation configuration
    cache_key = (
        str(l1_area_geojson) if l1_area_geojson else 'no_l1',
        tuple(l1_active_timesteps) if l1_active_timesteps else (),
        str(l2_asset_geojson) if l2_asset_geojson else 'no_l2',
        tuple(l2_active_timesteps) if l2_active_timesteps else (),
        n_timesteps,
        len(gdf_assets)
    )
    
    # Check memory cache first
    if cache_key in _ADAPTATION_CACHE:
        if verbose:
            print("Using cached adaptation arrays (memory)")
        return _ADAPTATION_CACHE[cache_key]
    
    # Check disk cache if config provided
    if config is not None:
        cache_path = _get_adaptation_cache_path(config, cache_key)
        if cache_path.exists():
            if verbose:
                print(f"Loading adaptation arrays from disk cache: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                adaptation_array = pickle.load(f)
            _ADAPTATION_CACHE[cache_key] = adaptation_array
            return adaptation_array
    
    # Build from scratch
    if verbose:
        print("Building adaptation depth reduction arrays...")
    
    adaptation_array = np.zeros((n_timesteps, len(gdf_assets), 2), dtype=np.float32)
    l1_active_count = 0
    l2_active_count = 0
    
    # L1 area-based adaptation (e.g., pumping stations, flood barriers)
    if l1_area_geojson is not None and l1_active_timesteps:
        l1_gdf = gpd.read_file(l1_area_geojson)
        if l1_gdf.crs != gdf_assets.crs:
            l1_gdf = l1_gdf.to_crs(gdf_assets.crs)
        
        l1_polygon = l1_gdf.geometry.unary_union
        assets_in_l1 = gdf_assets.geometry.intersects(l1_polygon)
        asset_indices_in_l1 = np.where(assets_in_l1)[0]
        
        for t in l1_active_timesteps:
            if t < n_timesteps:
                adaptation_array[t, asset_indices_in_l1, 0] = 0.5
                l1_active_count += len(asset_indices_in_l1)
    
    # L2 asset-based adaptation (e.g., individual asset protection)
    if l2_asset_geojson is not None and l2_active_timesteps:
        l2_gdf = gpd.read_file(l2_asset_geojson)
        if l2_gdf.crs != gdf_assets.crs:
            l2_gdf = l2_gdf.to_crs(gdf_assets.crs)
        
        l2_polygon = l2_gdf.geometry.unary_union
        assets_in_l2 = gdf_assets.geometry.intersects(l2_polygon)
        asset_indices_in_l2 = np.where(assets_in_l2)[0]
        
        for t in l2_active_timesteps:
            if t < n_timesteps:
                adaptation_array[t, asset_indices_in_l2, 1] = 0.5
                l2_active_count += len(asset_indices_in_l2)
    
    if verbose:
        print(f"Built adaptation reduction array: shape {adaptation_array.shape}")
        print(f"  L1 active entries: {l1_active_count}")
        print(f"  L2 active entries: {l2_active_count}")
    
    # Store in memory cache
    _ADAPTATION_CACHE[cache_key] = adaptation_array
    
    # Store in disk cache
    if config is not None:
        cache_path = _get_adaptation_cache_path(config, cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(adaptation_array, f)
        if verbose:
            print(f"Saved adaptation arrays to disk: {cache_path.name}")
    
    return adaptation_array


def _get_adaptation_cache_path(config: Dict, cache_key: tuple) -> Path:
    """Generate a filepath for adaptation array cache."""
    cache_dir = Path(config['interim_dir']) / 'adaptation_cache'
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Create hash from cache_key
    key_str = str(cache_key)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    return cache_dir / f"adaptation_{key_hash}.pkl"


def clear_adaptation_cache() -> None:
    """Clear the in-memory adaptation cache."""
    global _ADAPTATION_CACHE
    _ADAPTATION_CACHE.clear()
    print("Cleared adaptation cache")


def get_adaptation_cache_stats() -> Dict[str, int]:
    """Get statistics about the adaptation cache."""
    return {
        'memory_entries': len(_ADAPTATION_CACHE),
        'total_size_bytes': sum(arr.nbytes for arr in _ADAPTATION_CACHE.values())
            }

def build_l1_l2_reduction_array(
    gdf_assets, 
    l1_area_geojson=None, 
    l1_active_timesteps=None,  
    l2_asset_geojson=None,
    l2_active_timesteps=None,  
    hazard_maps=None,
    major_timestep=24,
    config=None,
    verbose=False
):
    """
    Build a 3D (n_timesteps x n_assets x 2) array of depth reductions. Timesteps are hours by default.
    Channel 0: L1 (area-based), Channel 1: L2 (asset-specific)
    
    L1: Area-based pumping/protection (polygon features with depth_red attribute)
    L2: Asset-specific measures (point/polygon features with depth_red attribute OR legacy dict format)
    
    Args:
        gdf_assets: GeoDataFrame of assets (must have geometry column and index)
        l1_area_geojson: Path to GeoJSON or GeoDataFrame with 'depth_red' column
        l1_active_timesteps: List of timesteps (hours) where L1 applies
        l2_asset_geojson: Path to GeoJSON or GeoDataFrame with 'depth_red' column
        l2_active_timesteps: List of timesteps (hours) where L2 applies
        l2_asset_depth_red: Legacy dict format {timestep: [(asset_idx, depth_red), ...]}
        hazard_maps: List of hazard maps (to determine n_timesteps)
        major_timestep: Hours per hazard map
        config: Configuration dict
        verbose: Print progress
    
    Returns:
        np.ndarray: Shape (n_timesteps, n_assets, 2) with depth reductions
                   [:, :, 0] = L1 reductions
                   [:, :, 1] = L2 reductions
    """
    n_timesteps = len(hazard_maps) * major_timestep
    n_assets = len(gdf_assets)
    reductions = np.zeros((n_timesteps, n_assets, 2), dtype=np.float32)
    
    # Convert None to all timesteps
    if l1_active_timesteps is None:
        l1_active_timesteps = list(range(n_timesteps))
    if l2_active_timesteps is None:
        l2_active_timesteps = list(range(n_timesteps))

    # L1: Area-based depth reductions
    if l1_area_geojson is not None and l1_active_timesteps is not None:
        l1_gdf = (gpd.read_file(l1_area_geojson) 
                  if isinstance(l1_area_geojson, (str, Path)) 
                  else l1_area_geojson)
        
        if l1_gdf.crs != gdf_assets.crs:
            l1_gdf = l1_gdf.to_crs(gdf_assets.crs)
        
        if "depth_red" not in l1_gdf.columns:
            l1_gdf['depth_red'] = 0.3
            print("Warning: L1 GeoJSON missing 'depth_red' column, using default 0.3m")
        
        # Build STRtree from ASSETS (larger dataset)
        asset_tree = shapely.STRtree(gdf_assets.geometry.values)
        
        l1_per_asset = np.zeros(n_assets, dtype=np.float32)
        
        for l1_idx, l1_row in l1_gdf.iterrows():
            l1_geom = l1_row.geometry
            l1_depth_red = l1_row['depth_red']
            
            # BROAD-PHASE: Get candidates from spatial index
            candidate_indices = asset_tree.query(l1_geom, predicate='intersects')
            
            # NARROW-PHASE: Verify actual intersection
            for asset_idx in candidate_indices:
                asset_geom = gdf_assets.geometry.iloc[asset_idx]
                if asset_geom.intersects(l1_geom):  # Actual geometry check
                    l1_per_asset[asset_idx] = max(l1_per_asset[asset_idx], l1_depth_red)
        
        # Apply to specified timesteps
        for timestep in l1_active_timesteps:
            if 0 <= timestep < n_timesteps:
                reductions[timestep, :, 0] = l1_per_asset
        
        if verbose:
            print(f"Applied L1 to {np.count_nonzero(l1_per_asset)} assets at {len(l1_active_timesteps)} timesteps")

    # L2: Asset-specific depth reductions (GeoJSON approach)
    if l2_asset_geojson is not None and l2_active_timesteps is not None:
        l2_gdf = (gpd.read_file(l2_asset_geojson) 
                  if isinstance(l2_asset_geojson, (str, Path)) 
                  else l2_asset_geojson)
        
        if l2_gdf.crs != gdf_assets.crs:
            l2_gdf = l2_gdf.to_crs(gdf_assets.crs)
        
        if "depth_red" not in l2_gdf.columns:
            l2_gdf['depth_red'] = 0.15
            print("Warning: L2 GeoJSON missing 'depth_red' column, using default 0.15m")
        
        l2_per_asset = np.zeros(n_assets, dtype=np.float32)
        
        # Adaptive query direction
        if len(l2_gdf) < n_assets:
            asset_tree = shapely.STRtree(gdf_assets.geometry.values)
            
            for l2_idx, l2_row in l2_gdf.iterrows():
                l2_geom = l2_row.geometry
                l2_depth_red = l2_row['depth_red']
                
                intersecting_asset_indices = asset_tree.query(l2_geom, predicate='intersects')
                
                for asset_idx in intersecting_asset_indices:
                    l2_per_asset[asset_idx] = max(l2_per_asset[asset_idx], l2_depth_red)
        else:
            l2_tree = shapely.STRtree(l2_gdf.geometry.values)
            
            for asset_idx, (idx, asset_row) in enumerate(gdf_assets.iterrows()):
                asset_geom = asset_row.geometry
                
                # BROAD-PHASE: Get candidates
                candidate_indices = l2_tree.query(asset_geom, predicate='intersects')
                
                # NARROW-PHASE: Verify actual intersection
                if len(candidate_indices) > 0:
                    max_reduction = 0
                    for l2_idx in candidate_indices:
                        l2_geom = l2_gdf.geometry.iloc[l2_idx]
                        if asset_geom.intersects(l2_geom):  # Actual geometry check
                            max_reduction = max(max_reduction, l2_gdf.iloc[l2_idx]['depth_red'])
                    
                    if max_reduction > 0:
                        l2_per_asset[asset_idx] = max_reduction
                                
        # Apply to specified timesteps
        for timestep in l2_active_timesteps:
            if 0 <= timestep < n_timesteps:
                reductions[timestep, :, 1] = l2_per_asset
        
        if verbose:
            print(f"Applied L2 to {np.count_nonzero(l2_per_asset)} assets at {len(l2_active_timesteps)} timesteps")
    

    return reductions

def convert_l2_dict_to_geodataframe(l2_dict, gdf_assets):
    """
    Convert L2 adaptation dictionary to GeoDataFrame format.
    
    Args:
        l2_dict: {timestep: [(asset_id, depth_red), ...]}
        gdf_assets: GeoDataFrame with asset geometries
        
    Returns:
        GeoDataFrame with 'depth_red' column
    """
    # Group by asset_id and find max depth_red across all timesteps
    asset_depth_red = {}
    for timestep, measures in l2_dict.items():
        for asset_id, depth_red in measures:
            asset_depth_red[asset_id] = max(
                asset_depth_red.get(asset_id, 0), 
                depth_red
            )
    
    # Create GeoDataFrame
    l2_gdf = gdf_assets.loc[list(asset_depth_red.keys())].copy()
    l2_gdf['depth_red'] = [asset_depth_red[aid] for aid in l2_gdf.index]
    
    return l2_gdf


def simulate_asset_damage_recovery_access_breakdown_ema(*args, **kwargs):
    """
    EMA-compatible wrapper for simulation with population and monetary impact calculation.

    This wrapper manages memory by only creating arrays for requested outcomes. 
    Use of 3D outcomes quickly causes out of memory errors. 
    This functionality is kept in for troubleshooting/inspecting.

    Parameters:
    -----------
    keep_3d_vars : list of str, optional
        Variables to keep as 3D arrays (n_timesteps, n_assets).
        Example: ['flooded', 'operational']
        Default: []
        
    keep_2d_vars : list of str, optional
        Variables to keep as 2D arrays (n_experiments, n_timesteps) - aggregated across assets.
        Example: ['damage_ratio', 'repair_time', 'hazard_value']
        Default: []
        
    Available asset metrics:
        - 3D or 2D: 'damage_ratio', 'repair_time', 'operational', 'accessible',
                    'unreachable', 'flooded', 'crew_assigned', 'hazard_value'
        - 3D only: 'island_id' (arbitrary IDs, meaningless when aggregated)
        
    Note: Population and monetary impacts are always 1D (n_timesteps) - already aggregated.
    """
    import numpy as np
    import pandas as pd
    from src.simulation import simulate_asset_damage_recovery_access_breakdown  # ✅ ADD THIS!
    from src.impacts import calculate_population_impacts, calculate_cumulative_monetized_impacts_ema

    # Helper functions
    
    def _extract_configuration(kwargs):
        """Extract and validate configuration parameters."""
        monetary_categories = ['residential', 'commercial', 'industrial',
                              'transport', 'public_sector']
        
        asset_population_map = kwargs.get('asset_population_map', {})
        asset_to_lu = kwargs.get('asset_to_lu', {})
        
        keep_3d_vars = kwargs.pop('keep_3d_vars', [])
        keep_2d_vars = kwargs.pop('keep_2d_vars', [])
        
        if keep_3d_vars is None:
            keep_3d_vars = []
        if keep_2d_vars is None:
            keep_2d_vars = []
        
        return {
            'monetary_categories': monetary_categories,
            'asset_population_map': asset_population_map,
            'asset_to_lu': asset_to_lu,
            'keep_3d_vars': keep_3d_vars,
            'keep_2d_vars': keep_2d_vars
        }
    
    def _add_impact_metrics(timestep_results, detailed_results, asset_population_map, 
                            asset_to_lu, monetary_categories):
        """Calculate and add population and monetary impacts to timestep results."""
        # Add population impacts
        if asset_population_map:
            timestep_results = calculate_population_impacts(detailed_results, asset_population_map)
        
        # Add monetary impacts
        if asset_to_lu:
            monetized_results = calculate_cumulative_monetized_impacts_ema(  
                detailed_results, 
                asset_to_lu,
            )
            
            # Ensure all expected monetary columns exist
            for category in monetary_categories:
                expected_col = f'monetary_impact_{category}'
                if expected_col not in monetized_results.columns:
                    monetized_results[expected_col] = 0.0
            
            if 'monetary_impact_total' not in monetized_results.columns:
                monetized_results['monetary_impact_total'] = 0.0
            
            # Copy monetary columns to timestep_results
            for col in monetized_results.columns:
                if col.startswith('monetary_'):
                    timestep_results[col] = monetized_results[col]
        else:
            # No land use mapping - create zero columns
            for category in monetary_categories:
                timestep_results[f'monetary_impact_{category}'] = 0.0
            timestep_results['monetary_impact_total'] = 0.0
        
        return timestep_results
    
    def _initialize_result_arrays(n_timesteps, n_assets, keep_3d_vars, keep_2d_vars,
                                   asset_population_map, monetary_categories):
        """Initialize result arrays based on requested metrics."""
        aggregatable_metrics = [
            'damage_ratio', 'repair_time', 'operational', 'accessible',
            'unreachable', 'flooded', 'crew_assigned', 'hazard_value'
        ]
        three_d_only_metrics = ['island_id']
        
        result = {}
        
        # Always include timesteps
        result['timesteps'] = np.zeros(n_timesteps)
        
        # Population impacts - always 1D (already aggregated)
        if asset_population_map:
            result['affected_population'] = np.zeros(n_timesteps)
            result['served_population'] = np.zeros(n_timesteps)
            result['affected_population_ratio'] = np.zeros(n_timesteps)
        
        # Monetary impacts - always 1D (already aggregated)
        result['monetary_impact_total'] = np.zeros(n_timesteps)
        for category in monetary_categories:
            result[f'monetary_impact_{category}'] = np.zeros(n_timesteps)
        
        # Asset metrics 
        # Tier 1: 3D arrays (per-asset, per-timestep)
        for metric in keep_3d_vars:
            if metric in aggregatable_metrics or metric in three_d_only_metrics:
                result[metric] = np.zeros((n_timesteps, n_assets))
            else:
                print(f"Warning: '{metric}' in keep_3d_vars is not a valid asset metric. Skipping.")
        
        # Tier 2: 2D arrays (aggregated across assets)
        for metric in keep_2d_vars:
            if metric in aggregatable_metrics:
                result[metric] = np.zeros(n_timesteps)
            elif metric in three_d_only_metrics:
                raise ValueError(
                    f"Cannot include '{metric}' in keep_2d_vars - this metric is only meaningful "
                    f"as a 3D array (per-asset). Island IDs are arbitrary and lose meaning when aggregated. "
                    f"Either add it to keep_3d_vars or omit it."
                )
            else:
                print(f"Warning: '{metric}' in keep_2d_vars is not a valid asset metric. Skipping.")
        
        return result
    
    def _fill_timestep_arrays(result, timestep_results, n_timesteps, n_assets,
                              keep_3d_vars, keep_2d_vars, asset_population_map, 
                              monetary_categories):
        """Fill result arrays from timestep data for requested metrics."""
        summary_to_metric = {
            'avg_damage_ratio': 'damage_ratio',
            'avg_repair_time': 'repair_time',
        }
        count_to_binary = {
            'operational_count': 'operational',
            'accessible_count': 'accessible',
            'unreachable_count': 'unreachable',
            'flooded_count': 'flooded',
            'crews_assigned_count': 'crew_assigned'
        }
        
        for t_idx, (_, ts) in enumerate(timestep_results.iterrows()):
            # Timesteps
            result['timesteps'][t_idx] = ts.get('timestep', t_idx)
            
            # Population impacts (1D)
            if asset_population_map:
                result['affected_population'][t_idx] = ts.get('affected_population', 0)
                result['served_population'][t_idx] = ts.get('served_population', 0)
                result['affected_population_ratio'][t_idx] = ts.get('affected_population_ratio', 0)
            
            # Monetary impacts (1D)
            result['monetary_impact_total'][t_idx] = ts.get('monetary_impact_total', 0)
            for category in monetary_categories:
                col_name = f'monetary_impact_{category}'
                result[col_name][t_idx] = ts.get(col_name, 0)
            
            # Asset metrics - ONLY process requested metrics
            all_requested_metrics = keep_3d_vars + keep_2d_vars
            
            for metric in all_requested_metrics:
                value = _get_metric_value(ts, metric, summary_to_metric, count_to_binary)
                
                if value is None:
                    continue  # Skip if data not available
                
                if metric in keep_3d_vars:
                    _store_3d_metric(result, metric, t_idx, value, n_assets, count_to_binary)
                elif metric in keep_2d_vars:
                    _store_2d_metric(result, metric, t_idx, value, n_assets, count_to_binary)
        
        return result
    
    def _get_metric_value(ts, metric, summary_to_metric, count_to_binary):
        """Extract metric value from timestep data, checking multiple sources."""
        if metric in ts:
            return ts[metric]
        
        # Try summary columns
        for summary_col, target_metric in summary_to_metric.items():
            if target_metric == metric and summary_col in ts:
                return ts[summary_col]
        
        # Try count columns
        for count_col, target_metric in count_to_binary.items():
            if target_metric == metric and count_col in ts:
                return ts[count_col]
        
        return None
    
    def _store_3d_metric(result, metric, t_idx, value, n_assets, count_to_binary):
        """Store per-asset values in 3D array."""
        if isinstance(value, (list, np.ndarray)):
            result[metric][t_idx, :] = value[:n_assets]
        elif metric in count_to_binary.values():
            # Convert count to ratio for binary metrics
            ratio = value / n_assets if n_assets > 0 else 0
            result[metric][t_idx, :] = ratio
        else:
            result[metric][t_idx, :] = value
    
    def _store_2d_metric(result, metric, t_idx, value, n_assets, count_to_binary):
        """Aggregate and store metric as scalar in 2D array."""
        if isinstance(value, (list, np.ndarray)):
            # Aggregation strategy depends on metric type
            if metric in ['damage_ratio', 'repair_time', 'hazard_value']:
                # Mean for continuous values
                result[metric][t_idx] = np.mean(value)
            else:
                # Sum for binary/count values
                result[metric][t_idx] = np.sum(value)
        elif metric in count_to_binary.values():
            # Already a count, just store it
            result[metric][t_idx] = value
        else:
            result[metric][t_idx] = value
    
    # ============================================================================
    # Main function
    
    # Extract configuration
    config = _extract_configuration(kwargs)
    
    # Run the simulation
    results_df, final_state, cache_updated = simulate_asset_damage_recovery_access_breakdown(*args, **kwargs)
    timestep_data = results_df[0][1]
    detailed_results = results_df[0][2]
    timestep_results = pd.DataFrame(timestep_data)
    
    # Add impact metrics
    timestep_results = _add_impact_metrics(
        timestep_results, 
        detailed_results,
        config['asset_population_map'],
        config['asset_to_lu'],
        config['monetary_categories']
    )
    
    # Get dimensions
    n_assets = len(kwargs['gdf_assets'])
    n_timesteps = len(timestep_results)
    
    # Initialize result arrays (MEMORY OPTIMIZED)
    result = _initialize_result_arrays(
        n_timesteps,
        n_assets,
        config['keep_3d_vars'],
        config['keep_2d_vars'],
        config['asset_population_map'],
        config['monetary_categories']
    )
    
    # Fill result arrays (ONLY for requested metrics)
    result = _fill_timestep_arrays(
        result,
        timestep_results,
        n_timesteps,
        n_assets,
        config['keep_3d_vars'],
        config['keep_2d_vars'],
        config['asset_population_map'],
        config['monetary_categories']
    )
    
    return result