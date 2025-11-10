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
    
    # L1: Area-based depth reductions
    if l1_area_geojson is not None and l1_active_timesteps is not None:
        l1_gdf = (gpd.read_file(l1_area_geojson) 
                  if isinstance(l1_area_geojson, (str, Path)) 
                  else l1_area_geojson)
        
        if l1_gdf.crs != gdf_assets.crs:
            l1_gdf = l1_gdf.to_crs(gdf_assets.crs)
        
        if "depth_red" not in l1_gdf.columns:
            # Default depth reduction if not specified
            l1_gdf['depth_red'] = 0.3
            print("Warning: L1 GeoJSON missing 'depth_red' column, using default 0.3m")
        
        # Build STRtree from ASSETS (larger dataset)
        asset_tree = shapely.STRtree(gdf_assets.geometry.values)
        
        # Find assets protected by L1
        l1_per_asset = np.zeros(n_assets, dtype=np.float32)
        
        for l1_idx, l1_row in l1_gdf.iterrows():
            l1_geom = l1_row.geometry
            l1_depth_red = l1_row['depth_red']
            
            intersecting_asset_indices = asset_tree.query(l1_geom, predicate='intersects')
            
            for asset_idx in intersecting_asset_indices:
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
                
                intersecting_l2_indices = l2_tree.query(asset_geom, predicate='intersects')
                
                if len(intersecting_l2_indices) > 0:
                    l2_per_asset[asset_idx] = l2_gdf.iloc[intersecting_l2_indices]['depth_red'].max()
        
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


