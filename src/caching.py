"""
Functions for caching output of expensive deterministic calculations.
The cached operations include:
- Grid accessibility calculations
- Hazard extraction
- Island creation
- Island overlap calculations
For each operation, a unique cache key is generated based on input parameters.
The cache is saved as pickle files for efficient loading and saving.
"""
import pickle
import json
from pathlib import Path
import hashlib
import warnings

def _get_hazard_dir_name(hazard_dir):
    """Helper function to consistently get hazard directory name."""
    if hazard_dir:
        return Path(hazard_dir).name
    return "unknown"

def get_asset_centroid_hash(temp_gdf):
    n_assets = len(temp_gdf)
    bbox = temp_gdf.total_bounds  # [minx, miny, maxx, maxy]
    bbox_str = ",".join([f"{v:.6f}" for v in bbox])
    hashkey = hashlib.md5(bbox_str.encode()).hexdigest()
    return f"n{n_assets}_{hashkey}"

def create_grid_accessibility_cache_key(hazard_map, flood_threshold, hazard_dir=None):
    """
    Create a unique cache key for grid accessibility results.
    
    Args:
        hazard_map (str or Path): Path to hazard map
        flood_threshold (float): Flood threshold used
        hazard_dir (str or Path, optional): Hazard directory path for naming
    
    Returns:
        str: Unique cache key
    """
    hazard_dir_name = _get_hazard_dir_name(hazard_dir) if hazard_dir else Path(hazard_map).parent.name
    hazard_file = Path(hazard_map).stem  # filename without extension
    
    cache_key = f"grid_accessibility_{hazard_dir_name}_{hazard_file}_{flood_threshold}"
    return cache_key

def save_grid_accessibility_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save grid accessibility cache to disk as JSON and pickle files.
    
    Args:
        cache_dict (dict): Dictionary containing grid accessibility results
        cache_dir (Path or str): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    
    # Create hazard-specific subdirectory
    hazard_subdir = cache_dir / f"grid_accessibility_{hazard_dir_name}"
    hazard_subdir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    pickle_file = hazard_subdir / "grid_accessibility_cache.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    # Save metadata as JSON for easier inspection
    metadata = {}
    for key, result in cache_dict.items():
        if isinstance(result, dict):
            metadata[key] = {
                'timestamp': result.get('timestamp', 'unknown'),
                'flood_threshold': result.get('flood_threshold', 'unknown'),
                'hazard_map': str(result.get('hazard_map', 'unknown')),
                'grid_size': len(result.get('grid_results', [])) if result.get('grid_results') else 0
            }
    
    json_file = hazard_subdir / "grid_accessibility_metadata.json"
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"Saved grid accessibility cache: {len(cache_dict)} entries to {hazard_subdir}")

def load_grid_accessibility_cache(cache_dir, hazard_dir=None):
    """
    Load grid accessibility cache from disk.
    
    Args:
        cache_dir (Path or str): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Grid accessibility cache dictionary
    """
    cache_dir = Path(cache_dir)
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    
    hazard_subdir = cache_dir / f"grid_accessibility_{hazard_dir_name}"
    pickle_file = hazard_subdir / "grid_accessibility_cache.pkl"
    
    if pickle_file.exists():
        try:
            with open(pickle_file, 'rb') as f:
                cache_dict = pickle.load(f)
            print(f"Loaded grid accessibility cache: {len(cache_dict)} entries from {hazard_subdir}")
            return cache_dict
        except Exception as e:
            print(f"Error loading grid accessibility cache: {e}")
            return {}
    else:
        print(f"No grid accessibility cache found at {pickle_file}")
        return {}

def create_accessibility_cache_key(timestep, threshold, hazard_dir, accessibility_model=None):
    """
    Create cache key for single timestep accessibility.
    
    Args:
        timestep (int): Simulation timestep
        threshold (float): Hazard threshold value
        hazard_dir (str or Path): Hazard directory path
        accessibility_model (object, optional): Accessibility model instance
        
    Returns:
        str: Unique cache key
    """
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    model_type = type(accessibility_model).__name__ if accessibility_model else "default"
    return f"accessibility_{hazard_dir_name}_{timestep}_{threshold}_{model_type}"

def save_accessibility_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save accessibility cache to disk.
    
    Args:
        cache_dict (dict): Dictionary containing accessibility results
        cache_dir (Path or str): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"accessibility_cache_{hazard_dir_name}.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved accessibility cache: {len(cache_dict)} entries to {cache_file}")

def load_accessibility_cache(cache_dir, hazard_dir=None):
    """
    Load accessibility cache from disk.
    
    Args:
        cache_dir (Path or str): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Accessibility cache dictionary
    """
    cache_dir = Path(cache_dir)
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"accessibility_cache_{hazard_dir_name}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_dict = pickle.load(f)
            print(f"Loaded accessibility cache: {len(cache_dict)} entries from {cache_file}")
            return cache_dict
        except Exception as e:
            print(f"Error loading accessibility cache: {e}")
            return {}
    else:
        print(f"No accessibility cache found at {cache_file}")
        return {}

def create_hazard_extraction_cache_key(hazard_map_path, map_counter, extraction_method, gdf_hash, hazard_dir):
    """
    Create a unique cache key for hazard extraction based on hazard map and asset geometry.
    
    Args:
        hazard_map_path (str or Path): Path to hazard map file
        map_counter (int): Map index/counter
        extraction_method (str): Method used for hazard extraction
        gdf_hash (str): Hash of the geodataframe
        hazard_dir (str or Path): Hazard directory path
        
    Returns:
        str: Unique cache key
    """
    # Get hazard file identifier
    hazard_file = Path(hazard_map_path).stem  # e.g., "hazard_01"
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)

    return f"hazard_extraction_{extraction_method}_{hazard_dir_name}_{hazard_file}_map{map_counter:02d}_{gdf_hash}"

def save_hazard_extraction_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save hazard extraction cache to disk with hazard directory context.
    
    Args:
        cache_dict (dict): Dictionary containing hazard extraction results
        cache_dir (Path or str): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"hazard_extraction_cache_{hazard_dir_name}.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved hazard extraction cache: {len(cache_dict)} entries to {cache_file}")

def load_hazard_extraction_cache(cache_dir, hazard_dir=None):
    """
    Load hazard extraction cache from disk.
    
    Args:
        cache_dir (Path or str): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Hazard extraction cache dictionary
    """
    cache_dir = Path(cache_dir)
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"hazard_extraction_cache_{hazard_dir_name}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_dict = pickle.load(f)
            print(f"Loaded hazard extraction cache: {len(cache_dict)} entries from {cache_file}")
            return cache_dict
        except Exception as e:
            print(f"Error loading hazard extraction cache: {e}")
            return {}
    else:
        print(f"No hazard extraction cache found at {cache_file}")
        return {}

def create_island_cache_key(hazard_column, hazard_threshold, asset_hash=None):
    """
    Create a standardized cache key for island assignment, including asset count and bounds.
    
    Args:
        hazard_column (str): The hazard column name (e.g., 'EV1_ma')
        hazard_threshold (float): The hazard threshold value (e.g., 0.2)
        asset_hash (str, optional): Hash of the asset geodataframe for uniqueness
        
    Returns:
        str: Standardized cache key
    """
    if asset_hash is None:
        asset_hash = "unknown_assets"

    key = f"island_assignment_{hazard_column}_{hazard_threshold}_{asset_hash}"

    return key

def save_island_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save island assignment cache to disk.
    
    Args:
        cache_dict (dict): Dictionary containing island assignment results
        cache_dir (Path or str): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
        
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_dict, f)
        print(f"Saved island cache: {len(cache_dict)} entries to {cache_file}")
    except Exception as e:
        print(f"ERROR: Failed to save island cache: {e}")

def load_island_cache(cache_dir, hazard_dir=None):
    """
    Load island assignment cache from disk.
    
    Args:
        cache_dir (Path or str): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Island assignment cache dictionary
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)  # Added this line from initialize_island_cache
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_dict = pickle.load(f)
            print(f"Loaded island cache: {len(cache_dict)} entries from {cache_file}")
            return cache_dict
        except Exception as e:
            print(f"Error loading island cache: {e}")
            return {}
    else:
        print(f"No island cache found at {cache_file}")
        return {}

def create_overlap_cache_key(prev_map, current_map, hazard_threshold, hazard_dir=None):
    """
    Create cache key for overlap percentages between timesteps.
    
    Args:
        prev_map (str or int): Previous map identifier
        current_map (str or int): Current map identifier
        hazard_threshold (float): Hazard threshold value
        hazard_dir (str or Path, optional): Hazard directory path
        
    Returns:
        str: Unique cache key
    """
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    return f"overlap_{hazard_dir_name}_{prev_map}_{current_map}_{hazard_threshold}"

def save_overlap_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save overlap cache as pickle.
    
    Args:
        cache_dict (dict): Dictionary containing overlap results
        cache_dir (Path or str): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"overlap_cache_{hazard_dir_name}.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved overlap cache: {len(cache_dict)} entries to {cache_file}")

def load_overlap_cache(cache_dir, hazard_dir=None):
    """
    Load overlap cache pickle.
    
    Args:
        cache_dir (Path or str): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Overlap cache dictionary
    """
    cache_dir = Path(cache_dir)
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    cache_file = cache_dir / f"overlap_cache_{hazard_dir_name}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_dict = pickle.load(f)
            print(f"Loaded overlap cache: {len(cache_dict)} entries from {cache_file}")
            return cache_dict
        except Exception as e:
            print(f"Error loading overlap cache: {e}")
            return {}
    else:
        return {}
    
def load_simulation_caches(cache_dir, hazard_dir=None):
    """
    Load all simulation caches from the cache directory.
    
    Args:
        cache_dir (Path or str): Directory where cache files are stored
        hazard_dir (Path or str, optional): Hazard directory for cache naming
    
    Returns:
        dict: Dictionary with loaded caches with keys:
            'accessibility_cache': Loaded accessibility cache or None
            'hazard_extraction_cache': Loaded hazard extraction cache or None  
            'overlap_cache': Loaded overlap cache or None
            'island_cache': Loaded island cache or None
    """
    # Convert to Path objects
    cache_dir = Path(cache_dir)
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize empty caches
    caches = {
        'accessibility_cache': None,
        'hazard_extraction_cache': None,
        'overlap_cache': None,
        'island_cache': None
    }
    
    print("Loading simulation caches...")
    
    # Load accessibility cache
    try:
        caches['accessibility_cache'] = load_accessibility_cache(cache_dir, hazard_dir)
    except Exception as e:
        print(f"Note: Could not load accessibility cache: {e}")
    
    # Load hazard extraction cache
    try:
        caches['hazard_extraction_cache'] = load_hazard_extraction_cache(cache_dir, hazard_dir)
    except Exception as e:
        print(f"Note: Could not load hazard extraction cache: {e}")
    
    # Load overlap cache
    try:
        caches['overlap_cache'] = load_overlap_cache(cache_dir, hazard_dir)
    except Exception as e:
        print(f"Note: Could not load overlap cache: {e}")
    
    # Load island cache
    try:
        caches['island_cache'] = load_island_cache(cache_dir, hazard_dir)
    except Exception as e:
        print(f"Note: Could not load island cache: {e}")
    
    return caches

def save_simulation_caches(cache_updated, cache_dir, hazard_dir=None):
    """
    Save all updated simulation caches to the cache directory.
    
    Args:
        cache_updated (dict): Dictionary containing updated caches with keys like 
                             'accessibility_cache', 'hazard_extraction_cache', etc.
        cache_dir (Path or str): Directory where cache files will be stored
        hazard_dir (Path or str, optional): Hazard directory for cache naming
    
    Returns:
        dict: Dictionary with saved cache stats (counts, paths, etc.)
    """
    saved_stats = {}
    
    if not cache_updated:
        print("No caches to update.")
        return saved_stats
    
    print("\nSaving updated simulation caches...")
    
    # Convert to Path object
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    hazard_dir_name = _get_hazard_dir_name(hazard_dir)
    
    # Save accessibility cache if updated
    if 'accessibility_cache' in cache_updated:
        try:
            accessibility_cache = cache_updated['accessibility_cache']
            save_accessibility_cache(accessibility_cache, cache_dir, hazard_dir)
            saved_stats['accessibility_cache'] = {
                'count': len(accessibility_cache),
                'path': str(cache_dir / f"accessibility_cache_{hazard_dir_name}.pkl")
            }
            print(f"Saved accessibility cache with {len(accessibility_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not save accessibility cache: {e}")
    
    # Save hazard extraction cache if updated
    if 'hazard_extraction_cache' in cache_updated:
        try:
            hazard_extraction_cache = cache_updated['hazard_extraction_cache']
            save_hazard_extraction_cache(hazard_extraction_cache, cache_dir, hazard_dir)
            saved_stats['hazard_extraction_cache'] = {
                'count': len(hazard_extraction_cache),
                'path': str(cache_dir / f"hazard_extraction_cache_{hazard_dir_name}.pkl")
            }
            print(f"Saved hazard extraction cache with {len(hazard_extraction_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not save hazard extraction cache: {e}")
    
    # Save overlap cache if updated
    if 'overlap_cache' in cache_updated:
        try:
            overlap_cache = cache_updated['overlap_cache']
            save_overlap_cache(overlap_cache, cache_dir, hazard_dir)
            saved_stats['overlap_cache'] = {
                'count': len(overlap_cache),
                'path': str(cache_dir / f"overlap_cache_{hazard_dir_name}.pkl")
            }
            print(f"Saved overlap cache with {len(overlap_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not save overlap cache: {e}")
    
    # Save island cache if updated
    if 'island_cache' in cache_updated:
        try:
            island_cache = cache_updated['island_cache']
            save_island_cache(island_cache, cache_dir, hazard_dir)
            saved_stats['island_cache'] = {
                'count': len(island_cache),
                'path': str(cache_dir / f"island_cache_{hazard_dir_name}.pkl")
            }
            print(f"Saved island cache with {len(island_cache)} entries")
        except Exception as e:
            print(f"Warning: Could not save island cache: {e}")
    
    return saved_stats