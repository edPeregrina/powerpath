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
    # Get hazard directory name for cache organization
    if hazard_dir is not None:
        hazard_dir_name = Path(hazard_dir).name
    else:
        hazard_dir_name = Path(hazard_map).parent.name
    
    hazard_file = Path(hazard_map).stem  # filename without extension
    
    cache_key = f"grid_accessibility_{hazard_dir_name}_{hazard_file}_{flood_threshold}"
    return cache_key

def save_grid_accessibility_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save grid accessibility cache to disk as JSON and pickle files.
    
    Args:
        cache_dict (dict): Dictionary containing grid accessibility results
        cache_dir (Path): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hazard-specific subdirectory
    if hazard_dir:
        hazard_subdir = cache_dir / f"grid_accessibility_{Path(hazard_dir).name}"
    else:
        hazard_subdir = cache_dir / "grid_accessibility"
    
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
        cache_dir (Path): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Grid accessibility cache dictionary
    """
    if hazard_dir:
        hazard_subdir = cache_dir / f"grid_accessibility_{Path(hazard_dir).name}"
    else:
        hazard_subdir = cache_dir / "grid_accessibility"
    
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
    """Create cache key for single timestep accessibility."""
    hazard_dir_name = Path(hazard_dir).name if hazard_dir else "unknown"
    model_type = type(accessibility_model).__name__ if accessibility_model else "default"
    return f"accessibility_{hazard_dir_name}_{timestep}_{threshold}_{model_type}"

def save_accessibility_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save accessibility cache to disk.
    
    Args:
        cache_dict (dict): Dictionary containing accessibility results
        cache_dir (Path): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hazard-specific cache file
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"accessibility_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "accessibility_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved accessibility cache: {len(cache_dict)} entries to {cache_file}")

def load_accessibility_cache(cache_dir, hazard_dir=None):
    """
    Load accessibility cache from disk.
    
    Args:
        cache_dir (Path): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Accessibility cache dictionary
    """
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"accessibility_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "accessibility_cache.pkl"
    
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

def create_hazard_extraction_cache_key(hazard_map_path, day_counter, extraction_method, gdf_hash, hazard_dir):
    """Create a unique cache key for hazard extraction based on hazard map, day, and asset geometry."""
    # Get hazard file identifier
    hazard_file = Path(hazard_map_path).stem  # e.g., "hazard_day_01"
    
    # Include hazard directory name for differentiation
    hazard_dir_name = Path(hazard_dir).name if hazard_dir else "default"
    
    return f"hazard_extraction_{extraction_method}_{hazard_dir_name}_{hazard_file}_day{day_counter:02d}_{gdf_hash}"

def save_hazard_extraction_cache(cache_dict, cache_dir, hazard_dir=None):
    """Save hazard extraction cache to disk with hazard directory context."""
    cache_dir = Path(cache_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hazard-specific cache file
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"hazard_extraction_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "hazard_extraction_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved hazard extraction cache: {len(cache_dict)} entries to {cache_file}")

def load_hazard_extraction_cache(cache_dir, hazard_dir=None):
    """Load hazard extraction cache from disk."""
    cache_dir = Path(cache_dir) / "cache"
    
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"hazard_extraction_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "hazard_extraction_cache.pkl"
    
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

def save_island_cache_silent(cache_dict, cache_dir, hazard_dir=None):
    """Save island assignment cache to disk without printing message."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "island_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    # No print message for silent version

def save_island_cache(cache_dict, cache_dir, hazard_dir=None):
    """
    Save island assignment cache to disk.
    
    Args:
        cache_dict (dict): Dictionary containing island assignment results
        cache_dir (Path): Directory to save cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hazard-specific cache file
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "island_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved island cache: {len(cache_dict)} entries to {cache_file}")

def load_island_cache(cache_dir, hazard_dir=None):
    """
    Load island assignment cache from disk.
    
    Args:
        cache_dir (Path): Directory containing cache files
        hazard_dir (str or Path, optional): Hazard directory for organization
    
    Returns:
        dict: Island assignment cache dictionary
    """
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"island_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "island_cache.pkl"
    
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

def create_overlap_cache_key(prev_day, current_day, hazard_threshold, hazard_dir=None):
    """Create cache key for overlap percentages between timesteps."""
    hazard_dir_name = Path(hazard_dir).name if hazard_dir else "unknown"
    return f"overlap_{hazard_dir_name}_{prev_day}_{current_day}_{hazard_threshold}"

def save_overlap_cache(cache_dict, cache_dir, hazard_dir=None):
    """Save overlap cache as pickle."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"overlap_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "overlap_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_dict, f)
    
    print(f"Saved overlap cache: {len(cache_dict)} entries to {cache_file}")

def load_overlap_cache(cache_dir, hazard_dir=None):
    """Load overlap cache pickle."""
    if hazard_dir:
        hazard_dir_name = Path(hazard_dir).name
        cache_file = cache_dir / f"overlap_cache_{hazard_dir_name}.pkl"
    else:
        cache_file = cache_dir / "overlap_cache.pkl"
    
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