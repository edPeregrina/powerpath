"""
Functions for loading assets and hazard maps
Assets are geodataframes of electricity stations and hazard maps are lists of file paths to .tif files.
"""
import geopandas as gpd
import pandas as pd
import os

# Data Loading Functions
def load_electricity_assets(electricity_dir, asset_types):
    """Load electricity assets from shapefile
    
    Args:
        electricity_dir (Path): Path to the directory containing electricity station shapefiles
        asset_types (list): List of asset types to load (e.g., ['ls', 'ms', 'msls'])

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of electricity assets with type column

    """
    # List station shapefiles in the electricity directory
    station_files = [
        f for f in os.listdir(electricity_dir) if f.endswith('.shp') and 'station' in f
    ]

    # Fix: Use exact matching with underscore to avoid 'ls' matching 'msls'
    # Sort asset_types by length (longest first) to prioritize longer matches
    asset_types_sorted = sorted(asset_types, key=len, reverse=True)
    
    matched_files = []
    for station_file in station_files:
        filename_prefix = station_file.split('_')[0]
        if filename_prefix in asset_types_sorted:
            matched_files.append(station_file)
    
    station_files = matched_files

    print(f"Found {len(station_files)} electricity station files matching types {asset_types}")
    
    # Add debug information to see what files were found
    print(f"All .shp files in directory: {[f for f in os.listdir(electricity_dir) if f.endswith('.shp')]}")
    print(f"Files with 'station': {[f for f in os.listdir(electricity_dir) if f.endswith('.shp') and 'station' in f]}")
    print(f"Final matched files: {station_files}")
    combined_assets = []
    
    for station_file in station_files:
        station_path = electricity_dir / station_file
        if station_path.exists():
            print(f"Loading electricity assets from {station_file}")
            gdf = gpd.read_file(station_path)
            
            # Ensure proper CRS
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            
            # Add type column based on filename, loading msls first for naming practicality
            if 'msls_' in station_file:
                gdf['type'] = 'msls'
            elif 'ms_' in station_file:
                gdf['type'] = 'ms'
            elif 'ls_' in station_file:
                gdf['type'] = 'ls'
            
            combined_assets.append(gdf)
            print(f"Loaded {len(gdf)} {gdf['type'].iloc[0]} assets")
    
    if combined_assets:
        # Combine all assets into a single GeoDataFrame
        gdf_assets = gpd.GeoDataFrame(pd.concat(combined_assets, ignore_index=True))
        print(f"Combined total: {len(gdf_assets)} electricity assets")
        print(f"Asset types: {gdf_assets['type'].value_counts().to_dict()}")
        return gdf_assets
    
    raise FileNotFoundError(f"No electricity station files found in {electricity_dir}")

def load_hazard_maps(hazard_dir, max_days=None):
    """Find and load hazard map files from directory
    Args:
        hazard_dir (Path): Path to the directory containing hazard map .tif files
        max_days (int, optional): Maximum number of days to load. If None, all available days are loaded.

    Returns:
        list: List of file paths to hazard map .tif files
    """
    
    # Find all .tif files in hazard directory
    hazard_files = list(hazard_dir.glob("*.tif"))
    
    if not hazard_files:
        raise FileNotFoundError(f"No .tif hazard files found in {hazard_dir}")
    
    # Sort files to ensure consistent ordering
    # If more than 10 files, must be sorted based on integer value in filename, not alphabetically
    def extract_int_from_stem(path_obj):
        digits = ''.join(filter(str.isdigit, path_obj.stem))
        return int(digits) if digits else 0

    if len(hazard_files) > 9:
        hazard_files = sorted(hazard_files, key=extract_int_from_stem)
    else:
        hazard_files = sorted(hazard_files, key=lambda x: x.name)
    
    if max_days:
        hazard_files = hazard_files[:max_days]
    
    print(f"Found {len(hazard_files)} hazard map files")
    return [str(f) for f in hazard_files]