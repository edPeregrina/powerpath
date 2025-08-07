"""
Functions for loading assets and hazard maps
Assets are geodataframes of electricity stations and hazard maps are lists of file paths to .tif files.
"""
import geopandas as gpd
import pandas as pd

# Data Loading Functions
def load_electricity_assets(electricity_dir):
    """Load electricity assets from shapefile
    
    Args:
        electricity_dir (Path): Path to the directory containing electricity station shapefiles

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of electricity assets with type column
   
    """
    
    station_files = [
        'ls_stations_clipped.shp',
        # 'ms_stations_clipped.shp', 
        'msls_stations_clipped.shp'
    ]
    
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
    hazard_files.sort()
    
    if max_days:
        hazard_files = hazard_files[:max_days]
    
    print(f"Found {len(hazard_files)} hazard map files")
    return [str(f) for f in hazard_files]