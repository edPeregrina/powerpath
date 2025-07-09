"""
Accessibility Analysis for Electrical Infrastructure

This module provides functions to analyze the accessibility of electrical infrastructure 
(substations, power lines, etc.) during flooding events. It combines road network analysis
with hazard mapping to determine which assets can be reached by repair crews.

Key Functions:
- initialize_project_paths(): Set up project directory structure
- create_baseline_grid_analysis(): Create baseline accessibility grid (run once)
- accessibility_model(): Callable function to assess asset accessibility given a hazard map
- run_hazard_analysis(): Run detailed road flooding analysis for specific hazards

Typical Usage:
1. Import this module in a Jupyter notebook
2. Use accessibility_model(asset_geometries, hazard_map_path, **other arguments) for each scenario

"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import rasterio
from tqdm import tqdm
import networkx as nx
from shapely.geometry import box, Point
from pyproj import Transformer
import warnings

# RA2CE imports for hazard analysis
from ra2ce.network.network_config_data.enums.aggregate_wl_enum import AggregateWlEnum
from ra2ce.network.network_config_data.network_config_data import (
    HazardSection,
    NetworkConfigData,
    NetworkSection,
)
from ra2ce.ra2ce_handler import Ra2ceHandler
from ra2ce.network.network_config_data.enums.network_type_enum import NetworkTypeEnum
from ra2ce.network.network_config_data.enums.road_type_enum import RoadTypeEnum

# Global cache for analysis results
_grid_analysis_cache = None
_baseline_analysis_cache = None

def run_ra2ce_hazard_analysis(root_dir, static_path, output_path, study_polygon, hazard_files):
    """
    Run RA2CE analysis to overlay hazard data on road network and generate EV1 variables.
    
    Parameters:
    -----------
    root_dir : Path
        Root directory for the project
    static_path : Path
        Static data directory
    output_path : Path
        Output directory for results
    study_polygon : Polygon
        Study area polygon
    hazard_files : list
        List of processed hazard file paths
        
    Returns:
    --------
    None (saves results to files)
    """
    # Define network section
    _network_section = NetworkSection(
        network_type=NetworkTypeEnum.DRIVE,
        polygon=study_polygon,
        save_gpkg=True,
        road_types=[
            RoadTypeEnum.MOTORWAY, RoadTypeEnum.MOTORWAY_LINK,
            RoadTypeEnum.PRIMARY, RoadTypeEnum.PRIMARY_LINK,
            RoadTypeEnum.TRUNK, RoadTypeEnum.SECONDARY,
            RoadTypeEnum.SECONDARY_LINK, RoadTypeEnum.TERTIARY,
            RoadTypeEnum.RESIDENTIAL
        ], 
    )

    # Define hazard section for each hazard file
    for hazard_file in hazard_files:
        _hazard_section = HazardSection(
            hazard_map=[hazard_file],
            hazard_id=None,
            hazard_field_name="waterdepth",
            aggregate_wl=AggregateWlEnum.MAX,
            hazard_crs="EPSG:4326",
            overlay_segmented_network=False
        )

        _network_config_data = NetworkConfigData(
            root_path=root_dir,
            static_path=static_path,
            output_path=output_path,
            network=_network_section,
            hazard=_hazard_section
        )

        # Run RA2CE analysis
        _handler = Ra2ceHandler.from_config(_network_config_data, analysis=None)
        _handler.configure()
        _handler.run_analysis()
        
        print(f"RA2CE analysis completed for {hazard_file.name}")

def load_hazard_roads(output_path):
    """
    Load the hazard-analyzed road network from RA2CE output.
    
    Parameters:
    -----------
    output_path : Path
        Path to RA2CE output directory
        
    Returns:
    --------
    gpd.GeoDataFrame
        Road network with EV1_fr and EV1_ma variables
    """
    hazard_road_path = output_path.joinpath("base_graph_hazard_edges.gpkg")
    if not hazard_road_path.exists():
        raise FileNotFoundError(f"Hazard road file not found: {hazard_road_path}")
    
    roads_hazard = gpd.read_file(hazard_road_path, driver='GPKG')
    
    # Verify required columns exist
    required_cols = ['EV1_fr', 'EV1_ma']
    missing_cols = [col for col in required_cols if col not in roads_hazard.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in hazard roads: {missing_cols}")
    
    return roads_hazard

def calculate_road_lengths_optimized(grid_cells, road_network, fr_threshold, ma_threshold):
    """
    Calculate road lengths affected by flooding within grid cells.
    
    Parameters:
    -----------
    grid_cells : gpd.GeoDataFrame
        Grid cells (polygons) to analyze
    road_network : gpd.GeoDataFrame
        Road network with flood attributes (EV1_fr, EV1_ma)
    fr_threshold : float
        Minimum fraction flooded threshold (0-1)
    ma_threshold : float
        Minimum water depth threshold (meters)
    
    Returns:
    --------
    gpd.GeoDataFrame
        Grid cells with added road length statistics
    """
    # Make a copy to avoid modifying the original
    grid_result = grid_cells.copy()
    
    # Ensure both GeoDataFrames use the same CRS (projected for accurate length calculations)
    if grid_result.crs != "EPSG:28992":
        grid_result = grid_result.to_crs("EPSG:28992")
    if road_network.crs != "EPSG:28992":
        road_network = road_network.to_crs("EPSG:28992")

    # Spatial join once
    joined = gpd.sjoin(road_network, grid_result, how="inner", predicate="within")

    if joined.empty:
        print("Warning: No roads found within grid cells")
        # Initialize empty columns
        grid_result["total_road_length_m"] = 0.0
        length_col = f"length_{int(fr_threshold*100)}%_fr_{ma_threshold}"
        perc_col = f"%_length_{int(fr_threshold*100)}%_fr_{ma_threshold}"
        grid_result[length_col] = 0.0
        grid_result[perc_col] = 0.0
        grid_result["mean_max_water_depth"] = np.nan
        return grid_result

    # Group by grid cell index
    grouped = joined.groupby("index_right")

    # Initialize result columns with zeros for all grid cells
    grid_result["total_road_length_m"] = 0.0
    length_col = f"length_{int(fr_threshold*100)}%_fr_{ma_threshold}"
    perc_col = f"%_length_{int(fr_threshold*100)}%_fr_{ma_threshold}"
    grid_result[length_col] = 0.0
    grid_result[perc_col] = 0.0
    grid_result["mean_max_water_depth"] = np.nan

    # Calculate total road length per grid cell
    total_lengths = grouped["geometry"].apply(lambda g: g.length.sum())
    grid_result.loc[total_lengths.index, "total_road_length_m"] = total_lengths

    # Filtered roads (meeting flood criteria)
    filtered = joined[
        (joined["EV1_fr"] > fr_threshold) &
        (joined["EV1_ma"] > ma_threshold)
    ]
    
    if not filtered.empty:
        filtered_grouped = filtered.groupby("index_right")
        affected_lengths = filtered_grouped["geometry"].apply(lambda g: g.length.sum())
        grid_result.loc[affected_lengths.index, length_col] = affected_lengths

    # Calculate percentage (avoid division by zero)
    mask = grid_result["total_road_length_m"] > 0
    grid_result.loc[mask, perc_col] = (
        grid_result.loc[mask, length_col] / grid_result.loc[mask, "total_road_length_m"]
    ) * 100

    # Mean water depth per grid cell
    mean_depths = grouped["EV1_ma"].mean()
    grid_result.loc[mean_depths.index, "mean_max_water_depth"] = mean_depths

    # Convert back to original CRS if needed
    if grid_result.crs != grid_cells.crs:
        grid_result = grid_result.to_crs(grid_cells.crs)

    return grid_result

def complete_flood_road_analysis(grid_cells, root_dir, static_path, output_path, 
                                study_polygon, hazard_files, fr_threshold=0.3, ma_threshold=0.2):
    """
    Complete workflow: Run RA2CE analysis and calculate road impact statistics.
    
    Parameters:
    -----------
    grid_cells : gpd.GeoDataFrame
        Grid cells to analyze
    root_dir : Path
        Root directory for the project
    static_path : Path
        Static data directory
    output_path : Path
        Output directory
    study_polygon : Polygon
        Study area polygon
    hazard_files : list
        List of hazard file paths
    fr_threshold : float, default 0.3
        Fraction flooded threshold
    ma_threshold : float, default 0.2
        Water depth threshold in meters
        
    Returns:
    --------
    gpd.GeoDataFrame
        Grid cells with road impact analysis
    """
    print("Step 1: Running RA2CE hazard analysis...")
    run_ra2ce_hazard_analysis(root_dir, static_path, output_path, study_polygon, hazard_files)
    
    print("Step 2: Loading hazard-analyzed roads...")
    road_network = load_hazard_roads(output_path)
    
    print("Step 3: Calculating road length impacts...")
    result = calculate_road_lengths_optimized(grid_cells, road_network, fr_threshold, ma_threshold)
    
    print(f"Analysis complete! Processed {len(result)} grid cells.")
    return result

def initialize_project_paths(project_root=None):
    """
    Initialize project paths. Call this function before using other functions.
    
    Parameters:
    -----------
    project_root : Path or str, optional
        Root directory of the project. If None, will try to find it automatically.
        
    Returns:
    --------
    dict
        Dictionary containing all relevant paths
    """
    if project_root is None:
        cwd = Path.cwd()
        # Try to find the project root by looking for 'data' directory
        if (cwd / 'data').exists():
            root_dir = cwd / 'data'
        elif (cwd.parent / 'data').exists():
            root_dir = cwd.parent / 'data'
        else:
            raise FileNotFoundError("Could not find 'data' directory. Please specify project_root parameter.")
    else:
        root_dir = Path(project_root) / 'data'
        if not root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {root_dir}")
    
    paths = {
        'root_dir': root_dir,
        'static_path': root_dir / "static",
        'hazard_path': root_dir / "static" / "hazard",
        'network_path': root_dir / "static" / "network", 
        'output_path': root_dir / "static" / "output_graph"
    }
    
    return paths

def create_baseline_grid_analysis(project_root=None, cell_size=500, force_recreate=False):
    """
    Create baseline grid analysis without hazards. This should be run once to cache results.
    
    Parameters:
    -----------
    project_root : Path or str, optional
        Root directory of the project
    cell_size : int, default 500
        Grid cell size in meters
    force_recreate : bool, default False
        Force recreation even if cache exists
        
    Returns:
    --------
    gpd.GeoDataFrame
        Grid with baseline accessibility analysis
    """
    global _baseline_analysis_cache
    
    if _baseline_analysis_cache is not None and not force_recreate:
        print("Using cached baseline analysis")
        return _baseline_analysis_cache.copy()
    
    print("Creating baseline grid analysis...")
    paths = initialize_project_paths(project_root)
    
    # Load study area
    extent_path = paths['network_path'] / "try_study_area_larger.shp"
    if not extent_path.exists():
        raise FileNotFoundError(f"Study area file not found: {extent_path}")
    
    extent = gpd.read_file(extent_path)
    shapely_polygon = extent.geometry.iloc[0]
    
    # Create grid
    study_area_rd = extent.to_crs("EPSG:28992")
    minx, miny, maxx, maxy = study_area_rd.total_bounds
    cols = int((maxx - minx) / cell_size)
    rows = int((maxy - miny) / cell_size)
    
    cells = []
    for i in range(cols):
        for j in range(rows):
            x = minx + i * cell_size
            y = miny + j * cell_size
            cell = box(x, y, x + cell_size, y + cell_size)
            if study_area_rd.geometry.unary_union.intersects(cell):
                cells.append(cell)
    
    grid = gpd.GeoDataFrame(geometry=cells, crs="EPSG:28992")
    # Calculate centroid in projected CRS before converting to WGS84
    grid["centroid_rd"] = grid.geometry.centroid
    grid = grid.to_crs("EPSG:4326")
    # Convert centroid to WGS84
    centroid_gdf = gpd.GeoDataFrame(geometry=grid["centroid_rd"], crs="EPSG:28992")
    centroid_wgs84 = centroid_gdf.to_crs("EPSG:4326")
    grid["centroid"] = centroid_wgs84.geometry
    # Drop the temporary column
    grid = grid.drop(columns=["centroid_rd"])
    
    # Load road network and calculate baseline distances
    road_network_path = paths['output_path'] / "base_graph.p"
    if not road_network_path.exists():
        raise FileNotFoundError(f"Road network file not found: {road_network_path}")
    
    with open(road_network_path, "rb") as f:
        G = pickle.load(f)
    
    G = project_graph_to_meters(G)
    node_gdf = prepare_node_gdf(G)
    
    # Calculate baseline distances
    distances = []
    for idx, row in tqdm(grid.iterrows(), total=len(grid), desc="Calculating baseline distances"):
        center = row["centroid"]
        neighbors = grid[grid.geometry.touches(row.geometry) | (grid.index == idx)]
        
        try:
            center_node = get_nearest_node(center, node_gdf)
            if center_node is None:
                distances.append(np.nan)
                continue
                
            dists = []
            for _, n_row in neighbors.iterrows():
                if n_row["centroid"].equals(center):
                    continue
                try:
                    neighbor_node = get_nearest_node(n_row["centroid"], node_gdf)
                    if neighbor_node is None:
                        continue
                    length = nx.shortest_path_length(G, source=neighbor_node, target=center_node, weight="length")
                    dists.append(length)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            
            avg_dist = np.mean(dists) if dists else np.nan
        except Exception as e:
            print(f"Error calculating distance for cell {idx}: {e}")
            avg_dist = np.nan
        
        distances.append(avg_dist)
    
    grid["avg_distance_no_flood"] = distances
    
    # Cache the result
    _baseline_analysis_cache = grid.copy()
    print(f"Baseline analysis complete! Created grid with {len(grid)} cells.")
    
    return grid

def accessibility_model(asset_geometries, hazard_map_path, hazard_values=None, hazard_threshold=0.2, 
                       road_flood_threshold=50.0, project_root=None):
    """
    Accessibility model to determine if electrical assets are accessible based on road flooding.
    
    Parameters:
    -----------
    asset_geometries : gpd.GeoSeries or gpd.GeoDataFrame
        Geometries of the electrical assets (stations, substations, etc.)
    hazard_map_path : str or Path
        Path to the hazard raster file (.tif)
    hazard_threshold : float, default 0.2
        Water depth threshold in meters for local flooding
    road_flood_threshold : float, default 50.0
        Percentage of roads flooded threshold for area accessibility. If more than this percentage of roads in a grid cell are flooded, the area is considered inaccessible.
    project_root : Path or str, optional
        Root directory of the project
        
    Returns:
    --------
    pd.Series
        Boolean series indicating accessibility for each asset (True = accessible, False = not accessible)
    """
    global _grid_analysis_cache
    
    try:
        # Initialize paths
        paths = initialize_project_paths(project_root)
        
        # Ensure we have baseline analysis
        if _baseline_analysis_cache is None:
            print("Creating baseline analysis (this may take a while)...")
            create_baseline_grid_analysis(project_root)
        
        # Get hazard analysis for this specific hazard map
        if _grid_analysis_cache is None or not hasattr(_grid_analysis_cache, '_hazard_path') or \
           _grid_analysis_cache._hazard_path != str(hazard_map_path):
            print(f"Running hazard analysis for {hazard_map_path}...")
            _grid_analysis_cache = run_hazard_analysis(hazard_map_path, project_root)
            _grid_analysis_cache._hazard_path = str(hazard_map_path)
        
        # Convert asset geometries to GeoDataFrame if needed
        if hasattr(asset_geometries, 'geometry'):
            # It's already a GeoDataFrame
            assets_gdf = asset_geometries.copy()
        else:
            # It's a GeoSeries, convert to GeoDataFrame
            assets_gdf = gpd.GeoDataFrame(geometry=asset_geometries.copy(), crs="EPSG:4326")
        
        # Ensure CRS is WGS84
        if assets_gdf.crs != "EPSG:4326":
            assets_gdf = assets_gdf.to_crs("EPSG:4326")
        
        # Extract hazard values at asset locations if not provided
        if hazard_values is None:
            hazard_values = extract_hazard_values_at_points(assets_gdf.geometry, hazard_map_path)
        
        # Check local accessibility (assets not flooded above threshold)
        local_accessible = hazard_values <= hazard_threshold
        
        # Check road accessibility using spatial join with grid analysis
        try:
            # Ensure assets_gdf has the right structure for spatial join
            if not isinstance(assets_gdf, gpd.GeoDataFrame):
                assets_gdf = gpd.GeoDataFrame(geometry=assets_gdf, crs="EPSG:4326")
            
            # Convert both to projected CRS for accurate spatial join
            assets_proj = assets_gdf.to_crs("EPSG:28992")
            grid_proj = _grid_analysis_cache.to_crs("EPSG:28992")
            
            joined = gpd.sjoin_nearest(assets_proj, grid_proj, how="left", max_distance=1000)
            
            # Use road flooding percentage as accessibility metric
            road_flood_column = f"%_length_30%_fr_{hazard_threshold}"
            
            if road_flood_column in joined.columns:
                road_flood_pct = joined[road_flood_column].fillna(0.0)
                road_accessible = road_flood_pct <= road_flood_threshold

            else:
                warnings.warn(f"Road flooding column {road_flood_column} not found, using fallback")
                road_accessible = pd.Series(True, index=assets_gdf.index)
            
            # Combine local flooding and road accessibility
            accessible = local_accessible & road_accessible.values
            
        except Exception as e:
            warnings.warn(f"Could not perform road accessibility analysis: {e}")
            # Fallback to local flooding only
            accessible = local_accessible
        
        return pd.Series(accessible, index=assets_gdf.index, name='accessible')
        
    except Exception as e:
        warnings.warn(f"Error in accessibility model: {e}")
        # Conservative fallback - assume not accessible if any significant local flooding
        n_assets = len(asset_geometries)
        # Handle both GeoSeries and GeoDataFrame
        if hasattr(asset_geometries, 'geometry'):
            geoms_to_extract = asset_geometries.geometry
        else:
            geoms_to_extract = asset_geometries
        hazard_values = extract_hazard_values_at_points(geoms_to_extract, hazard_map_path)
        accessible = hazard_values <= hazard_threshold  
        return pd.Series(accessible, name='accessible')

def extract_hazard_values_at_points(geometries, hazard_map_path):
    """
    Extract hazard values from raster at point locations.
    
    Parameters:
    -----------
    geometries : gpd.GeoSeries or gpd.GeoDataFrame
        Point geometries
    hazard_map_path : str or Path
        Path to hazard raster file
        
    Returns:
    --------
    np.array
        Hazard values at each point location
    """
    # Handle different input types
    if hasattr(geometries, 'geometry'):
        # It's a GeoDataFrame, extract the geometry column
        geom_series = geometries.geometry
    else:
        # It's already a GeoSeries
        geom_series = geometries
    
    points = []
    for geom in geom_series:
        if hasattr(geom, 'centroid'):
            points.append(geom.centroid)
        else:
            points.append(geom)
    
    with rasterio.open(hazard_map_path) as src:
        values = []
        for point in points:
            try:
                # Sample the raster at the point location
                sampled = list(src.sample([(point.x, point.y)]))
                value = sampled[0][0] if sampled and len(sampled[0]) > 0 else 0.0
                # Handle nodata values
                if value == src.nodata or np.isnan(value):
                    value = 0.0
                values.append(value)
            except Exception:
                values.append(0.0)
    
    return np.array(values)

def run_hazard_analysis(hazard_map_path, project_root=None):
    """
    Run complete hazard analysis for a specific hazard map.
    
    Parameters:
    -----------
    hazard_map_path : str or Path
        Path to hazard raster file
    project_root : Path or str, optional
        Root directory of the project
        
    Returns:
    --------
    gpd.GeoDataFrame
        Grid with road flooding analysis
    """
    paths = initialize_project_paths(project_root)
    
    # Ensure baseline analysis exists
    if _baseline_analysis_cache is None:
        create_baseline_grid_analysis(project_root)
    
    grid = _baseline_analysis_cache.copy()
    
    # Load study area
    extent_path = paths['network_path'] / "try_study_area_larger.shp"
    extent = gpd.read_file(extent_path)
    shapely_polygon = extent.geometry.iloc[0]
    
    # Run RA2CE analysis with the specific hazard file
    hazard_files = [Path(hazard_map_path)]
    
    result = complete_flood_road_analysis(
        grid_cells=grid,
        root_dir=paths['root_dir'],
        static_path=paths['static_path'],
        output_path=paths['output_path'],
        study_polygon=shapely_polygon,
        hazard_files=hazard_files,
        fr_threshold=0.3,
        ma_threshold=0.2
    )
    
    return result

def check_project_setup(project_root=None):
    """
    Check if the project is properly set up with all required files.
    
    Parameters:
    -----------
    project_root : Path or str, optional
        Root directory of the project
        
    Returns:
    --------
    dict
        Dictionary with setup status and missing files
    """
    try:
        paths = initialize_project_paths(project_root)
    except FileNotFoundError as e:
        return {'status': 'error', 'message': str(e), 'missing_files': ['data directory']}
    
    required_files = {
        'study_area': paths['network_path'] / "try_study_area_larger.shp",
        'road_network': paths['output_path'] / "base_graph.p",
        'static_path': paths['static_path'],
        'hazard_path': paths['hazard_path'],
        'output_path': paths['output_path']
    }
    
    missing_files = []
    existing_files = []
    
    for name, path in required_files.items():
        if path.exists():
            existing_files.append(name)
        else:
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        status = 'incomplete'
        message = f"Project setup incomplete. Found {len(existing_files)} of {len(required_files)} required files."
    else:
        status = 'complete'
        message = "Project setup complete. All required files found."
    
    return {
        'status': status,
        'message': message,
        'existing_files': existing_files,
        'missing_files': missing_files,
        'paths': paths
    }

def project_graph_to_meters(G, from_crs="EPSG:4326", to_crs="EPSG:28992"):
    """Project graph coordinates to meters and calculate edge lengths."""
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for node, data in G.nodes(data=True):
        x, y = data['x'], data['y']
        x_m, y_m = transformer.transform(x, y)
        data['x_m'] = x_m
        data['y_m'] = y_m

    for u, v, data in G.edges(data=True):
        x1, y1 = G.nodes[u]['x_m'], G.nodes[u]['y_m']
        x2, y2 = G.nodes[v]['x_m'], G.nodes[v]['y_m']
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        data['length'] = length

    return G

def prepare_node_gdf(G):
    """Convert graph nodes to GeoDataFrame for spatial operations."""
    nodes = []
    for node, data in G.nodes(data=True):
        nodes.append({
            "node": node,
            "geometry": Point(data["x_m"], data["y_m"])
        })
    return gpd.GeoDataFrame(nodes, crs="EPSG:28992")

def get_nearest_node(point_wgs84, node_gdf):
    """Find nearest graph node to a point."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    x, y = transformer.transform(point_wgs84.x, point_wgs84.y)
    point_rd = gpd.GeoSeries([Point(x, y)], crs="EPSG:28992")
    try:
        nearest = gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=point_rd),
            node_gdf,
            how="left",
            max_distance=1000
        )
        return nearest.iloc[0]["node"] if len(nearest) > 0 else None
    except Exception:
        return None