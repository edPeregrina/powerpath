"""
Grid-Based Accessibility Analysis Module

This module provides grid-based accessibility analysis for electrical infrastructure assets
using road network flooding analysis. It integrates with RA2CE for network analysis and 
provides functionality to determine asset accessibility during flood events.

Main Functions:
- accessibility_model(): Primary function for determining asset accessibility
- initialize_grid_analysis(): Setup baseline grid analysis (called automatically)

Usage:
    from grid_based_accessibility import accessibility_model
    
    # This is called automatically during first use
    accessible = accessibility_model(asset_geometries, hazard_map_path)

Requirements:
- RA2CE must be installed and configured
- Required graph files (base_graph.p, base_graph_hazard_editted.p) in output directory
- Study area shapefile in network directory
- Hazard raster files in hazard/processed directory
"""

# Nieuwe methode proberen
# Do your imports
# === Standard Library ===
from pathlib import Path
import pickle

# === Scientific & Data Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Geospatial Libraries ===
import geopandas as gpd
import rasterio
import folium
from shapely.geometry import box, Point, LineString, Polygon, shape
from pyproj import Transformer
import networkx as nx
# for hexagons:
import geohexgrid as ghg


# === RA2CE Project Imports ===
from ra2ce.network.network_config_data.enums.aggregate_wl_enum import AggregateWlEnum
from ra2ce.network.network_config_data.enums.source_enum import SourceEnum
from ra2ce.network.network_config_data.enums.network_type_enum import NetworkTypeEnum
from ra2ce.network.network_config_data.enums.road_type_enum import RoadTypeEnum
from ra2ce.network.network_config_data.network_config_data import (
    HazardSection,
    NetworkConfigData,
    NetworkSection,
    OriginsDestinationsSection
)
from ra2ce.network.exporters.geodataframe_network_exporter import GeoDataFrameNetworkExporter
from ra2ce.network.exporters.multi_graph_network_exporter import MultiGraphNetworkExporter
from ra2ce.network.network_wrappers.osm_network_wrapper.osm_network_wrapper import OsmNetworkWrapper
from ra2ce.ra2ce_handler import Ra2ceHandler

# Global variables to cache grid analysis
_baseline_grid = None
_baseline_graph = None
_baseline_node_gdf = None
_cached_hazard_analysis = {}
_verbose = False

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
        print("No project root specified, trying to find it automatically...")
        cwd = Path.cwd()
        # Try to find the project root by looking for 'data' directory
        if (cwd / 'data').exists():
            root_dir = cwd / 'data'
        elif (cwd.parent / 'data').exists():
            root_dir = cwd.parent / 'data'
        else:
            raise FileNotFoundError("Could not find 'data' directory. Please specify project_root parameter.")
    else:
        root_dir = Path(project_root)
        data_dir = root_dir / 'data'
        static_dir = data_dir / "static"
        print(f"Using project root: {root_dir}")
        if not root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {root_dir}")
    
    paths = {
        'root_dir': root_dir,
        'static_path': static_dir,
        'hazard_path': static_dir / "hazard",
        'network_path': static_dir / "network", 
        'output_path': static_dir / "output_graph",
        'output_directory': data_dir / "output"
    }
    
    return paths

def get_all_files(directory: str) -> list[Path]:
    p = Path(directory)
    return [file for file in p.iterdir() if file.is_file()]

def read_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_gpkg_to_gdf(file_path: str, layer: str = None) -> gpd.GeoDataFrame:
    # Read the geopackage file into a GeoDataFrame
    gdf = gpd.read_file(file_path, layer=layer)
    return gdf
def setup_ra2ce_analysis(project_root=None, hazard_file=None):
    """
    Setup and run RA2CE analysis if needed.
    This function handles the network and hazard analysis preparation.
    """
    # Initialize paths
    paths = initialize_project_paths(project_root)
    root_dir = paths['root_dir']
    static_path = paths['static_path']
    output_path = paths['output_path']
    network_path = paths['network_path']
    hazard_path = paths['hazard_path']
    
    # Get hazard files
    hazard_path_processed = hazard_path.joinpath("processed")
    hazard_files = get_all_files(hazard_path_processed)
    hazard_crs = "EPSG:4326"  # for the hackathon case
    
    if not hazard_files:
        raise FileNotFoundError("No hazard files found in processed directory")
    
    if _verbose:
        for hazard_file in hazard_files:
            print(hazard_file)
    
    # Check if analysis outputs already exist
    base_graph_path = output_path.joinpath("base_graph.p")
    hazard_graph_path = output_path.joinpath("base_graph_hazard_editted.p")
    
    if base_graph_path.exists() and hazard_graph_path.exists():
        if _verbose:
            print("RA2CE analysis outputs already exist, skipping...")
        return
    
    # Find the study area
    extent_path = network_path.joinpath("try_study_area_larger.shp")
    extent = gpd.read_file(extent_path, driver='ESRI Shapefile')
    shapely_polygon = extent.geometry.iloc[0]
    
    # RA2CE network section
    _network_section = NetworkSection(
        network_type=NetworkTypeEnum.DRIVE,
        source=SourceEnum.OSM_DOWNLOAD,
        polygon=extent_path,  # it needs a path without the list!
        save_gpkg=True,
        road_types=[RoadTypeEnum.PRIMARY, RoadTypeEnum.PRIMARY_LINK, RoadTypeEnum.TRUNK, 
                   RoadTypeEnum.SECONDARY, RoadTypeEnum.SECONDARY_LINK, RoadTypeEnum.TERTIARY, 
                   RoadTypeEnum.RESIDENTIAL], 
        attributes_to_exclude_in_simplification=['bridge', 'tunnel'],
    )

    # Use the first hazard file for the analysis
    target_hazard_file = hazard_file if hazard_file else hazard_files[0]
    
    # Make the NetworkConfigData
    _hazard_section = HazardSection(
        hazard_map=[target_hazard_file],
        hazard_id=None,
        hazard_field_name="waterdepth",
        aggregate_wl=AggregateWlEnum.MAX,
        hazard_crs=hazard_crs,
        overlay_segmented_network=False
    )

    _network_config_data = NetworkConfigData(
        root_path=root_dir,
        static_path=static_path,
        output_path=output_path,
        network=_network_section,
        hazard=_hazard_section
    )

    # Run analysis
    if _verbose:
        print("Running RA2CE analysis...")
    _handler = Ra2ceHandler.from_config(_network_config_data, analysis=None)
    _handler.configure()
    _handler.run_analysis()
    if _verbose:
        print("RA2CE analysis complete")

# Refactored modular script
def load_graph(path: Path) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)

def project_graph_coords(G: nx.Graph, from_crs: str, to_crs: str) -> nx.Graph:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for n, d in G.nodes(data=True):
        d["x_m"], d["y_m"] = transformer.transform(d["x"], d["y"])
    for u, v, d in G.edges(data=True):
        x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
        x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
        d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return G

def filter_motorway_edges(G: nx.Graph) -> nx.Graph:
    G_filtered = G.copy()
    edges_to_remove = [
        (u, v) for u, v, d in G_filtered.edges(data=True)
        if "highway" in d and isinstance(d["highway"], str) and "motorway" in d["highway"].lower()
    ]
    G_filtered.remove_edges_from(edges_to_remove)
    G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))
    return G_filtered

def create_grid(study_area: gpd.GeoDataFrame, cell_size: int, target_crs: str) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = study_area.total_bounds
    # Create hexagonal grid
    grid = ghg.make_grid_from_bounds(
        minx, miny, maxx, maxy,
        R=cell_size,
        crs=study_area.crs
    )

    # Calculate centroids directly in the same CRS as the grid
    grid["centroid"] = grid.geometry.centroid

    if not grid.empty and grid.geometry.is_valid.all():
        if _verbose:
            print(f"Grid created with {len(grid)} cells")
    else:
        print("Grid is empty or contains invalid geometries. Skipping plot.")

    return grid

def build_node_gdf(G: nx.Graph, crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame([{"node": n, "geometry": Point(d["x_m"], d["y_m"])} for n, d in G.nodes(data=True)], crs=crs)

def get_nearest_node(point: Point, node_gdf: gpd.GeoDataFrame, transformer: Transformer, max_distance: int) -> int:
    if transformer is not None:
        x, y = transformer.transform(point.x, point.y)
        point_rd = gpd.GeoSeries([Point(x, y)], crs=node_gdf.crs)
    else:
        point_rd = gpd.GeoSeries([point], crs=node_gdf.crs)
    try:
        nearest = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=point_rd), node_gdf, how="left", max_distance=max_distance)
        return nearest.iloc[0]["node"]
    except:
        return None

def assign_nearest_nodes(grid: gpd.GeoDataFrame, node_gdf: gpd.GeoDataFrame, transformer: Transformer, max_distance: int) -> gpd.GeoDataFrame:
    grid["nearest_node"] = grid["centroid"].apply(lambda pt: get_nearest_node(pt, node_gdf, transformer, max_distance))
    return grid

def compute_grid_distances(grid: gpd.GeoDataFrame, G: nx.Graph, node_col: str, label_prefix: str) -> gpd.GeoDataFrame:
    avg_distances, reachable_counts = [], []
    for idx, row in tqdm(grid.iterrows(), total=len(grid)):
        center_node = row[node_col]
        if center_node is None:
            print(f"Warning: No nearest node for grid cell {idx}, skipping...")
            avg_distances.append(np.nan)
            reachable_counts.append(0)
            continue
        neighbors = grid[grid.geometry.touches(row.geometry) | (grid.index == idx)]
        dists, reachable = [], 0
        for _, n_row in neighbors.iterrows():
            if n_row["centroid"].equals(row["centroid"]): continue
            neighbor_node = n_row[node_col]
            if neighbor_node is None: continue
            try:
                length = nx.shortest_path_length(G, source=neighbor_node, target=center_node, weight="length")
                dists.append(length)
                reachable += 1
            except:
                continue
        avg_distances.append(np.mean(dists) if dists else np.nan)
        reachable_counts.append(reachable)
    grid[f"avg_distance_{label_prefix}"] = avg_distances
    grid[f"reachable_cells_{label_prefix}"] = reachable_counts
    return grid

def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str) -> nx.Graph:
    def is_motorway(highway):
        if isinstance(highway, str):
            return "motorway" in highway.lower()
        elif isinstance(highway, list):
            return any("motorway" in str(h).lower() for h in highway)
        return False

    def is_protected(d):
        # Keep edge if it has a bridge or tunnel value
        return pd.notna(d.get("bridge")) or pd.notna(d.get("tunnel"))

    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get(hazard_column, 0) > threshold and not is_motorway(d.get("highway")) and not is_protected(d)
    ]

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def sample_flood_depths(grid: gpd.GeoDataFrame, raster_path: Path, threshold: float) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        # Convert centroids to raster CRS for sampling
        grid_centroids = grid.set_geometry("centroid")
        grid_hazard = grid_centroids.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in grid_hazard.geometry]
        sampled_values = list(src.sample(coords))
    grid["flood_depth"] = [val[0] if val[0] is not None else np.nan for val in sampled_values]
    grid["flooded"] = grid["flood_depth"] > threshold
    return grid

def compute_hazard_distances(grid: gpd.GeoDataFrame, G: nx.Graph, node_gdf: gpd.GeoDataFrame, transformer: Transformer, max_distance: int, label_prefix: str) -> gpd.GeoDataFrame:
    def get_node(pt): return get_nearest_node(pt, node_gdf, transformer, max_distance)
    distances, reachable_counts = [], []
    for idx, row in tqdm(grid.iterrows(), total=len(grid)):
        center = row["centroid"]
        neighbors = grid[grid.geometry.touches(row.geometry) | (grid.index == idx)]
        try:
            center_node = get_node(center)
            if center_node is None: raise ValueError("No center node")
            dists, reachable = [], 0
            for _, n_row in neighbors.iterrows():
                if n_row["centroid"].equals(center): continue
                neighbor_node = get_node(n_row["centroid"])
                if neighbor_node is None: continue
                try:
                    length = nx.shortest_path_length(G, source=neighbor_node, target=center_node, weight="length")
                    dists.append(length)
                    reachable += 1
                except:
                    continue
            distances.append(np.mean(dists) if dists else np.nan)
            reachable_counts.append(reachable)
        except:
            distances.append(np.nan)
            reachable_counts.append(0)
            
    # Add hazard-aware metrics
    grid[f"avg_distance_{label_prefix}"] = distances
    grid[f"reachable_cells_{label_prefix}"] = reachable_counts

    # Compute differences from baseline
    grid["distance_diff"] = grid[f"avg_distance_{label_prefix}"] - grid["avg_distance_no_flood"]
    grid["reachable_diff"] = grid["reachable_cells_no_flood"] - grid[f"reachable_cells_{label_prefix}"]

    return grid


def initialize_grid_analysis(project_root=None):
    """
    Initialize the baseline grid analysis that can be reused for multiple accessibility calls.
    This should be called once before using the accessibility_model function.
    """
    global _baseline_grid, _baseline_graph, _baseline_node_gdf
    
    if _baseline_grid is not None:
        return  # Already initialized
    
    # Initialize paths
    paths = initialize_project_paths(project_root)
    
    # === PARAMETERS ===
    cell_size = 250
    from_crs = "EPSG:4326"
    to_crs = "EPSG:28992"
    max_distance = 250

    # === PATHS ===
    road_network_path = paths['output_path'].joinpath("base_graph.p")
    
    # Check if the required files exist
    if not road_network_path.exists():
        raise FileNotFoundError(f"Required graph file not found: {road_network_path}")
    
    # === LOAD STUDY AREA ===
    extent_path = paths['network_path'].joinpath("try_study_area_larger.shp")
    extent = gpd.read_file(extent_path, driver='ESRI Shapefile')
    study_area_rd = extent.to_crs(to_crs)

    # === BASE GRAPH ===
    G = project_graph_coords(load_graph(road_network_path), from_crs, to_crs)
    # --- FILTER OUT MOTORWAY EDGES ---
    G_filtered = filter_motorway_edges(G)
    node_gdf = build_node_gdf(G_filtered, crs=to_crs) #before node_gdf_filtered
    # node_gdf = build_node_gdf(G, crs=to_crs)

    # === GRID ===
    grid = create_grid(study_area_rd, cell_size, target_crs=to_crs)  # Create grid in projected CRS
    transformer = None  # No transformer needed since grid and nodes are in same CRS
    grid = assign_nearest_nodes(grid, node_gdf, transformer, max_distance)
    grid = compute_grid_distances(grid, G, node_col="nearest_node", label_prefix="no_flood")

    # Store in global variables
    _baseline_grid = grid
    _baseline_graph = G
    _baseline_node_gdf = node_gdf
    
    if _verbose:
        print("Grid analysis initialized successfully")

def compute_hazard_graph_from_map(hazard_map_path, base_graph, project_root=None, day_string='01'):
    """
    PLACEHOLDER: Compute hazard-aware graph from hazard map and base graph.
    
    This function should overlay the hazard map onto the base graph and create
    a new graph with hazard attributes (like water depth) on edges/nodes.
    
    Parameters:
    -----------
    hazard_map_path : str or Path
        Path to the hazard raster file
    base_graph : nx.Graph
        The base road network graph
    project_root : Path or str, optional
        Root directory of the project
        
    Returns:
    --------
    nx.Graph
        Graph with hazard attributes added
    """
    # PLACEHOLDER - This should implement the hazard overlay logic
    # For now, return the base graph (this will need to be implemented)
    print("PLACEHOLDER: compute_hazard_graph_from_map - Using base graph as fallback")
    return base_graph.copy()

def load_or_compute_hazard_graph(hazard_map_path, project_root=None):
    """
    Load existing hazard graph or compute it from the hazard map.
    
    Parameters:
    -----------
    hazard_map_path : str or Path
        Path to the hazard raster file
    project_root : Path or str, optional
        Root directory of the project
        
    Returns:
    --------
    nx.Graph
        Hazard-aware road network graph
    """
    global _baseline_graph
    
    # Initialize baseline if not done
    if _baseline_graph is None:
        initialize_grid_analysis(project_root)
    
    # Initialize paths
    paths = initialize_project_paths(project_root)
    
    # Extract day string from hazard map path for naming
    hazard_file = Path(hazard_map_path)
    day_string = hazard_file.stem  # Use filename without extension as day identifier
    
    # Try to load existing hazard graph for this specific day/hazard
    hazard_graph_path = paths['output_path'].joinpath(f"base_graph_hazard_editted.p")
    
    if hazard_graph_path.exists():
        if _verbose:
            print(f"Loading existing hazard graph: {hazard_graph_path}")
        return load_graph(hazard_graph_path)
    else:
        print(f"Hazard graph not found for {day_string}, computing from hazard map...")
        #TODO: PLACEHOLDER: Compute hazard graph from the hazard map; currently if not available, save a copy of the baseline graph
        hazard_graph = compute_hazard_graph_from_map(hazard_map_path, _baseline_graph, project_root, day_string=day_string)
        
        # Save the computed graph for future use
        with open(hazard_graph_path, 'wb') as f:
            pickle.dump(hazard_graph, f)
            print(f"Saved computed hazard graph: {hazard_graph_path}")
        
        return hazard_graph

def run_hazard_grid_analysis(hazard_map_path, threshold=0.2, project_root=None, output_path=None, day_string='01'):
    """
    Run hazard-specific grid analysis for a given hazard map.
    Returns a grid with hazard-aware accessibility metrics.
    """
    global _baseline_grid, _baseline_graph, _baseline_node_gdf, _cached_hazard_analysis
    road_network_path = output_path.joinpath("base_graph.p") 
    hazard_graph_path = output_path.joinpath("base_graph_hazard_editted.p")
    print(f"Running hazard grid analysis for {hazard_map_path} with threshold {threshold} m")
    # Check if already cached
    cache_key = f"{hazard_map_path}_{threshold}"
    if cache_key in _cached_hazard_analysis:
        if _verbose:
            print(f"Using cached hazard analysis for {cache_key}")
        return _cached_hazard_analysis[cache_key]
    
    # Initialize baseline if not done
    if _baseline_grid is None:
        if _verbose:
            print("Initializing baseline grid analysis...")
        initialize_grid_analysis(project_root)
    
    # === PARAMETERS ===
    from_crs = "EPSG:4326"
    to_crs = "EPSG:28992"
    max_distance = 250

    # Copy baseline grid
    grid = _baseline_grid.copy()
    
    # === HAZARD GRAPH ===
    # Load or compute hazard graph specific to this hazard map
    column_name = 'EV'+ str(int(day_string)+1) + '_ma'
    # G_hazard_raw = load_or_compute_hazard_graph(hazard_map_path, project_root)
    G_hazard = project_graph_coords(load_graph(hazard_graph_path), from_crs, to_crs)
    G_hazard = filter_hazard_graph(G_hazard, threshold, hazard_column=column_name)
    node_gdf_hazard = build_node_gdf(G_hazard, crs=to_crs)


    # === FLOOD DEPTHS ===
    grid = sample_flood_depths(grid, Path(hazard_map_path), threshold)

    # === HAZARD DISTANCES ===
    transformer = None  # No transformer needed since grid and nodes are in same CRS (to_crs)
    grid = compute_hazard_distances(
        grid,
        G_hazard,
        node_gdf_hazard,
        transformer,
        max_distance=max_distance,
        label_prefix="hazard"
    )
    
    # Cache the result
    _cached_hazard_analysis[cache_key] = grid
    
    return grid

def accessibility_model(asset_geometries, hazard_map_path, hazard_values=None, hazard_threshold=0.2, project_root=None, day_string='01', verbose=None):
    """
    Determine accessibility of assets based on grid-based road network analysis.
    
    Parameters:
    -----------
    asset_geometries : gpd.GeoSeries or gpd.GeoDataFrame
        Geometries of the assets to check accessibility for
    hazard_map_path : str or Path
        Path to the hazard raster file (.tif)
    hazard_values : array-like, optional
        Pre-computed hazard values at asset locations (currently not used in grid-based analysis)
    hazard_threshold : float, default 0.2
        Water depth threshold in meters for flooding analysis
    project_root : Path or str, optional
        Root directory of the project
    verbose : bool, optional
        Whether to print detailed debug information. If None, uses global _verbose setting
        
    Returns:
    --------
    pd.Series
        Boolean series indicating accessibility for each asset (True = accessible, False = not accessible)
    """
    # Only set verbose if explicitly passed, otherwise use current global setting
    if verbose is not None:
        set_verbose(verbose)
    
    try:
        # Initialize paths to ensure project_root is properly set
        paths = initialize_project_paths(project_root)
        actual_project_root = paths['root_dir']
        output_path = paths['output_path']
        
        # Run hazard analysis for this specific hazard map
        print(f"Running hazard grid analysis for {hazard_map_path} with threshold {hazard_threshold} m")
        grid = run_hazard_grid_analysis(hazard_map_path, hazard_threshold, actual_project_root, output_path=output_path, day_string=day_string)
        if _verbose:
            print(f"Grid has {len(grid)} cells")
            print(f"Baseline reachable cells: {grid['reachable_cells_no_flood'].mean():.2f} (avg)")
            print(f"Hazard reachable cells: {grid['reachable_cells_hazard'].mean():.2f} (avg)")
        
        # Convert asset geometries to GeoDataFrame if needed
        if hasattr(asset_geometries, 'geometry'):
            # It's already a GeoDataFrame
            assets_gdf = asset_geometries.copy()
        else:
            # It's a GeoSeries, convert to GeoDataFrame
            assets_gdf = gpd.GeoDataFrame(geometry=asset_geometries.copy())
        
        # Ensure CRS matches grid CRS
        if assets_gdf.crs != grid.crs:
            assets_gdf = assets_gdf.to_crs(grid.crs)
        

        accessible_list = []
        for idx, asset_geom in enumerate(assets_gdf.geometry):
            try:
                # Create point from geometry
                if hasattr(asset_geom, 'centroid'):
                    asset_point = asset_geom.centroid
                else:
                    asset_point = asset_geom
                
                # Find the grid cell that contains the asset
                grid_cell = grid[grid.geometry.contains(asset_point)]
                # print(f"Asset {idx} - Found {len(grid_cell)} grid cells containing the asset")
                
                if len(grid_cell) == 0:
                    # Asset not in any grid cell, check nearest
                    distances = grid.geometry.distance(asset_point)
                    nearest_idx = distances.idxmin()
                    grid_cell = grid.iloc[[nearest_idx]]
                
                # Check accessibility based on reachable cells
                if len(grid_cell) > 0:
                    # print(grid_cell)
                    reachable_cells = grid_cell['reachable_cells_hazard'].iloc[0]
                    accessible = reachable_cells > 0
                else:
                    # Conservative fallback
                    accessible = False
                    
                accessible_list.append(accessible)
                
            except Exception as e:
                if _verbose:
                    print(f"Warning: Error processing asset {idx}: {e}")
                # Conservative fallback
                accessible_list.append(False)
        
        return pd.Series(accessible_list, name='accessible')
        
    except Exception as e:
        if _verbose:
            print(f"Error in grid-based accessibility model: {e}")
        # Conservative fallback - assume not accessible
        n_assets = len(asset_geometries)
        return pd.Series([False] * n_assets, name='accessible')

def test_accessibility_analysis(project_root=None):
    """
    Test function to demonstrate the accessibility analysis.
    Returns a simple test case using the electrical stations.
    """
    try:
        # Initialize paths
        paths = initialize_project_paths(project_root)
        
        # Load some test assets (electrical stations)
        electricity_path = paths['root_dir'] / 'electricity'
        
        if (electricity_path / 'ls_stations_clipped.shp').exists():
            stations = gpd.read_file(electricity_path / 'ls_stations_clipped.shp')
            stations = stations.to_crs("EPSG:4326")  # Ensure WGS84
            
            # Get a hazard file
            hazard_path_processed = paths['hazard_path'] / 'processed'
            hazard_files = get_all_files(hazard_path_processed)
            
            if hazard_files:
                hazard_file = hazard_files[0]
                if _verbose:
                    print(f"Testing with {len(stations)} stations and hazard file: {hazard_file}")
                
                # Run accessibility analysis
                accessible = accessibility_model(stations.geometry, hazard_file, project_root=project_root)
                
                if _verbose:
                    print(f"Results: {accessible.sum()} out of {len(accessible)} stations are accessible")
                return accessible
            else:
                if _verbose:
                    print("No hazard files found for testing")
                return None
        else:
            if _verbose:
                print("No station files found for testing")
            return None
            
    except Exception as e:
        if _verbose:
            print(f"Test failed: {e}")
        return None

def set_verbose(verbose: bool = True):
    """
    Set the global verbose flag for controlling print statements.
    
    Parameters:
    -----------
    verbose : bool, default True
        Whether to enable verbose output
    """
    global _verbose
    _verbose = verbose

def compute_island_geodataframe_from_graph(graph_pickle_path: str, hazard_threshold: float, hazard_column: str) -> gpd.GeoDataFrame:
    with open(graph_pickle_path, "rb") as f:
        G = pickle.load(f)
        G = nx.DiGraph(G)

    G = project_graph_coords(G, from_crs="EPSG:4326", to_crs="EPSG:28992")
    G = filter_hazard_graph(G, hazard_threshold, hazard_column)

    # Identify strongly connected components
    components = list(nx.strongly_connected_components(G))
    fid_to_island = {}

    # Assign island_id to each fid
    for i, comp in enumerate(components):
        subgraph = G.subgraph(comp)
        for u, v, data in subgraph.edges(data=True):
            fid_to_island[v] = i

    # Create transformer for projecting geometries
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    
    # Build edge records with geometry and length
    records = []

    for u, v, data in G.edges(data=True):
        # Project the actual edge geometry
        original_geom = data['geometry']
        if hasattr(original_geom, 'coords'):
            # Project all coordinates in the geometry
            projected_coords = [transformer.transform(x, y) for x, y in original_geom.coords]
            projected_geom = LineString(projected_coords)
        else:
            # Fallback: create LineString from node positions if no edge geometry
            projected_geom = LineString([(G.nodes[u]["x_m"], G.nodes[u]["y_m"]),
                                       (G.nodes[v]["x_m"], G.nodes[v]["y_m"])])
        
        length_m = data.get("length", None)
        if length_m is None:
            print(f"Length not found for edge ({u}, {v}), calculating from projected geometry.")
            length_m = projected_geom.length  

        island_id = fid_to_island.get(v, -1)

        record = data.copy()
        record["geometry"] = projected_geom  # Use properly projected geometry
        record["length_m"] = length_m
        record["island_id"] = island_id
        records.append(record)

    # Create GeoDataFrame with correct CRS
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")

    # Compute island sizes using groupby
    island_sizes = gdf.groupby("island_id")["length_m"].sum().reset_index()
    island_sizes["island_size_km"] = island_sizes["length_m"] / 1000.0
    island_sizes = island_sizes[["island_id", "island_size_km"]]

    # Merge island sizes back into GeoDataFrame
    gdf = gdf.merge(island_sizes, on="island_id", how="left")
    gdf["island_size_km"] = gdf["island_size_km"].fillna(0.0)

    return gdf

# Export main functions for easy import
__all__ = [
    'accessibility_model',
    'initialize_grid_analysis', 
    'load_or_compute_hazard_graph',
    'compute_hazard_graph_from_map',
    'test_accessibility_analysis',
    'set_verbose'
]
