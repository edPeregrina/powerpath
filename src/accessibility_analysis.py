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

cwd = Path.cwd()
print(f"Current working directory: {cwd}")
# specify the name of the path to the project folder where you created the RA2CE folder setup

# root_dir = Path(r'C:\python\powerpath\data')
root_dir = cwd.parent / 'data'
assert root_dir.exists()
static_path = root_dir.joinpath("static")
hazard_path = static_path.joinpath("hazard")
network_path = static_path.joinpath("network")
output_path = static_path.joinpath("output_graph")



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
    
    # Ensure both GeoDataFrames use the same CRS
    if grid_result.crs != road_network.crs:
        road_network = road_network.to_crs(grid_result.crs)

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


# === PARAMETERS ===
road_network_path = output_path.joinpath("base_graph.p")  # <-- Pad naar je pickle-bestand
cell_size = 500  # Rasterresolutie in meters
# === DEBUG SETTINGS ===
MAX_DEBUG_PRINTS_NEAREST_NODE = 5
MAX_DEBUG_PRINTS_DISTANCE = 5
debug_count_nearest_node = 0
debug_count_distance = 0

Extent_path = network_path.joinpath("try_study_area_larger.shp")
Extent = gpd.read_file(Extent_path, driver='ESRI Shapefile')
shapely_polygon = Extent.geometry.iloc[0]


# === 1. Laad shapefile van studiegebied en projecteer tijdelijk naar RD (EPSG:28992) ===
study_area_rd = Extent.to_crs("EPSG:28992")

# === 2. Genereer raster binnen studiegebied ===
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

# === 3. Projecteer raster naar WGS84 ===
grid = grid.to_crs("EPSG:4326")

#Herbereken centroid in WGS84
grid["centroid"] = grid.geometry.centroid

# === 4. Laad wegennetwerk ===
with open(road_network_path, "rb") as f:
    G = pickle.load(f)

# === 5. Projecteer netwerk naar RD en bereken lengtes ===
def project_graph_to_meters(G, from_crs="EPSG:4326", to_crs="EPSG:28992"):
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

G = project_graph_to_meters(G)

# === 5b. Zet knopen om naar GeoDataFrame voor ruimtelijke zoekactie ===
def prepare_node_gdf(G):
    nodes = []
    for node, data in G.nodes(data=True):
        nodes.append({
            "node": node,
            "geometry": Point(data["x_m"], data["y_m"])
        })
    return gpd.GeoDataFrame(nodes, crs="EPSG:28992")

node_gdf = prepare_node_gdf(G)

# === 6. Functie om dichtstbijzijnde knoop te vinden via spatial join ===
def get_nearest_node(point_wgs84):
    global debug_count_nearest_node
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    x, y = transformer.transform(point_wgs84.x, point_wgs84.y)
    point_rd = gpd.GeoSeries([Point(x, y)], crs="EPSG:28992")
    try:
        nearest = gpd.sjoin_nearest(
            gpd.GeoDataFrame(geometry=point_rd),
            node_gdf,
            how="left",
            max_distance=100
        )
        node_id = nearest.iloc[0]["node"]
        if debug_count_nearest_node < MAX_DEBUG_PRINTS_NEAREST_NODE:
            print(f"Centroid: ({x:.1f}, {y:.1f}) → Nearest node: {node_id}")
            debug_count_nearest_node += 1
        return node_id
    except Exception as e:
        #print(f"Geen knoop gevonden voor centroid ({x:.1f}, {y:.1f}): {e}")
        return None



# === 7. Bereken gemiddelde afstand vanuit omliggende cellen ===
distances = []
for idx, row in tqdm(grid.iterrows(), total=len(grid)):
    center = row["centroid"]
    neighbors = grid[grid.geometry.touches(row.geometry) | (grid.index == idx)]

    try:
        center_node = get_nearest_node(center)
        if center_node is None:
            raise ValueError("Geen center node gevonden")

        dists = []
        for _, n_row in neighbors.iterrows():
            if n_row["centroid"].equals(center):
                continue
            try:
                neighbor_node = get_nearest_node(n_row["centroid"])
                if neighbor_node is None:
                    continue
                length = nx.shortest_path_length(G, source=neighbor_node, target=center_node, weight="length")

                if debug_count_distance < MAX_DEBUG_PRINTS_DISTANCE:
                    print(f"Center node: {center_node}, Neighbor node: {neighbor_node}, Distance: {length}")
                    debug_count_distance += 1

                dists.append(length)
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                #print(f"Geen pad tussen {neighbor_node} en {center_node}: {e}")
                continue

        avg_dist = np.mean(dists) if dists else np.nan
    except Exception as e:
        print(f"Fout bij cel {idx}: {e}")
        avg_dist = np.nan

    distances.append(avg_dist)


grid["avg_distance_no_flood"] = distances


# Example usage (with variables from your notebook):
if 'grid' in locals() and 'root_dir' in locals():
    # Get processed hazard files
    hazard_path_processed = static_path.joinpath("hazard", "processed")
    hazard_files_processed = [f for f in hazard_path_processed.iterdir() if f.suffix == '.tif']
    
    # Run complete analysis
    grid_with_road_analysis = complete_flood_road_analysis(
        grid_cells=grid.copy(),
        root_dir=root_dir,
        static_path=static_path,
        output_path=output_path,
        study_polygon=shapely_polygon,
        hazard_files=hazard_files_processed,
        fr_threshold=0.3,
        ma_threshold=0.2
    )
    
    print(f"Columns in result: {list(grid_with_road_analysis.columns)}")
# make all 0 values null and explore
grid_with_road_analysis.replace(0, np.nan, inplace=True)
grid_with_road_analysis.explore(column="%_length_30%_fr_0.2", tiles='CartoDBPositron')