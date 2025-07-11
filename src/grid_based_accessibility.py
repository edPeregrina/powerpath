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

# specify the name of the path to the project folder where you created the RA2CE folder setup

# root_dir = Path(r'C:\python\powerpath\data')
#TODO: Set input path to data path
root_dir = Path( r'C:/repos/powerpath/data')
assert root_dir.exists()
static_path = root_dir.joinpath("static")
hazard_path = static_path.joinpath("hazard")
network_path = static_path.joinpath("network")
output_path = static_path.joinpath("output_graph")

## Find the study area
# <font color='blue'>To do in a later stage: make this flexible based on hazard map</font> 
Extent_path = network_path.joinpath("try_study_area_larger.shp") #TODO Make flexible based on hazard map
Extent = gpd.read_file(Extent_path, driver='ESRI Shapefile')
shapely_polygon = Extent.geometry.iloc[0]
# Data pre-processing
# some preliminary functions

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
hazard_path_processed = hazard_path.joinpath("processed") #TODO: get from the calling function
hazard_files = get_all_files(hazard_path_processed)
hazard_crs = "EPSG:4326" # for the hackathon case => "EPSG:4326" 

for hazard_file in hazard_files:
    print (hazard_file)
# RA2CE
### Let us first initalize and perform the ra2ce run so we have all the data that we need
#### Cutting RoadTypeEnum.MOTORWAY,RoadTypeEnum.MOTORWAY_LINK to make analysis more realistic
_network_section = NetworkSection(
    network_type=NetworkTypeEnum.DRIVE,
    source=SourceEnum.OSM_DOWNLOAD,
    polygon=Extent_path, #it needs a path without the list!
    save_gpkg=True,
    road_types=[RoadTypeEnum.PRIMARY, RoadTypeEnum.PRIMARY_LINK,RoadTypeEnum.TRUNK, RoadTypeEnum.SECONDARY,RoadTypeEnum.SECONDARY_LINK, RoadTypeEnum.TERTIARY, RoadTypeEnum.RESIDENTIAL], 
    attributes_to_exclude_in_simplification=['bridge', 'tunnel'],
)

for hazard_file in hazard_files:
    # Make the NetworkConfigData
    _hazard_section = HazardSection(
        hazard_map=[hazard_file],
        hazard_id=None,
        hazard_field_name="waterdepth",
        aggregate_wl=AggregateWlEnum.MAX,
        hazard_crs=hazard_crs,
        overlay_segmented_network = False
    )

_network_config_data = NetworkConfigData(
    root_path=root_dir,
    static_path=static_path,
    output_path=output_path,
    network=_network_section,
    hazard=_hazard_section)

# Run analysis
_handler = Ra2ceHandler.from_config(_network_config_data, analysis=None)
_handler.configure()
_handler.run_analysis()
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

def create_grid(study_area: gpd.GeoDataFrame, cell_size: int, target_crs: str) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = study_area.total_bounds
    cols, rows = int((maxx - minx) / cell_size), int((maxy - miny) / cell_size)
    cells = [box(minx + i * cell_size, miny + j * cell_size, minx + (i + 1) * cell_size, miny + (j + 1) * cell_size)
             for i in range(cols) for j in range(rows)
             if study_area.geometry.unary_union.intersects(box(minx + i * cell_size, miny + j * cell_size,
                                                               minx + (i + 1) * cell_size, miny + (j + 1) * cell_size))]
    grid = gpd.GeoDataFrame(geometry=cells, crs=study_area.crs).to_crs(target_crs)
    grid["centroid"] = grid.geometry.centroid
    return grid

def build_node_gdf(G: nx.Graph, crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame([{"node": n, "geometry": Point(d["x_m"], d["y_m"])} for n, d in G.nodes(data=True)], crs=crs)

def get_nearest_node(point: Point, node_gdf: gpd.GeoDataFrame, transformer: Transformer, max_distance: int) -> int:
    x, y = transformer.transform(point.x, point.y)
    point_rd = gpd.GeoSeries([Point(x, y)], crs=node_gdf.crs)
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

def filter_hazard_graph(G: nx.Graph, threshold: float) -> nx.Graph:
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d.get("EV1_ma", 0) > threshold])
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def sample_flood_depths(grid: gpd.GeoDataFrame, raster_path: Path, threshold: float) -> gpd.GeoDataFrame:
    with rasterio.open(raster_path) as src:
        grid_hazard = grid.set_geometry("centroid").to_crs(src.crs)
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

# === PARAMETERS ===
cell_size = 500
threshold = 0.2
from_crs = "EPSG:4326"
to_crs = "EPSG:28992"
max_distance = 250

# === PATHS ===
road_network_path = output_path.joinpath("base_graph.p") 
hazard_graph_path= output_path.joinpath("base_graph_hazard.p")
flood_tiff_path = hazard_file


# === LOAD STUDY AREA ===
study_area_rd = Extent.to_crs(to_crs)

# === BASE GRAPH ===
G = project_graph_coords(load_graph(road_network_path), from_crs, to_crs)
node_gdf = build_node_gdf(G, crs=to_crs)

# === GRID ===
grid = create_grid(study_area_rd, cell_size, target_crs=from_crs)
transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
grid = assign_nearest_nodes(grid, node_gdf, transformer, max_distance)
grid = compute_grid_distances(grid, G, node_col="nearest_node", label_prefix="no_flood")

# === HAZARD GRAPH ===
G_hazard = project_graph_coords(load_graph(hazard_graph_path), from_crs, to_crs)
G_hazard = filter_hazard_graph(G_hazard, threshold)
node_gdf_hazard = build_node_gdf(G_hazard, crs=to_crs)

# === FLOOD DEPTHS ===
grid = sample_flood_depths(grid, flood_tiff_path, threshold)

# === HAZARD DISTANCES ===
grid = compute_hazard_distances(
    grid,
    G_hazard,
    node_gdf_hazard,
    transformer,
    max_distance=max_distance,
    label_prefix="hazard"
)

# grid.explore(column='reachable_diff',
#              cmap='Reds')
# for each asset in the calling function, check in which grid cell it falls and output accessible as a boolean (True or false)
def accessibility_model(asset_geometries, hazard_map_path):
    for idx, asset in enumerate(asset_geometries):
        # Find the grid cell that contains the asset
        asset_point = Point(asset.x, asset.y)
        grid_cell = grid[grid.geometry.contains(asset_point)]
        
        if grid_cell.reachable_cells_hazard == 0:
            accessible = False
        else:
            accessible = True
        
        # Store as pandas series of True/False
        if idx == 0:
            accessible_series = pd.Series([accessible], name='accessible')
        else:
            accessible_series = accessible_series.append(pd.Series([accessible]), ignore_index=True)
    return accessible_series