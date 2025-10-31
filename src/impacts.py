import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from pathlib import Path

def load_voll_data(voll_path=None):
    """
    Load Value of Lost Load (VOLL) data from CSV and create mapping dictionaries.

    Args:
        voll_path: Path to CSV file with VOLL data. If None, uses default path.

    Returns:
        tuple: (bg_to_group_map, consumption_per_sqm, voll_per_sqm, lu_voll_data, lu_cat_dict, lu_consumpt_dict, lu_voll_dict)
    """
    if voll_path is None:
        voll_path = Path("C:/repos/powerpath/data/land_use/voll_lu.csv")
    
    # Load VOLL data from CSV
    lu_voll_data = pd.read_csv(voll_path, sep=None, engine='python')
    
    # Fix column names
    lu_voll_data.columns = [
        'Grouping', 
        'Consumption on country level (MWh/h)', 
        'VOLL €/MWh', 
        'BG2017 categories'
    ]
    
    # Create mapping dictionaries
    lu_cat_dict = {
        row['Grouping']: [int(x) for x in row['BG2017 categories'].strip().split(',')]
        for _, row in lu_voll_data.iterrows()
    }
    
    lu_consumpt_dict = {
        row['Grouping']: row['Consumption on country level (MWh/h)']
        for _, row in lu_voll_data.iterrows()
    }
    
    lu_voll_dict = {
        row['Grouping']: row['VOLL €/MWh']
        for _, row in lu_voll_data.iterrows()
    }
    
    # Create reverse mapping: BG code -> grouping
    bg_to_group_map = {}
    for group, codes in lu_cat_dict.items():
        for code in codes:
            bg_to_group_map[code] = group
    
    return bg_to_group_map, {}, {}, lu_voll_data, lu_cat_dict, lu_consumpt_dict, lu_voll_dict

# Load VOLL data at module level to be available to all functions
BG_TO_GROUP_MAP, CONSUMPTION_PER_SQM, VOLL_PER_SQM, VOLL_DATA, LU_CAT_DICT, LU_CONSUMPT_DICT, LU_VOLL_DICT = load_voll_data()

def create_voronoi_for_asset_type(gdf_assets, asset_type, boundary=None):
    """
    Create Voronoi polygons for a specific asset type and clip to boundary.
    
    Args:
        gdf_assets (GeoDataFrame): GeoDataFrame with assets
        asset_type (str): Type of asset to filter for
        boundary (Optional[Union[GeoSeries, Polygon]]): Optional boundary to clip Voronoi polygons to
        
    Returns:
        GeoDataFrame: Voronoi polygons with asset_id column
    """
    # Filter assets by type and get centroids
    assets_filtered = gdf_assets[gdf_assets['type'] == asset_type].copy()
    
    # Convert to EPSG:28992 and extract centroids
    assets_filtered = assets_filtered.to_crs("EPSG:28992")
    assets_filtered['geometry'] = assets_filtered.geometry.centroid
    
    # Create Voronoi polygons with asset IDs
    points = assets_filtered.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
    asset_ids = assets_filtered.index.tolist()
    if not points:
        return gpd.GeoDataFrame(columns=['asset_id', 'geometry'], crs="EPSG:28992")

    # Check if boundary has CRS and convert if needed
    if boundary is not None and hasattr(boundary, 'crs') and boundary.crs != "EPSG:28992":
        boundary = boundary.to_crs("EPSG:28992")

    vor = Voronoi(points)
    polygons = []
    valid_asset_ids = []

    # If no boundary is provided, use buffered convex hull of centroids
    centroids_union = assets_filtered.geometry.unary_union
    convex_hull = centroids_union.convex_hull.buffer(200)  # 200 meters buffer

    if boundary is None:
        boundary = convex_hull
    else:
        # Make sure boundary is a geometry, not a GeoDataFrame/GeoSeries
        if hasattr(boundary, 'geometry'):
            boundary = boundary.geometry.unary_union
        boundary = boundary.intersection(convex_hull)

    # Sample a few points to verify coordinate system
    if len(points) > 0:
        sample_points = points[:3]
        print(f"Sample points: {sample_points}")

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not -1 in region and len(region) > 0:
            try:
                polygon = Polygon([vor.vertices[i] for i in region])
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                # Clip polygon to boundary
                clipped_polygon = polygon.intersection(boundary)
                if not clipped_polygon.is_empty and clipped_polygon.is_valid:
                    polygons.append(clipped_polygon)
                    valid_asset_ids.append(asset_ids[point_idx])
                    
            except Exception as e:
                print(f"Error processing Voronoi region for asset {asset_ids[point_idx]}: {e}")

    voronoi_gdf = gpd.GeoDataFrame({'asset_id': valid_asset_ids, 'geometry': polygons}, crs="EPSG:28992")
    
    # Finally, clip the gdf to the study area
    if boundary is not None:
        try:
            voronoi_gdf = gpd.clip(voronoi_gdf, boundary)
        except Exception as e:
            print(f"Error clipping Voronoi polygons: {e}")
    
    return voronoi_gdf

def create_voronoi_boundary(gdf_assets, buffer=200):
    """Create a boundary polygon for Voronoi tessellation based on asset centroids with buffer
    
    Args:
        gdf_assets: GeoDataFrame with asset geometries
        buffer: Buffer distance in meters (default is 200)

    Returns:
        GeoSeries: Boundary polygon for Voronoi clipping
    """
    # Convert to projected CRS to find the centroids and buffer in meters
    projected_centroids = gdf_assets.to_crs("EPSG:28992").geometry.centroid
    centroids_union = projected_centroids.unary_union
    convex_hull = centroids_union.convex_hull.buffer(buffer)
    return convex_hull

def assign_impact_metric_to_voronoi(voronoi_gdf, impact_data, impact_column='aantal_inwoners'):
    """Assign impact metric to Voronoi polygons using area-weighted intersection
    
    Args:
        voronoi_gdf: GeoDataFrame with Voronoi polygons and asset IDs
        impact_data: GeoDataFrame with impact metric data (e.g., population)
        impact_column: Column name in impact_data containing the metric to assign

    Returns:
        GeoDataFrame: Voronoi polygons with assigned impact metric
    """
    if voronoi_gdf.empty:
        print("Warning: Empty Voronoi GeoDataFrame provided")
        return voronoi_gdf
        
    voronoi = voronoi_gdf.copy()
    impact_grid = impact_data.copy()
    
    # Ensure CRS match
    impact_grid = impact_grid.to_crs(voronoi.crs)
    
    # Calculate area of each impact grid cell
    impact_grid["cell_area"] = impact_grid.geometry.area
    
    # Create spatial index for impact data
    impact_sindex = impact_grid.sindex
    
    # Process each Voronoi polygon - use asset_id directly
    asset_impacts = {}
    
    for _, row in voronoi.iterrows():
        voronoi_geom = row.geometry
        asset_id = row['asset_id']
        
        # Use spatial index to quickly find potential intersections
        possible_matches_index = list(impact_sindex.intersection(voronoi_geom.bounds))
        possible_matches = impact_grid.iloc[possible_matches_index]
        
        # Calculate total impact for this Voronoi polygon
        total_weighted_impact = 0.0
        
        # Calculate precise intersections with each potential matching grid cell
        for _, impact_cell in possible_matches.iterrows():
            if voronoi_geom.intersects(impact_cell.geometry):
                intersection = voronoi_geom.intersection(impact_cell.geometry)
                intersect_area = intersection.area
                
                # Area-weighted impact calculation
                weighted_impact = (
                    impact_cell[impact_column] * intersect_area / impact_cell["cell_area"]
                )
                total_weighted_impact += weighted_impact
        
        # Store the result directly in the dictionary using asset_id as key
        asset_impacts[asset_id] = total_weighted_impact
    
    # Add the impact metrics directly to the Voronoi GeoDataFrame
    voronoi['assigned_impact_metric'] = voronoi['asset_id'].map(asset_impacts).fillna(0)
    
    # Print summary
    total_impact = voronoi['assigned_impact_metric'].sum()
    nonzero_count = (voronoi['assigned_impact_metric'] > 0).sum()
    print(f"Population assignment: {nonzero_count}/{len(voronoi)} polygons with impact values")
    print(f"Total assigned population: {total_impact:.0f}")
    
    return voronoi

def prepare_population_impact_data(population_data, voronoi_gdf=None):
    """
    Prepare population impact data using pre-computed Voronoi polygons

    Args:
        population_data: GeoDataFrame with population data
        voronoi_gdf: GeoDataFrame with Voronoi polygons and asset_id column (optional)

    Returns:
        dict: Dictionary with asset_id keys and population impact values
    """
    # Initialize asset population map
    asset_population_map = {}
    
    # If pre-computed Voronoi GeoDataFrame is provided, use it directly
    if voronoi_gdf is not None and not voronoi_gdf.empty:
        # Assign population data to Voronoi polygons
        voronoi_with_impact = assign_impact_metric_to_voronoi(
            voronoi_gdf, population_data, impact_column='aantal_inwoners')
        
        # Map asset_id to assigned population
        for _, row in voronoi_with_impact.iterrows():
            asset_population_map[row['asset_id']] = row['assigned_impact_metric']
            
    else:
        # No Voronoi data provided
        print("Warning: No Voronoi polygons provided for population impact calculation")
    
    return asset_population_map

def calculate_population_impacts(detailed_results, asset_population_map):
    """
    Calculate population impacts for each timestep based on operational status.
    
    Args:
        detailed_results: List of dictionaries, each representing a timestep
                         with arrays for metrics like 'operational'
        asset_population_map: Dictionary mapping asset_id to population
    
    Returns:
        pd.DataFrame: DataFrame with timestep data and population impact metrics
    """
    # Convert to DataFrame for easier manipulation
    timestep_results = []
    
    # Process each timestep
    for timestep_dict in detailed_results:
        # Create a copy of the timestep dictionary
        timestep_data = timestep_dict.copy()
        
        # Get asset IDs for this timestep
        asset_ids = timestep_data.get('asset_id', range(len(timestep_data['operational'])))
        
        # Initialize population metrics
        affected_population = 0
        served_population = 0
        total_population = sum(asset_population_map.values())
        
        # Calculate population impacts for this timestep
        for idx, asset_id in enumerate(asset_ids):
            # Skip if asset not in population map
            if asset_id not in asset_population_map:
                continue
                
            pop = asset_population_map[asset_id]
            # Check if operational array exists and has values
            if 'operational' in timestep_data and idx < len(timestep_data['operational']):
                is_operational = timestep_data['operational'][idx]
                
                if is_operational:
                    served_population += pop
                else:
                    affected_population += pop
        
        # Add population metrics to timestep data
        timestep_data['affected_population'] = affected_population
        timestep_data['served_population'] = served_population
        timestep_data['total_population'] = total_population
        timestep_data['affected_population_ratio'] = affected_population / total_population if total_population > 0 else 0
        
        # Add to results list
        timestep_results.append(timestep_data)
    
    # Convert to DataFrame
    population_impact_df = pd.DataFrame(timestep_results)
    return population_impact_df

def update_voll_rates(land_use_data):
    """
    Update VOLL rates and consumption values based on actual land use areas.
    Uses MWh as the energy unit throughout.
    
    Args:
        land_use_data: GeoDataFrame with land use data
        
    Returns:
        tuple: (consumption_per_sqm [MWh/m²/h], voll_per_sqm [€/m²/h])
    """
    global CONSUMPTION_PER_SQM, VOLL_PER_SQM
    
    # Calculate total area by land use in sq meters
    land_use_data['Area'] = land_use_data.geometry.area
    total_area_by_land_use = land_use_data.groupby('BG2017')['Area'].sum()
    
    # Create a dictionary of grouping: sum of areas of all the classes in the grouping
    grouped_area_by_land_use = {
        grouping: sum(total_area_by_land_use.get(land_use_class, 0) for land_use_class in cat)
        for grouping, cat in LU_CAT_DICT.items()
    }
    
    # Consumption per square meter [MWh/m²/h]
    CONSUMPTION_PER_SQM = {
        grouping: cons / grouped_area_by_land_use[grouping] if grouped_area_by_land_use[grouping] > 0 else 0
        for grouping, cons in LU_CONSUMPT_DICT.items()
    }
    
    # VOLL per square meter [€/m²/h = MWh/m²/h * €/MWh]
    VOLL_PER_SQM = {
        grouping: cons * LU_VOLL_DICT[grouping] for grouping, cons in CONSUMPTION_PER_SQM.items()
    }
    
    print("Consumption per square meter by land use [MWh/h/m²]:")
    for group, value in CONSUMPTION_PER_SQM.items():
        print(f"  {group}: {value:.10f}")
    
    print("\nVOLL per square meter by land use [€/h/m²]:")
    for group, value in VOLL_PER_SQM.items():
        print(f"  {group}: {value:.6f}")
    
    return CONSUMPTION_PER_SQM, VOLL_PER_SQM

def map_bg_code_to_category(bg_code):
    """
    Map bg 2017 land use codes to categories using VOLL data mapping.
    
    Args:
        bg_code: bg 2017 land use code
        
    Returns:
        str: Land use category name
    """
    # Use the BG to group mapping if available
    if bg_code in BG_TO_GROUP_MAP:
        return BG_TO_GROUP_MAP[bg_code]
    
    # Fallback to original mapping for codes not in VOLL data
    # Residential
    if bg_code in [20]:
        return 'residential'
    # Commercial
    elif bg_code in [21,24,43,44]:
        return 'commercial'
    # Industrial
    elif bg_code in [30,31,33,34,50]:
        return 'industrial'
    # Public services (including recreation, transport)
    elif bg_code in [22,23,32,41]:
        return 'public_services'
    # Transport
    elif bg_code in [10,11,12]:
        return 'transport'
    # Other (forest, water, etc.)
    else:
        return 'other'

def calculate_monetized_impacts_ema(detailed_results, asset_land_use_map, voll_rates=None):
    """
    Calculate monetized impacts for EMA simulations based on land use types and Value of Lost Load (VoLL).
    
    Args:
        detailed_results: List of timestep dictionaries with arrays for metrics
        asset_land_use_map: Dictionary mapping asset_id to dict of land use types and areas
        voll_rates: Dictionary mapping land use types to VOLL per square meter [€/m²/h]. 
                     If None, uses global VOLL_PER_SQM.

    Returns:
        pd.DataFrame: DataFrame with timestep data and monetary impact metrics
    """
    # Use global VOLL_PER_SQM directly if no rates provided
    if voll_rates is None:
        voll_rates = VOLL_PER_SQM if VOLL_PER_SQM else {}
    
    # Convert to DataFrame for easier manipulation
    timestep_results = pd.DataFrame(detailed_results)
    
    # Initialize impact metrics
    timestep_results['monetary_impact_total'] = 0.0
    for category in set(voll_rates.keys()):
        timestep_results[f'monetary_impact_{category}'] = 0.0
    
    # Process each timestep
    for idx, timestep_dict in enumerate(detailed_results):
        asset_ids = timestep_dict.get('asset_id', np.arange(len(timestep_dict['operational'])))
        operational = timestep_dict['operational']
        
        # Process each asset
        for i, asset_id in enumerate(asset_ids):
            # Skip if asset operational or not in land use map
            if operational[i] or asset_id not in asset_land_use_map:
                continue
                
            # Get land use breakdown for this asset
            land_use_areas = asset_land_use_map[asset_id]
            
            # Calculate impact for each land use type
            for land_use_type, area in land_use_areas.items():
                if land_use_type in voll_rates:
                    # Calculate monetary impact directly using VOLL per square meter [€/m²/h * m² = €/h]
                    impact = area * voll_rates[land_use_type]
                    
                    # Add to total and category impacts
                    timestep_results.at[idx, f'monetary_impact_{land_use_type}'] += impact
                    timestep_results.at[idx, 'monetary_impact_total'] += impact
    
    return timestep_results

def calculate_cumulative_monetized_impacts_ema(detailed_results, asset_to_lu):
    """
    Calculate cumulative monetized impacts for EMA simulations using fast vectorized operations.
    
    Args:
        detailed_results: List of timestep dictionaries with arrays for metrics
        asset_to_lu: Pre-computed lookup dict {asset_id: [(lu_type, area, voll_rate), ...]}

    Returns:
        pd.DataFrame: DataFrame with timestep data and cumulative monetary impact metrics
    """
    
    # Get monetary categories from the pre-computed lookup
    monetary_categories = set()
    for asset_data in asset_to_lu.values():
        for lu_type, _, _ in asset_data:
            monetary_categories.add(lu_type)

    # Preallocate results dictionary
    n_timesteps = len(detailed_results)
    impact_by_ts = {ts_idx: {f'monetary_impact_{cat}': 0.0 for cat in monetary_categories}
                    for ts_idx in range(n_timesteps)}
    
    # Initialize monetary_impact_total for all timesteps
    for ts_idx in impact_by_ts.keys():
        impact_by_ts[ts_idx]['monetary_impact_total'] = 0.0
    
    # Single pass through timesteps and assets
    for ts_idx, ts in enumerate(detailed_results):
        asset_ids = ts.get('asset_id', np.arange(len(ts['operational'])))
        operational = np.array(ts['operational'], dtype=bool)
        
        for i, aid in enumerate(asset_ids):
            if operational[i] or aid not in asset_to_lu:
                continue
            
            # Calculate impact for this asset's land use types
            for lu_type, area, voll_rate in asset_to_lu[aid]:
                impact = area * voll_rate
                impact_by_ts[ts_idx][f'monetary_impact_{lu_type}'] += impact
                impact_by_ts[ts_idx]['monetary_impact_total'] += impact

    # Convert to DataFrame
    df = pd.DataFrame([
        {'timestep': ts_idx, **impact_by_ts[ts_idx]} 
        for ts_idx in range(n_timesteps)
    ])

    # Apply cumsum on monetary columns
    monetary_cols = [col for col in df.columns if col.startswith('monetary_impact_')]
    df[monetary_cols] = df[monetary_cols].cumsum()

    return df

def prepare_land_use_impact_data(land_use_data, voronoi_gdf=None):
    """
    Prepare land use data for impact calculation by associating land use types 
    with assets based on Voronoi polygons.
    
    Args:
        land_use_data: GeoDataFrame with land use data
        voronoi_gdf: GeoDataFrame with Voronoi polygons and asset_id column (optional)
    
    Returns:
        dict: Dictionary mapping asset_id to dict of land use types and areas
    """
    # Update VOLL rates based on actual land use data
    update_voll_rates(land_use_data)
    
    # Create asset to land use mapping
    asset_land_use_map = {}
    
    # Create a spatial index for the land use data to speed up operations
    land_use_sindex = land_use_data.sindex
    
    # If we have a voronoi_gdf, use that directly
    if voronoi_gdf is not None and not voronoi_gdf.empty:
        voronoi_data = voronoi_gdf.to_crs(land_use_data.crs)
    else:
        # No Voronoi data provided
        print("Warning: No Voronoi polygons provided for land use impact calculation")
        return asset_land_use_map
    
    # Process each voronoi polygon
    for _, row in voronoi_data.iterrows():
        asset_id = row['asset_id']
        voronoi_geom = row.geometry
        
        # Use spatial index to quickly find potential intersections
        possible_matches_index = list(land_use_sindex.intersection(voronoi_geom.bounds))
        possible_matches = land_use_data.iloc[possible_matches_index]
        
        # Calculate precise intersections
        land_use_areas = {}
        
        for _, land_use in possible_matches.iterrows():
            land_use_geom = land_use.geometry
            if voronoi_geom.intersects(land_use_geom):
                intersection = voronoi_geom.intersection(land_use_geom)
                area = intersection.area
                
                # Map BG land use code to category using updated mapping
                land_use_type = map_bg_code_to_category(land_use['BG2017'])
                
                # Add area to land use type
                if land_use_type not in land_use_areas:
                    land_use_areas[land_use_type] = 0
                land_use_areas[land_use_type] += area
        
        asset_land_use_map[asset_id] = land_use_areas
    
    return asset_land_use_map