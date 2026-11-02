import hashlib
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from src.caching import create_hazard_extraction_cache_key, save_hazard_extraction_cache, save_cache_if_due


def _sample_point_values(src, points):
    """Sample raster values directly at point coordinates.

    exactextract's coverage-fraction algorithm cannot resolve zero-area
    geometries -- it returns no match even for a point sitting exactly on a
    valid, nonzero pixel -- so point geometries must bypass zonal stats
    entirely and read the raster directly.
    """
    coords = [(geom.x, geom.y) for geom in points]
    if not coords:
        return []
    sampled = list(src.sample(coords, indexes=1))
    return [float(v[0]) if v.size and v[0] >= 0 else 0.0 for v in sampled]


def _zonal_polygon_values(src, band, geometries, stat_method):
    """Coverage-weighted zonal stats for polygon geometries via exactextract.

    The raster is wrapped with an explicit nodata override rather than
    trusting the GeoTIFF's own nodata tag, which does not always match the
    -9999 sentinel used elsewhere in this pipeline (observed file tags of
    0.0, which would incorrectly mask legitimate zero-depth pixels).

    Negative pixels are masked before aggregation so means are computed
    only over valid depths (e.g. -1/1 depths average to 1, not 0).
    """
    import numpy as np
    import geopandas as gpd
    from exactextract import exact_extract
    from exactextract.raster import NumPyRasterSource

    band_masked = np.where(band < 0, -9999, band)
    raster_source = NumPyRasterSource(
        band_masked,
        xmin=src.bounds.left, ymin=src.bounds.bottom,
        xmax=src.bounds.right, ymax=src.bounds.top,
        nodata=-9999,
        srs_wkt=src.crs.to_wkt() if src.crs else None,
    )
    vec = gpd.GeoDataFrame(geometry=list(geometries), crs=src.crs)
    result = exact_extract(raster_source, vec, [stat_method])

    values = []
    for feature in result:
        value = feature['properties'][stat_method]
        if value is None or (isinstance(value, float) and np.isnan(value)) or value < 0:
            values.append(0.0)
        else:
            values.append(float(value))
    return values





def find_hazard_value_at_points_optimized(hazard_map_path, gdf_assets, map_counter, extraction_method='max', 
                                          hazard_cache=None, hazard_dir=None, _config=None):
    """
    Extract hazard values at asset locations from a raster file, with caching support.
    Assets may be a mix of Point and Polygon geometries (e.g. OSM nodes vs.
    building footprints); Points are sampled directly and Polygons use
    exactextract's coverage-weighted zonal stats (see module-level helpers).

    Args:
        hazard_map_path (str or Path): Path to the hazard raster file
        gdf_assets (GeoDataFrame): GeoDataFrame containing asset geometries
        day_counter (int): Day counter for column naming
        extraction_method (str): Method for value extraction:
            - 'max': Maximum value within geometry (best for flood risk)
            - 'median': Median value within geometry (robust to outliers)
            - 'mean': Mean value within geometry
            - 'point'/'centroid': Point sampling at centroid (fastest)
        hazard_cache (dict, optional): Cache dictionary for hazard extraction results
        hazard_dir (str or Path, optional): Hazard directory for cache key generation
    
    Returns:
        GeoDataFrame: Updated GeoDataFrame with hazard values
    """
    import hashlib
    from datetime import datetime
    from pathlib import Path
    import sys
    sys.path.append(str(Path.cwd().parent))
    from src.caching import create_hazard_extraction_cache_key
    
    # Create column names
    map_counter_str = str(map_counter).zfill(2)
    haz_col_str = f'EV{map_counter}_ma'
    haz_val_str = f'hazard_value_{map_counter_str}'

    # Check cache first
    if hazard_cache is not None:
        # Create a simple hash of the GeoDataFrame geometry for cache validation
        geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
        gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
        
        cache_key = create_hazard_extraction_cache_key(
            hazard_map_path, map_counter, extraction_method, gdf_hash, hazard_dir
        )
        
        if cache_key in hazard_cache:
            cached_result = hazard_cache[cache_key]
            
            # Validate cache integrity and hazard map match
            if (len(cached_result.get('haz_col_values', [])) == len(gdf_assets) and
                len(cached_result.get('haz_val_values', [])) == len(gdf_assets) and
                cached_result.get('hazard_map') == str(hazard_map_path)):

                # print(f"\nUsing cached hazard extraction for map {map_counter} from {Path(hazard_map_path).name}")

                # Apply cached values to a copy of the GeoDataFrame
                gdf_result = gdf_assets.copy()
                gdf_result[haz_col_str] = cached_result['haz_col_values']
                gdf_result[haz_val_str] = cached_result['haz_val_values']
                
                return gdf_result
            else:
                print(f"Cache validation failed for {cache_key}, recomputing...")
    
    # If no cache hit, compute hazard values
    try:
        import numpy as np
        import rasterio

        # Copy the GeoDataFrame to avoid modifying the original
        gdf_result = gdf_assets.copy()
        
        print(f"Extracting hazard values using method: {extraction_method}")

        with rasterio.open(hazard_map_path) as src:
            raster_geoms = gdf_result.geometry
            if gdf_result.crs and src.crs and gdf_result.crs != src.crs:
                raster_geoms = raster_geoms.to_crs(src.crs)

            if extraction_method in ('point', 'centroid'):
                hazard_values = _sample_point_values(src, raster_geoms.centroid)
            else:
                stat_method = extraction_method if extraction_method in ('max', 'mean', 'median') else 'max'
                is_point = raster_geoms.geom_type == 'Point'

                hazard_values = [0.0] * len(gdf_result)
                if is_point.any():
                    point_values = _sample_point_values(src, raster_geoms[is_point])
                    for idx, value in zip(np.flatnonzero(is_point.values), point_values):
                        hazard_values[idx] = value
                if (~is_point).any():
                    band = src.read(1)
                    polygon_values = _zonal_polygon_values(src, band, raster_geoms[~is_point], stat_method)
                    for idx, value in zip(np.flatnonzero((~is_point).values), polygon_values):
                        hazard_values[idx] = value

        # Add values to both possible column names
        gdf_result[haz_col_str] = hazard_values
        gdf_result[haz_val_str] = hazard_values
        
        # Cache the results if cache is available
        if hazard_cache is not None:
            geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
            gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
            
            cache_key = create_hazard_extraction_cache_key(
                hazard_map_path, map_counter, extraction_method, gdf_hash, hazard_dir
            )
            
            hazard_cache[cache_key] = {
                'haz_col_values': hazard_values.copy(),
                'haz_val_values': hazard_values.copy(),
                'timestamp': datetime.now().isoformat(),
                'hazard_map': str(hazard_map_path),
                'map_counter': map_counter,
                'extraction_method': extraction_method,
                'num_assets': len(gdf_assets),
                'gdf_hash': gdf_hash
            }
            interim_dir = _config['interim_dir'] if _config and 'interim_dir' in _config else None
            save_cache_if_due(
                lambda: save_hazard_extraction_cache(hazard_cache, interim_dir, hazard_dir),
                cache_name="hazard_extraction_cache",
                hazard_dir=hazard_dir,
            )
            print(f"Cached hazard extraction results for map {map_counter} from {Path(hazard_map_path).name}")
        return gdf_result
        
    except ImportError as ie:
        print(f"Warning: Required library not available: {ie}")
        print("Please install exactextract: pip install exactextract")
        # Fallback to original method if exactextract not available
        return _fallback_rasterio_method(hazard_map_path, gdf_assets, map_counter, extraction_method, hazard_cache, hazard_dir)
    except Exception as e:
        print(f"Error extracting hazard values from {hazard_map_path}: {e}")
        # Return original GeoDataFrame with zero hazard values
        gdf_result = gdf_assets.copy()
        gdf_result[haz_col_str] = 0.0
        gdf_result[haz_val_str] = 0.0
        
        return gdf_result


def _fallback_rasterio_method(hazard_map_path, gdf_assets, day_counter, extraction_method, hazard_cache, hazard_dir):
    """Fallback method using rasterio if rasterstats is not available"""
    try:
        import rasterio
        import rasterio.mask
        import numpy as np
        
        # Copy the GeoDataFrame to avoid modifying the original
        gdf_result = gdf_assets.copy()
        
        # Create column names
        day_counter_str = str(day_counter).zfill(2)
        haz_col_str = f'EV{day_counter}_ma'
        haz_val_str = f'hazard_value_{day_counter_str}'
        
        print(f"Using fallback rasterio method for extraction: {extraction_method}")

        # Open the raster file
        with rasterio.open(hazard_map_path) as src:
            # Ensure geometries are in the correct CRS
            if gdf_result.crs != src.crs:
                gdf_result = gdf_result.to_crs(src.crs)
            
            hazard_values = []
            
            if extraction_method in ['point', 'centroid']:
                # Fast point-based sampling
                coords = [(geom.centroid.x, geom.centroid.y) for geom in gdf_result.geometry]
                
                # Sample raster values at coordinates
                sampled_values = list(src.sample(coords, indexes=1))
                hazard_values = [val[0] if val.size > 0 and not np.isnan(val[0]) and val[0] >= 0 else 0.0 
                               for val in sampled_values]
                
            else:
                # Polygon-based extraction for max, median, mean
                for geom in gdf_result.geometry:
                    try:
                        if hasattr(geom, 'x') and hasattr(geom, 'y'):
                            # Point geometry - just sample the point
                            sampled = list(src.sample([(geom.x, geom.y)]))
                            value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
                            hazard_values.append(value)
                        
                        else:
                            # Polygon geometry - extract all pixels within the polygon
                            masked_array, mask_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=src.nodata)
                            
                            # Get the first band data
                            band_data = masked_array[0]
                            
                            # Remove nodata values and negative values
                            if src.nodata is not None:
                                valid_data = band_data[(band_data != src.nodata) & (~np.isnan(band_data)) & (band_data >= 0)]
                            else:
                                valid_data = band_data[(~np.isnan(band_data)) & (band_data >= 0)]
                            
                            # Calculate the requested statistic
                            if len(valid_data) > 0:
                                if extraction_method == 'max':
                                    value = float(np.max(valid_data))
                                elif extraction_method == 'median':
                                    value = float(np.median(valid_data))
                                elif extraction_method == 'mean':
                                    value = float(np.mean(valid_data))
                                else:
                                    value = float(np.max(valid_data))
                            else:
                                # No valid data in polygon - fallback to centroid sampling
                                centroid = geom.centroid
                                sampled = list(src.sample([(centroid.x, centroid.y)]))
                                value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
                            
                            hazard_values.append(value)
                        
                    except Exception as e:
                        # Fallback to centroid sampling if polygon processing fails
                        print(f"Warning: Polygon extraction failed for geometry, using centroid: {e}")
                        try:
                            centroid = geom.centroid if hasattr(geom, 'centroid') else geom
                            coords = (centroid.x, centroid.y) if hasattr(centroid, 'x') else (geom.x, geom.y)
                            sampled = list(src.sample([coords]))
                            value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
                            hazard_values.append(value)
                        except Exception as e2:
                            print(f"Error sampling centroid: {e2}, using 0.0")
                            hazard_values.append(0.0)
            
            # Add values to both possible column names
            gdf_result[haz_col_str] = hazard_values
            gdf_result[haz_val_str] = hazard_values
            
            return gdf_result
            
    except Exception as e:
        print(f"Fallback method also failed: {e}")
        # Return original GeoDataFrame with zero hazard values
        day_counter_str = str(day_counter).zfill(2)
        haz_col_str = f'EV{day_counter}_ma'
        haz_val_str = f'hazard_value_{day_counter_str}'
        
        gdf_result = gdf_assets.copy()
        gdf_result[haz_col_str] = 0.0
        gdf_result[haz_val_str] = 0.0
        
        return gdf_result