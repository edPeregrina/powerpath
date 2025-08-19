import hashlib
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
from src.caching import create_hazard_extraction_cache_key


# def find_hazard_value_at_points_optimized(hazard_map_path, gdf_assets, day_counter, extraction_method='max', 
#                                           hazard_cache=None, hazard_dir=None):
#     """
#     Extract hazard values at asset locations from a raster file with different extraction methods and caching support.
    
#     Args:
#         hazard_map_path (str or Path): Path to the hazard raster file
#         gdf_assets (GeoDataFrame): GeoDataFrame containing asset geometries
#         day_counter (int): Day counter for column naming
#         extraction_method (str): Method for value extraction:
#             - 'max': Maximum value within geometry (best for flood risk)
#             - 'median': Median value within geometry (robust to outliers)
#             - 'mean': Mean value within geometry
#             - 'point'/'centroid': Point sampling at centroid (fastest)
#         hazard_cache (dict, optional): Cache dictionary for hazard extraction results
#         hazard_dir (str or Path, optional): Hazard directory for cache key generation
    
#     Returns:
#         GeoDataFrame: Updated GeoDataFrame with hazard values
#     """
#     # Create column names
#     day_counter_str = str(day_counter).zfill(2)
#     haz_col_str = f'EV{day_counter}_ma'
#     haz_val_str = f'hazard_value_{day_counter_str}'
    
#     # Check cache first
#     if hazard_cache is not None:
#         # Create a simple hash of the GeoDataFrame geometry for cache validation
#         geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
#         gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
        
#         cache_key = create_hazard_extraction_cache_key(
#             hazard_map_path, day_counter, extraction_method, gdf_hash, hazard_dir
#         )
        
#         if cache_key in hazard_cache:
#             cached_result = hazard_cache[cache_key]
            
#             # Validate cache integrity and hazard map match
#             if (len(cached_result.get('haz_col_values', [])) == len(gdf_assets) and
#                 len(cached_result.get('haz_val_values', [])) == len(gdf_assets) and
#                 cached_result.get('hazard_map') == str(hazard_map_path)):
                
#                 print(f"Using cached hazard extraction for day {day_counter} from {Path(hazard_map_path).name}")
                
#                 # Apply cached values to a copy of the GeoDataFrame
#                 gdf_result = gdf_assets.copy()
#                 gdf_result[haz_col_str] = cached_result['haz_col_values']
#                 gdf_result[haz_val_str] = cached_result['haz_val_values']
                
#                 return gdf_result
#             else:
#                 print(f"Cache validation failed for {cache_key}, recomputing...")
    
#     # If no cache hit, compute hazard values
#     try:
#         import rasterio
#         import rasterio.mask
#         from rasterio.features import geometry_mask
#         import numpy as np
        
#         # Copy the GeoDataFrame to avoid modifying the original
#         gdf_result = gdf_assets.copy()
        
#         # Create column names
#         day_counter_str = str(day_counter).zfill(2)
#         haz_col_str = f'EV{day_counter}_ma'
#         haz_val_str = f'hazard_value_{day_counter_str}'
        
#         print(f"Extracting hazard values using method: {extraction_method}")

#         # Open the raster file
#         with rasterio.open(hazard_map_path) as src:
#             # Ensure geometries are in the correct CRS
#             if gdf_result.crs != src.crs:
#                 gdf_result = gdf_result.to_crs(src.crs)
            
#             hazard_values = []
            
#             if extraction_method in ['point', 'centroid', 'bilinear']:
#                 # Fast point-based sampling
#                 coords = [(geom.centroid.x, geom.centroid.y) for geom in gdf_result.geometry]
                
#                 # Sample raster values at coordinates
#                 sampled_values = list(src.sample(coords, indexes=1))
#                 hazard_values = [val[0] if val.size > 0 and not np.isnan(val[0]) and val[0] >= 0 else 0.0 
#                                for val in sampled_values]
                
#             else:
#                 # Polygon-based extraction for max, median, mean
#                 for geom in gdf_result.geometry:
#                     try:
#                         if hasattr(geom, 'x') and hasattr(geom, 'y'):
#                             # Point geometry - just sample the point
#                             sampled = list(src.sample([(geom.x, geom.y)]))
#                             value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
#                             hazard_values.append(value)
                        
#                         else:
#                             # Polygon geometry - extract all pixels within the polygon
#                             # Mask the raster with the polygon
#                             masked_array, mask_transform = rasterio.mask.mask(src, [geom], crop=True, nodata=src.nodata)
                            
#                             # Get the first band data
#                             band_data = masked_array[0]
                            
#                             # Remove nodata values and negative values
#                             if src.nodata is not None:
#                                 valid_data = band_data[(band_data != src.nodata) & (~np.isnan(band_data)) & (band_data >= 0)]
#                             else:
#                                 valid_data = band_data[(~np.isnan(band_data)) & (band_data >= 0)]
                            
#                             # Calculate the requested statistic
#                             if len(valid_data) > 0:
#                                 if extraction_method == 'max':
#                                     value = float(np.max(valid_data))
#                                 elif extraction_method == 'median':
#                                     value = float(np.median(valid_data))
#                                 elif extraction_method == 'mean':
#                                     value = float(np.mean(valid_data))
#                                 else:
#                                     # Default to max if unknown method
#                                     value = float(np.max(valid_data))
#                             else:
#                                 # No valid data in polygon - fallback to centroid sampling
#                                 centroid = geom.centroid
#                                 sampled = list(src.sample([(centroid.x, centroid.y)]))
#                                 value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
                            
#                             hazard_values.append(value)
                        
                    
#                     except Exception as e:
#                         # Fallback to centroid sampling if polygon processing fails
#                         print(f"Warning: Polygon extraction failed for geometry, using centroid: {e}")
#                         try:
#                             centroid = geom.centroid if hasattr(geom, 'centroid') else geom
#                             coords = (centroid.x, centroid.y) if hasattr(centroid, 'x') else (geom.x, geom.y)
#                             sampled = list(src.sample([coords]))
#                             value = sampled[0][0] if sampled and sampled[0].size > 0 and not np.isnan(sampled[0][0]) and sampled[0][0] >= 0 else 0.0
#                             hazard_values.append(value)
#                         except Exception as e2:
#                             print(f"Error sampling centroid: {e2}, using 0.0")
#                             hazard_values.append(0.0)
            
#             # Add values to both possible column names
#             gdf_result[haz_col_str] = hazard_values
#             gdf_result[haz_val_str] = hazard_values
            
#             # Cache the results if cache is available
#             if hazard_cache is not None:
#                 geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
#                 gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
                
#                 cache_key = create_hazard_extraction_cache_key(
#                     hazard_map_path, day_counter, extraction_method, gdf_hash, hazard_dir
#                 )
                
#                 hazard_cache[cache_key] = {
#                     'haz_col_values': hazard_values.copy(),
#                     'haz_val_values': hazard_values.copy(),
#                     'timestamp': datetime.now().isoformat(),
#                     'hazard_map': str(hazard_map_path),
#                     'day_counter': day_counter,
#                     'extraction_method': extraction_method,
#                     'num_assets': len(gdf_assets),
#                     'gdf_hash': gdf_hash
#                 }
                
#                 print(f"Cached hazard extraction results for day {day_counter} from {Path(hazard_map_path).name}")
        
#         return gdf_result
        
#     except ImportError:
#         print("Warning: rasterio not available.")
#         return None
        
#     except Exception as e:
#         print(f"Error extracting hazard values from {hazard_map_path}: {e}")
#         # Return original GeoDataFrame with zero hazard values
#         day_counter_str = str(day_counter).zfill(2)
#         haz_col_str = f'EV{day_counter}_ma'
#         haz_val_str = f'hazard_value_{day_counter_str}'
        
#         gdf_result = gdf_assets.copy()
#         gdf_result[haz_col_str] = 0.0
#         gdf_result[haz_val_str] = 0.0
        
#         return gdf_result


def get_valid_mean(x_value, **kwargs):
    """
    Calculate mean of valid (non-masked, non-negative) values from rasterstats.
    **kwargs should not be removed - properties var is passed in zonal_stats.
    """
    import numpy as np
    from numpy.ma import MaskedArray
    
    if not isinstance(x_value, MaskedArray):
        return np.nan
    if x_value.mask.all():
        return np.nan
    
    # Additional filtering for your hazard data (non-negative values)
    valid_data = x_value.compressed()  # Get non-masked values
    valid_data = valid_data[valid_data >= 0]  # Filter out negative values
    
    if len(valid_data) == 0:
        return 0.0  # Return 0 instead of NaN for consistency with your logic
    
    return float(np.mean(valid_data))




def find_hazard_value_at_points_optimized(hazard_map_path, gdf_assets, day_counter, extraction_method='max', 
                                          hazard_cache=None, hazard_dir=None):
    """
    Extract hazard values at asset locations from a raster file with different extraction methods and caching support.
    Uses rasterstats for optimized performance.
    
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
    day_counter_str = str(day_counter).zfill(2)
    haz_col_str = f'EV{day_counter}_ma'
    haz_val_str = f'hazard_value_{day_counter_str}'
    
    # Check cache first
    if hazard_cache is not None:
        # Create a simple hash of the GeoDataFrame geometry for cache validation
        geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
        gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
        
        cache_key = create_hazard_extraction_cache_key(
            hazard_map_path, day_counter, extraction_method, gdf_hash, hazard_dir
        )
        
        if cache_key in hazard_cache:
            cached_result = hazard_cache[cache_key]
            
            # Validate cache integrity and hazard map match
            if (len(cached_result.get('haz_col_values', [])) == len(gdf_assets) and
                len(cached_result.get('haz_val_values', [])) == len(gdf_assets) and
                cached_result.get('hazard_map') == str(hazard_map_path)):
                
                print(f"Using cached hazard extraction for day {day_counter} from {Path(hazard_map_path).name}")
                
                # Apply cached values to a copy of the GeoDataFrame
                gdf_result = gdf_assets.copy()
                gdf_result[haz_col_str] = cached_result['haz_col_values']
                gdf_result[haz_val_str] = cached_result['haz_val_values']
                
                return gdf_result
            else:
                print(f"Cache validation failed for {cache_key}, recomputing...")
    
    # If no cache hit, compute hazard values using rasterstats
    try:
        from rasterstats import zonal_stats
        import numpy as np
        
        # Copy the GeoDataFrame to avoid modifying the original
        gdf_result = gdf_assets.copy()
        
        print(f"Extracting hazard values using method: {extraction_method}")

        # Prepare CRS - rasterstats handles CRS automatically if both have CRS info
        hazard_values = []
        
        if extraction_method in ['point', 'centroid']:
            # Point-based sampling using centroids
            centroids = gdf_result.geometry.centroid
            
            # Use zonal_stats with point geometries (effectively point sampling)
            stats = zonal_stats(
                centroids, 
                str(hazard_map_path),
                all_touched=True,
                stats=['mean'],  # For points, mean = the sampled value
                nodata=-9999
            )
            
            hazard_values = [
                stat['mean'] if stat['mean'] is not None and stat['mean'] >= 0 else 0.0 
                for stat in stats
            ]
            
        else:
            # Polygon-based extraction using zonal_stats
            stat_method = extraction_method if extraction_method in ['max', 'mean', 'median'] else 'max'
                        
            if stat_method == 'mean':
                # Use the enhanced valid mean function
                stats = zonal_stats(
                    gdf_result.geometry,
                    str(hazard_map_path),
                    all_touched=True,
                    add_stats={'valid_mean': get_valid_mean},
                    nodata=-9999
                )
                hazard_values = [
                    stat['valid_mean'] if stat['valid_mean'] is not None and not np.isnan(stat['valid_mean']) else 0.0 
                    for stat in stats
                ]

            else:
                # Use standard statistics (max, median)
                stats = zonal_stats(
                    gdf_result.geometry,
                    str(hazard_map_path),
                    all_touched=True,
                    stats=[stat_method],
                    nodata=-9999
                )
                
                hazard_values = []
                for stat in stats:
                    if stat[stat_method] is not None and stat[stat_method] >= 0:
                        hazard_values.append(float(stat[stat_method]))
                    else:
                        # Fallback: try to get any valid value or use 0.0
                        hazard_values.append(0.0)
        
        # Add values to both possible column names
        gdf_result[haz_col_str] = hazard_values
        gdf_result[haz_val_str] = hazard_values
        
        # Cache the results if cache is available
        if hazard_cache is not None:
            geom_str = str(len(gdf_assets)) + str(gdf_assets.geometry.bounds.sum().sum())
            gdf_hash = hashlib.md5(geom_str.encode()).hexdigest()[:8]
            
            cache_key = create_hazard_extraction_cache_key(
                hazard_map_path, day_counter, extraction_method, gdf_hash, hazard_dir
            )
            
            hazard_cache[cache_key] = {
                'haz_col_values': hazard_values.copy(),
                'haz_val_values': hazard_values.copy(),
                'timestamp': datetime.now().isoformat(),
                'hazard_map': str(hazard_map_path),
                'day_counter': day_counter,
                'extraction_method': extraction_method,
                'num_assets': len(gdf_assets),
                'gdf_hash': gdf_hash
            }
            
            print(f"Cached hazard extraction results for day {day_counter} from {Path(hazard_map_path).name}")
    
        return gdf_result
        
    except ImportError as ie:
        print(f"Warning: Required library not available: {ie}")
        print("Please install rasterstats: pip install rasterstats")
        # Fallback to original method if rasterstats not available
        return _fallback_rasterio_method(hazard_map_path, gdf_assets, day_counter, extraction_method, hazard_cache, hazard_dir)
        
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