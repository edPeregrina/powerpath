
import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely.geometry as sg
from pyproj import Transformer
from rtree import index
from scipy.spatial import Voronoi


def create_spatial_index(gdf):
    """
    Create R-tree spatial index for fast spatial queries.
    
    Arguments:
    - gdf: GeoDataFrame containing geometries to index

    Returns:
    - R-tree spatial index
    """
    from rtree import index

    idx = index.Index()
    
    # Insert each geometry's bounding box into the index
    for i, row in gdf.iterrows():
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        idx.insert(i, bounds)
    
    return idx


def project_graph_coords(G: nx.Graph, from_crs: str, to_crs: str) -> nx.Graph:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    for n, d in G.nodes(data=True):
        d["x_m"], d["y_m"] = transformer.transform(d["x"], d["y"])
    for u, v, d in G.edges(data=True):
        if 'geometry' in d and d['geometry'] is not None:
            # Transform the geometry coordinates
            if hasattr(d['geometry'], 'coords'):
                coords = list(d['geometry'].coords)
                transformed_coords = [transformer.transform(x, y) for x, y in coords]
                transformed_geometry = sg.LineString(transformed_coords)
                d["length"] = transformed_geometry.length
            else:
                # Fallback to straight-line distance if geometry doesn't have coords
                x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
                x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
                d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        else:
            # No geometry available, use straight-line distance between nodes
            x1, y1 = G.nodes[u]["x_m"], G.nodes[u]["y_m"]
            x2, y2 = G.nodes[v]["x_m"], G.nodes[v]["y_m"]
            d["length"] = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    return G

def filter_hazard_graph(G: nx.Graph, threshold: float, hazard_column: str, 
                        l1_area_geojson=None, l2_asset_geojson=None,
                        verbose=False) -> nx.Graph:
    """
    Filter graph edges based on hazard values, excluding protected infrastructure.
    Applies active L1/L2 depth reductions by adjusting each edge threshold.
    
    Args:
        G: NetworkX graph with hazard values on edges
        threshold: Base hazard value threshold for edge removal
        hazard_column: Name of edge attribute containing hazard values
        l1_area_geojson: Optional path/GeoDataFrame for L1 depth reductions
        l2_asset_geojson: Optional path/GeoDataFrame for L2 depth reductions
        verbose: Print progress messages
    
    Returns:
        Filtered graph with hazard edges removed
    """
    from pathlib import Path

    import geopandas as gpd
    import pandas as pd
    import shapely

    def is_motorway(highway):
        if isinstance(highway, str):
            return "motorway" in highway.lower()
        elif isinstance(highway, list):
            return any("motorway" in str(h).lower() for h in highway)
        return False

    def is_protected(d):
        """Check if an edge represents protected infrastructure (bridge/tunnel)"""
        def check_attribute(val):
            if val is None:
                return False
            
            if isinstance(val, list):
                return any(
                    item is not None and 
                    str(item).strip().lower() not in ['', 'nan', 'none', 'no'] 
                    for item in val
                )
            elif isinstance(val, str):
                val_clean = val.strip().lower()
                return val_clean not in ['', 'nan', 'none', 'no']
            else:
                return pd.notna(val)

        return check_attribute(d.get("bridge")) or check_attribute(d.get("tunnel")) or check_attribute(d.get("protected"))

    edge_depth_reductions = {}

    def iter_edges_with_keys():
        if G.is_multigraph():
            return G.edges(keys=True, data=True)
        return ((u, v, None, data) for u, v, data in G.edges(data=True))

    def add_adaptation_depth_reductions(adaptation, label, default_reduction):
        if adaptation is None:
            return

        adaptation_gdf = (
            gpd.read_file(adaptation)
            if isinstance(adaptation, (str, Path))
            else adaptation.copy()
        )
        if "depth_red" not in adaptation_gdf.columns:
            adaptation_gdf['depth_red'] = default_reduction
            if verbose:
                print(
                    f"Warning: {label} GeoJSON missing 'depth_red' column, "
                    f"using default {default_reduction}m"
                )

        # Ensure correct CRS (graph is in EPSG:4326)
        if adaptation_gdf.crs != "EPSG:4326":
            adaptation_gdf = adaptation_gdf.to_crs("EPSG:4326")

        adaptation_tree = shapely.STRtree(adaptation_gdf.geometry.values)
        adapted_edge_count = 0
        for u, v, edge_key, data in iter_edges_with_keys():
            edge_geom = data.get('geometry')

            if edge_geom is None:
                from shapely.geometry import LineString
                edge_geom = LineString([
                    (G.nodes[u]['x'], G.nodes[u]['y']),
                    (G.nodes[v]['x'], G.nodes[v]['y'])
                ])

            intersecting_indices = adaptation_tree.query(
                edge_geom, predicate='intersects'
            )
            if len(intersecting_indices) > 0:
                max_reduction = adaptation_gdf.iloc[
                    intersecting_indices
                ]['depth_red'].max()
                cache_key = (u, v, edge_key)
                edge_depth_reductions[cache_key] = (
                    edge_depth_reductions.get(cache_key, 0.0) + max_reduction
                )
                adapted_edge_count += 1

        if verbose:
            print(f"Applied {label} to {adapted_edge_count} edges")

    add_adaptation_depth_reductions(l1_area_geojson, "L1", 0.3)
    add_adaptation_depth_reductions(l2_asset_geojson, "L2", 0.15)
    
    # Filter edges based on adjusted thresholds
    edges_to_remove = []
    
    for u, v, edge_key, d in iter_edges_with_keys():
        hazard_value = d.get(hazard_column, 0)
        
        # Get edge-specific depth reduction
        depth_reduction = edge_depth_reductions.get((u, v, edge_key), 0.0)
        
        # Adjusted threshold: hazard must exceed (base_threshold + reduction)
        adjusted_threshold = threshold + depth_reduction
        
        # Remove edge if hazard exceeds adjusted threshold (and not protected)
        if (hazard_value > adjusted_threshold and 
            not is_motorway(d.get("highway")) and 
            not is_protected(d)):
            edge = (u, v, edge_key) if G.is_multigraph() else (u, v)
            edges_to_remove.append(edge)
    
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if verbose:
        print(f"Removed {len(edges_to_remove)} edges (adjusted thresholds)")
    
    return G

def compile_asset_gdfs(gdf_list: list) -> "gpd.GeoDataFrame":
    """Concatenate multiple per-type asset GeoDataFrames into one unified GeoDataFrame.

    The resulting index is a RangeIndex (0, 1, 2, …) so that each row's positional
    index is also its label.  This is required by :func:`build_voronoi_service_area_map`
    and by :func:`~src.impacts.create_voronoi_for_asset_type`, which both rely on
    the index being the positional identifier of each asset in the combined array.

    Args:
        gdf_list: Ordered list of GeoDataFrames to combine (e.g.
            ``[gdf_substations, gdf_hospitals]``).  All GDFs must share the same
            CRS and must contain a ``'type'`` column.

    Returns:
        A single GeoDataFrame with a reset RangeIndex.

    Example::

        gdf_assets = compile_asset_gdfs([gdf_substations, gdf_hospitals])
        # Rows 0..N-1  → substations
        # Rows N..N+M-1 → hospitals
    """
    if not gdf_list:
        raise ValueError("gdf_list must contain at least one GeoDataFrame.")
    combined = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
    return combined


def build_voronoi_service_area_map(
    voronoi_gdf: "gpd.GeoDataFrame",
    gdf_secondary_assets: "gpd.GeoDataFrame",
    *,
    return_diagnostics: bool = False,
) -> dict:
    """Map each secondary asset to the primary asset whose Voronoi polygon contains it.

    The mapping is built once before the simulation starts so the O(N·M) spatial
    search is performed only once rather than repeated every timestep.

    **Spatial search strategy** (reuses :func:`create_spatial_index`):

    1. Build an R-tree on the Voronoi polygons using bounding-box entries.
    2. For each secondary asset, query the R-tree with the asset geometry bounds
       to get a small set of candidate Voronoi polygons (bbox filter).
    3. Run an exact geometric intersection check against each candidate (fine
       intersection).  For polygon assets, choose the Voronoi polygon with the
       largest overlap area.  For point assets, use ``contains`` / ``intersects``
       on the anchor point.

    This two-step approach mirrors the pattern already used elsewhere in the codebase
    (e.g. ``assign_impact_metric_to_voronoi``) and keeps the function fast for large
    asset sets.

    Each secondary asset is assigned to **at most one** Voronoi polygon.
    If no geometric match is found (e.g. edge effects after clipping), the nearest
    Voronoi polygon to the asset anchor point is used as fallback so dependency
    mappings are never silently dropped.

    Args:
        voronoi_gdf: GeoDataFrame of Voronoi polygons.  Must have an ``'asset_id'``
            column whose values are the **positional indices** of the primary assets
            in the combined ``gdf_assets`` array (as produced by
            :func:`~src.impacts.create_voronoi_for_asset_type` when called on a
            combined GDF with a RangeIndex).
        gdf_secondary_assets: GeoDataFrame of secondary assets (e.g. hospitals).
            Must be in the same CRS as *voronoi_gdf*, or will be reprojected.
            The DataFrame **index** must hold the positional indices of these assets
            in the combined ``gdf_assets`` array (i.e. call this function on the
            slice ``gdf_assets[gdf_assets['type'] == 'hospital']`` after compiling
            with :func:`compile_asset_gdfs`).

    Returns:
        dict or tuple[dict, dict]:
            ``{primary_pos: [secondary_pos, …]}`` – maps each primary asset's
            positional index to the list of secondary asset positional indices that
            fall within its Voronoi polygon.  Primary assets with no secondary assets
            in their polygon are omitted.

            When ``return_diagnostics=True``, also returns a diagnostics dict with
            resolution counts for overlap, point-in-polygon, nearest fallback,
            and unresolved assets.

    Example::

        gdf_assets = compile_asset_gdfs([gdf_substations, gdf_hospitals])
        voronoi_gdf = create_voronoi_for_asset_type(gdf_assets, 'msls')
        gdf_hospitals_slice = gdf_assets[gdf_assets['type'] == 'hospital']
        gdf_hospitals_proj = gdf_hospitals_slice.to_crs(voronoi_gdf.crs)
        service_area_map = build_voronoi_service_area_map(voronoi_gdf, gdf_hospitals_proj)

    Note:
        This is a standalone spatial-mapping utility, not a configuration
        key consumed by ``run_simulation``/``config.py``. Runtime dependency
        service-area topology is instead generated automatically through the
        current knowledge_graph/dependency-edge expansion path -- a
        ``topology="voronoi"`` rule in ``config['dependency_parameters']
        ['knowledge_graph']`` is expanded by
        :func:`~src.dependency_topology.expand_dependency_edges` into
        concrete runtime edges; no ``service_area_map`` config injection is
        read or required. This helper remains useful for direct/ad hoc
        Voronoi mapping and is also used by the societal-access
        postprocessing path (see
        ``src.societal_access._build_service_area_population_maps``).
    """
    diagnostics = {
        "total_secondary_assets": len(gdf_secondary_assets),
        "resolved_by_overlap": 0,
        "resolved_by_point_in_polygon": 0,
        "resolved_by_nearest": 0,
        "unresolved": 0,
        "unresolved_asset_ids": [],
    }

    if voronoi_gdf.empty or gdf_secondary_assets.empty:
        diagnostics["summary"] = "0 overlap, 0 point-in-polygon, 0 nearest, 0 unresolved"
        return ({}, diagnostics) if return_diagnostics else {}

    # Reproject secondary assets to match the Voronoi CRS if needed.
    if (
        gdf_secondary_assets.crs is not None
        and voronoi_gdf.crs is not None
        and gdf_secondary_assets.crs != voronoi_gdf.crs
    ):
        gdf_secondary_assets = gdf_secondary_assets.to_crs(voronoi_gdf.crs)

    # Build R-tree spatial index on Voronoi polygons.
    # create_spatial_index inserts using the iterrows() index label as the key.
    # voronoi_gdf has a default RangeIndex, so label == iloc position.
    voronoi_sindex = create_spatial_index(voronoi_gdf)

    # Reset Voronoi to a plain dict for O(1) lookups by R-tree label.
    # record schema: label -> (asset_id, geometry)
    voronoi_records = {
        int(idx): (int(asset_id), geom)
        for idx, asset_id, geom in voronoi_gdf.itertuples(index=True, name=None)
    }

    service_area_map: dict = {}

    for sec_pos, sec_row in gdf_secondary_assets.iterrows():
        sec_geom = sec_row.geometry
        if sec_geom is None or sec_geom.is_empty:
            continue

        # Use centroid as anchor for fallback nearest/contains checks.
        point = sec_geom.centroid if hasattr(sec_geom, 'centroid') else sec_geom

        # Step 1 – bounding-box candidates from R-tree.
        candidate_labels = list(voronoi_sindex.intersection(sec_geom.bounds))

        best_label = None
        best_overlap_area = -1.0
        resolution_method = None

        # Step 2a – polygon/geometry overlap scoring.
        for vor_label in candidate_labels:
            record = voronoi_records.get(int(vor_label))
            if record is None:
                continue
            _, vor_geom = record
            if vor_geom is None or vor_geom.is_empty:
                continue
            if not vor_geom.intersects(sec_geom):
                continue
            try:
                overlap_area = vor_geom.intersection(sec_geom).area
            except Exception:
                overlap_area = 0.0
            if overlap_area > best_overlap_area:
                best_overlap_area = overlap_area
                best_label = int(vor_label)
                resolution_method = "overlap"

        # Step 2b – fallback to point-in-polygon if overlap did not resolve a match.
        if best_label is None:
            point_candidate_labels = list(voronoi_sindex.intersection(point.bounds))
            for vor_label in point_candidate_labels:
                record = voronoi_records.get(int(vor_label))
                if record is None:
                    continue
                _, vor_geom = record
                if vor_geom is None or vor_geom.is_empty:
                    continue
                if vor_geom.contains(point) or vor_geom.intersects(point):
                    best_label = int(vor_label)
                    resolution_method = "point_in_polygon"
                    break

        # Step 2c – nearest-neighbour fallback to avoid unmatched assets.
        if best_label is None:
            nearest_label = None
            nearest_distance = float("inf")
            for label, (_, vor_geom) in voronoi_records.items():
                if vor_geom is None or vor_geom.is_empty:
                    continue
                try:
                    dist = vor_geom.distance(point)
                except Exception:
                    continue
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_label = label
            best_label = nearest_label
            if best_label is not None:
                resolution_method = "nearest"

        if best_label is not None and best_label in voronoi_records:
            primary_pos = voronoi_records[best_label][0]
            service_area_map.setdefault(primary_pos, []).append(int(sec_pos))
            if resolution_method == "overlap":
                diagnostics["resolved_by_overlap"] += 1
            elif resolution_method == "point_in_polygon":
                diagnostics["resolved_by_point_in_polygon"] += 1
            elif resolution_method == "nearest":
                diagnostics["resolved_by_nearest"] += 1
        else:
            diagnostics["unresolved"] += 1
            diagnostics["unresolved_asset_ids"].append(int(sec_pos))

    diagnostics["summary"] = (
        f"{diagnostics['resolved_by_overlap']}/{diagnostics['total_secondary_assets']} overlap, "
        f"{diagnostics['resolved_by_point_in_polygon']} point-in-polygon, "
        f"{diagnostics['resolved_by_nearest']} nearest, "
        f"{diagnostics['unresolved']} unresolved"
    )

    if return_diagnostics:
        return service_area_map, diagnostics
    return service_area_map
