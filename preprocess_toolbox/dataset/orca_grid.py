"""
Support for NEMO ORCA grid regridding to regular lat/lon grids.

ORCA grids (like ORCA025 used in ORAS5) are tri-polar grids with 2D latitude/longitude
coordinates that don't map to standard EPSG codes.
"""
import logging
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import pickle
from pathlib import Path
import gc

# Cache for grid transformations to avoid recomputing
_GRID_TRANSFORM_CACHE = {}
_CACHE_DIR = Path.home() / '.cache' / 'preprocess_toolbox' / 'orca_grids'


def _get_or_build_transform(source_lats, source_lons, target_lats, target_lons, cache_key):
    """
    Get cached transform or build new one using KDTree for nearest neighbor mapping.
    
    Checks in-memory cache first, then disk cache, then builds new transform.
    Saves to both in-memory and disk cache for future use.
    
    Returns:
        tuple: (indices of nearest source points, valid source mask, target shape)
    """
    # Check in-memory cache first
    if cache_key in _GRID_TRANSFORM_CACHE:
        logging.info(f"Using in-memory cached grid transform")
        return _GRID_TRANSFORM_CACHE[cache_key]
    
    # Check disk cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        logging.info(f"Loading grid transform from disk cache: {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                transform = pickle.load(f)
            _GRID_TRANSFORM_CACHE[cache_key] = transform
            logging.info(f"Successfully loaded cached transform")
            return transform
        except Exception as e:
            logging.warning(f"Failed to load cached transform: {e}, will rebuild")
    
    logging.info(f"Building new grid transform (will be cached for reuse)")
    
    # Flatten source coordinates
    source_lons_flat = source_lons.ravel()
    source_lats_flat = source_lats.ravel()
    
    # Create source points array (only valid, non-NaN points)
    valid_mask = ~(np.isnan(source_lons_flat) | np.isnan(source_lats_flat))
    valid_indices = np.where(valid_mask)[0]
    
    source_points = np.column_stack([
        source_lons_flat[valid_mask],
        source_lats_flat[valid_mask]
    ])
    
    logging.info(f"Source grid: {len(source_points)} valid points")
    logging.info(f"Target grid: {len(target_lats)} x {len(target_lons)} = {len(target_lats) * len(target_lons)} points")
    
    # Build KDTree for fast nearest neighbor lookup
    logging.info(f"Building KDTree with {len(source_points)} valid source points...")
    tree = cKDTree(source_points)
    logging.info(f"KDTree construction complete")
    
    # Check if target coordinates are already point arrays (same length) vs separate 1D lat/lon arrays
    if len(target_lats) == len(target_lons):
        # Coordinates are already flattened point pairs (e.g., from a 2D grid)
        # Don't create a meshgrid - use them directly
        logging.info(f"Target coordinates are pre-flattened point pairs ({len(target_lats)} points)")
        target_points = np.column_stack([target_lons, target_lats])
        n_target_total = len(target_lats)
        
        # Query all points (no chunking needed for point-pair input)
        logging.info(f"Querying KDTree with {n_target_total} target points...")
        _, nearest_indices = tree.query(target_points, k=1, workers=-1)
        
        # Ensure 1D
        if nearest_indices.ndim > 1:
            nearest_indices = nearest_indices.ravel()
        
        # Map to original valid indices
        source_indices = valid_indices[nearest_indices]
        
        # Infer target shape - will be overridden by caller for 2D grids
        target_shape = (int(np.sqrt(n_target_total)), int(np.sqrt(n_target_total)))
        logging.info(f"KDTree query complete")
        
    else:
        # Separate 1D lat/lon arrays - create meshgrid in chunks
        # Query in chunks to avoid creating full meshgrid in memory
        chunk_size = 200  # Increased from 50 for better performance (use more if you have RAM)
        n_lat_chunks = (len(target_lats) - 1) // chunk_size + 1
        n_target_total = len(target_lats) * len(target_lons)
        source_indices = np.zeros(n_target_total, dtype=np.int32)
        
        logging.info(f"Querying KDTree in {n_lat_chunks} latitude chunks of {chunk_size} rows")
        
        idx = 0
        for lat_chunk_idx in range(0, len(target_lats), chunk_size):
            lat_end = min(lat_chunk_idx + chunk_size, len(target_lats))
            chunk_lats = target_lats[lat_chunk_idx:lat_end]
            chunk_num = lat_chunk_idx // chunk_size + 1
            
            # Create mini-meshgrid just for this chunk
            chunk_lon_grid, chunk_lat_grid = np.meshgrid(target_lons, chunk_lats)
            
            # Debug: check shapes
            logging.debug(f"chunk_lats shape: {chunk_lats.shape}, target_lons shape: {target_lons.shape}")
            logging.debug(f"chunk_lon_grid shape: {chunk_lon_grid.shape}, chunk_lat_grid shape: {chunk_lat_grid.shape}")
            
            chunk_points = np.column_stack([
                chunk_lon_grid.ravel(),
                chunk_lat_grid.ravel()
            ])
            
            logging.debug(f"chunk_points shape: {chunk_points.shape}")
            logging.info(f"Processing latitude chunk {chunk_num}/{n_lat_chunks} ({len(chunk_points)} points)")
            
            # Query this chunk - use all available cores for parallel processing
            _, nearest_chunk = tree.query(chunk_points, k=1, workers=-1)
            
            # Ensure nearest_chunk is 1D
            if nearest_chunk.ndim > 1:
                nearest_chunk = nearest_chunk.ravel()
            
            # Sanity check
            if len(nearest_chunk) != len(chunk_points):
                logging.error(f"Size mismatch: nearest_chunk={len(nearest_chunk)}, chunk_points={len(chunk_points)}")
                raise ValueError(f"KDTree query returned wrong size: {len(nearest_chunk)} != {len(chunk_points)}")
            
            # Store indices
            chunk_size_actual = len(chunk_points)
            source_indices[idx:idx+chunk_size_actual] = valid_indices[nearest_chunk]
            idx += chunk_size_actual
            
            # Free chunk memory
            del chunk_lon_grid, chunk_lat_grid, chunk_points, nearest_chunk
            
            if chunk_num % 5 == 0:
                logging.info(f"Progress: {chunk_num}/{n_lat_chunks} chunks complete ({100*chunk_num/n_lat_chunks:.1f}%)")
        
        logging.info(f"All {n_lat_chunks} chunks processed successfully")
        
        # Target shape for meshgrid case
        target_shape = (len(target_lats), len(target_lons))
    
    # Clean up to free memory before caching
    del tree, source_points
    gc.collect()
    
    # Create transform tuple (target_shape was set in the conditional logic above)
    transform = (source_indices, valid_mask, target_shape)
    
    # Cache in memory
    _GRID_TRANSFORM_CACHE[cache_key] = transform
    
    # Save to disk
    try:
        logging.info(f"Saving grid transform to disk cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(transform, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"Grid transform cached successfully")
    except Exception as e:
        logging.warning(f"Failed to save transform to disk: {e}")
    
    return transform


def regrid_orca_to_latlon(
    orca_data: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    cache_key: str = None
) -> xr.DataArray:
    """
    Regrid ORCA grid data to a regular lat/lon grid using cached nearest-neighbor mapping.
    
    Args:
        orca_data: xarray DataArray with ORCA grid data
        target_lats: 1D array of target latitudes
        target_lons: 1D array of target longitudes
        cache_key: Unique key for caching the transform (e.g., "oras5_south")
        
    Returns:
        Regridded data on regular lat/lon grid
    """
    # Get 2D lat/lon from ORCA data
    if 'nav_lat' in orca_data.coords and 'nav_lon' in orca_data.coords:
        source_lats = orca_data.coords['nav_lat'].values
        source_lons = orca_data.coords['nav_lon'].values
    elif 'latitude' in orca_data.coords and 'longitude' in orca_data.coords:
        source_lats = orca_data.coords['latitude'].values
        source_lons = orca_data.coords['longitude'].values
    else:
        raise ValueError("Cannot find latitude/longitude coordinates in ORCA data")
    
    # Get or build the transform
    if cache_key is None:
        cache_key = f"orca_{source_lats.shape}_{target_lats.shape}"
    
    source_indices, valid_mask, target_shape = _get_or_build_transform(
        source_lats, source_lons, target_lats, target_lons, cache_key
    )
    
    # Apply transform to data
    source_values = orca_data.values.ravel()
    regridded_values = np.full(len(source_indices), np.nan, dtype=np.float32)
    
    # Only copy values where source data is valid
    valid_data_mask = ~np.isnan(source_values[source_indices])
    regridded_values[valid_data_mask] = source_values[source_indices[valid_data_mask]]
    
    # Reshape to target grid
    regridded_grid = regridded_values.reshape(target_shape)
    
    # Create output DataArray
    regridded_data = xr.DataArray(
        regridded_grid,
        coords={
            'latitude': target_lats,
            'longitude': target_lons
        },
        dims=['latitude', 'longitude']
    )
    
    return regridded_data


def orca_coord_processing(ref_cube, orca_cube):
    """
    Coordinate processing function for ORCA grids to use with regrid_dataset.
    
    This function performs the full regridding of ORCA data and returns a cube
    that's already on the target grid with matching coordinates. The subsequent
    cube.regrid() call in process.py will recognize the cube is already on the
    target grid and pass it through unchanged.
    
    Args:
        ref_cube: Reference iris cube with target grid
        orca_cube: ORCA grid iris cube to regrid
        
    Returns:
        Regridded iris cube on the same grid as ref_cube
    """
    import iris
    from iris.coords import DimCoord
    
    logging.info("Processing ORCA grid - performing custom nearest-neighbor regridding")
    
    # Get target lat/lon from reference cube
    target_lats = ref_cube.coord('latitude').points
    target_lons = ref_cube.coord('longitude').points
    
    # Handle 2D coordinate grids (e.g., EASE grid projections)
    # For 2D grids, we can't extract simple 1D lat/lon arrays since each cell has unique coords
    # Instead, we'll work directly with the 2D grid structure
    if target_lats.ndim > 1:
        logging.info(f"Target grid is 2D ({target_lats.shape}) - using full 2D coordinates for regridding")
        # For 2D grids, we'll pass the full arrays and let the regridding handle it
        # The grid shape tells us the output dimensions
        target_grid_shape = target_lats.shape
    else:
        # For 1D coordinate arrays, this is a regular lat/lon grid
        target_grid_shape = (len(target_lats), len(target_lons))
    
    logging.info(f"Target grid: {target_grid_shape[0]} x {target_grid_shape[1]} cells")
    
    # Create a cache key based on grid dimensions
    cache_key = f"orca_{orca_cube.shape}_{ref_cube.shape}"
    
    # Convert ORCA cube to xarray for easier handling
    # Handle potential time dimension
    if orca_cube.ndim == 3:  # time, y, x
        n_times = orca_cube.shape[0]
        logging.info(f"Processing {n_times} time steps with vectorized regridding")
        
        # Get the transform once (will be cached after first call)
        # Use first time slice to build/retrieve the transform
        orca_slice_0 = orca_cube[0]
        
        # Get 2D lat/lon coordinates from ORCA data (same for all time steps)
        if hasattr(orca_slice_0.coord('latitude'), 'points'):
            source_lats = orca_slice_0.coord('latitude').points
            source_lons = orca_slice_0.coord('longitude').points
        else:
            raise ValueError("Cannot find latitude/longitude coordinates in ORCA data")
        
        # Get or build the transform ONCE
        cache_key = f"orca_{orca_cube.shape}_{ref_cube.shape}"
        
        # For 2D target grids, we need to flatten and pass as arrays for KDTree
        if target_lats.ndim > 1:
            # 2D grid (e.g., EASE projection) - flatten to create point arrays
            target_lats_for_transform = target_lats.ravel()
            target_lons_for_transform = target_lons.ravel()
            # The output shape comes from the original 2D grid
            expected_target_shape = target_grid_shape
        else:
            # 1D coordinate arrays (regular lat/lon grid)
            target_lats_for_transform = target_lats
            target_lons_for_transform = target_lons  
            expected_target_shape = target_grid_shape
        
        source_indices, valid_mask, returned_target_shape = _get_or_build_transform(
            source_lats, source_lons, target_lats_for_transform, target_lons_for_transform, cache_key
        )
        
        # Override the returned shape if we have a 2D target grid
        # (the transform function creates a meshgrid shape, but we want the actual 2D grid shape)
        if target_lats.ndim > 1:
            target_shape = expected_target_shape
        else:
            target_shape = returned_target_shape
        
        logging.info(f"Applying cached transform to all {n_times} time steps in vectorized operation")
        
        # Extract all time slices as a 3D array: (time, y, x)
        orca_3d = orca_cube.data  # Shape: (n_times, y, x)
        
        # Reshape to (time, flattened_space)
        orca_flat = orca_3d.reshape(n_times, -1)  # Shape: (n_times, y*x)
        
        # Apply the transform to all time steps at once using fancy indexing
        # source_indices maps each target point to its source point
        # Broadcasting applies the same index pattern to all time steps simultaneously
        regridded_flat = orca_flat[:, source_indices]  # Shape: (n_times, n_target_points)
        
        logging.info(f"Applied transform to all {n_times} time steps in single vectorized operation")
        
        # Reshape to (time, target_lat, target_lon)
        regridded_data = regridded_flat.reshape(n_times, target_shape[0], target_shape[1])
        
        logging.info(f"Vectorized regridding complete: {regridded_data.shape}")
        
        # Create new iris cube with regridded data
        # Handle both 1D (regular lat/lon) and 2D (projected) grids
        if target_lats.ndim == 1:
            # Regular lat/lon grid - create 1D DimCoords
            lat_coord = DimCoord(target_lats, standard_name='latitude', units='degrees')
            lon_coord = DimCoord(target_lons, standard_name='longitude', units='degrees')
            
            # Apply the ellipsoid (not the full projected coordinate system) for compatibility
            if ref_cube.coord_system() is not None:
                cs_ellipsoid = ref_cube.coord_system().ellipsoid
                lat_coord.coord_system = cs_ellipsoid
                lon_coord.coord_system = cs_ellipsoid
        else:
            # 2D projected grid (e.g., EASE) - copy auxiliary coordinates from reference cube
            # This preserves the projection information and coordinate structure
            logging.info("Using 2D auxiliary coordinates from reference cube (projected grid)")
            lat_coord = ref_cube.coord('latitude').copy()
            lon_coord = ref_cube.coord('longitude').copy()
        
        # Get time coordinate and shift to end-of-month
        time_coord = orca_cube.coord('time').copy()
        
        # Shift mid-month dates to end-of-month for consistency with other datasets
        import pandas as pd
        import cftime
        from datetime import datetime
        
        time_points = time_coord.units.num2date(time_coord.points)
        
        # Handle cftime objects by converting to standard datetime
        if isinstance(time_points[0], cftime.datetime):
            # Convert cftime to standard datetime (works for most calendar types)
            time_points = [dt.replace(tzinfo=None).to_pydatetime() if hasattr(dt, 'to_pydatetime') 
                          else pd.Timestamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second).to_pydatetime()
                          for dt in time_points]
        
        time_points_pd = pd.to_datetime(time_points)
        end_of_month = time_points_pd + pd.offsets.MonthEnd(0)
        
        # Normalize to midnight (00:00:00) to ensure consistency
        end_of_month_normalized = [datetime(dt.year, dt.month, dt.day, 0, 0, 0) for dt in end_of_month]
        
        # Convert back to numeric values using the same units
        new_time_points = time_coord.units.date2num(end_of_month_normalized)
        time_coord.points = new_time_points
        
        logging.info(f"Shifted ORCA time coordinates from mid-month to end-of-month and normalized to midnight")
        
        # Create iris cube with appropriate coordinate structure
        if target_lats.ndim == 1:
            # 1D coordinates - use as dimension coordinates
            regridded_cube = iris.cube.Cube(
                regridded_data,
                dim_coords_and_dims=[
                    (time_coord, 0),
                    (lat_coord, 1),
                    (lon_coord, 2)
                ]
            )
        else:
            # 2D coordinates - use as auxiliary coordinates with simple dimension coordinates
            from iris.coords import DimCoord as DC
            # Create simple dimension coordinates for the spatial dimensions
            yc_coord = DC(np.arange(target_shape[0]), long_name='yc', units='1')
            xc_coord = DC(np.arange(target_shape[1]), long_name='xc', units='1')
            
            regridded_cube = iris.cube.Cube(
                regridded_data,
                dim_coords_and_dims=[
                    (time_coord, 0),
                    (yc_coord, 1),
                    (xc_coord, 2)
                ],
                aux_coords_and_dims=[
                    (lat_coord, (1, 2)),
                    (lon_coord, (1, 2))
                ]
            )
            
            # Also copy the coordinate system from reference cube
            if ref_cube.coord_system() is not None:
                regridded_cube.add_aux_coord(ref_cube.coord_system())
    else:  # 2D: y, x
        orca_data = xr.DataArray(
            orca_cube.data,
            coords={
                'nav_lat': (['y', 'x'], orca_cube.coord('latitude').points),
                'nav_lon': (['y', 'x'], orca_cube.coord('longitude').points)
            },
            dims=['y', 'x']
        )
        regridded_data = regrid_orca_to_latlon(orca_data, target_lats, target_lons, cache_key=cache_key)
        
        # Handle both 1D (regular lat/lon) and 2D (projected) grids
        if target_lats.ndim == 1:
            # Regular lat/lon grid - create 1D DimCoords
            lat_coord = DimCoord(target_lats, standard_name='latitude', units='degrees')
            lon_coord = DimCoord(target_lons, standard_name='longitude', units='degrees')
            
            # Apply the ellipsoid for compatibility
            if ref_cube.coord_system() is not None:
                cs_ellipsoid = ref_cube.coord_system().ellipsoid
                lat_coord.coord_system = cs_ellipsoid
                lon_coord.coord_system = cs_ellipsoid
            
            regridded_cube = iris.cube.Cube(
                regridded_data.values,
                dim_coords_and_dims=[
                    (lat_coord, 0),
                    (lon_coord, 1)
                ]
            )
        else:
            # 2D projected grid - copy auxiliary coordinates from reference
            logging.info("Using 2D auxiliary coordinates from reference cube (projected grid)")
            lat_coord = ref_cube.coord('latitude').copy()
            lon_coord = ref_cube.coord('longitude').copy()
            
            # Create simple dimension coordinates
            yc_coord = DimCoord(np.arange(target_grid_shape[0]), long_name='yc', units='1')
            xc_coord = DimCoord(np.arange(target_grid_shape[1]), long_name='xc', units='1')
            
            regridded_cube = iris.cube.Cube(
                regridded_data.values,
                dim_coords_and_dims=[
                    (yc_coord, 0),
                    (xc_coord, 1)
                ],
                aux_coords_and_dims=[
                    (lat_coord, (0, 1)),
                    (lon_coord, (0, 1))
                ]
            )
            
            # Copy coordinate system
            if ref_cube.coord_system() is not None:
                regridded_cube.add_aux_coord(ref_cube.coord_system())
    
    # Copy metadata from original cube
    regridded_cube.standard_name = orca_cube.standard_name
    regridded_cube.long_name = orca_cube.long_name
    regridded_cube.var_name = orca_cube.var_name
    regridded_cube.units = orca_cube.units
    
    return regridded_cube
