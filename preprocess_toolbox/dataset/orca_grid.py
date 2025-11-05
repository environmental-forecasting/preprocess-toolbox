"""
Support for NEMO ORCA grid regridding to regular lat/lon grids.

ORCA grids (like ORCA025 used in ORAS5) are tri-polar grids with 2D latitude/longitude
coordinates that don't map to standard EPSG codes.
"""
import logging
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

# Cache for grid transformations to avoid recomputing
_GRID_TRANSFORM_CACHE = {}


def regrid_orca_to_latlon(
    orca_data: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    method: str = 'nearest'
) -> xr.DataArray:
    """
    Regrid ORCA grid data to a regular lat/lon grid using memory-efficient chunked processing.
    
    Args:
        orca_data: xarray DataArray with ORCA grid data
        target_lats: 1D array of target latitudes
        target_lons: 1D array of target longitudes
        method: Interpolation method ('nearest' recommended for large grids, 'linear' requires huge memory)
        
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
    
    # Flatten source coordinates and data
    source_lons_flat = source_lons.ravel()
    source_lats_flat = source_lats.ravel()
    source_values = orca_data.values.ravel()
    
    # Remove NaN values from source to reduce memory
    valid_mask = ~np.isnan(source_values)
    source_lons_valid = source_lons_flat[valid_mask]
    source_lats_valid = source_lats_flat[valid_mask]
    source_values_valid = source_values[valid_mask]
    
    logging.info(f"Regridding ORCA data: {len(source_values_valid)} valid source points to {len(target_lats)}x{len(target_lons)} target grid")
    
    # Process in chunks to reduce memory usage
    chunk_size = 50  # Process 50 latitudes at a time
    n_chunks = int(np.ceil(len(target_lats) / chunk_size))
    
    regridded_grid = np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)
    
    # Pre-build source points array once (memory efficient)
    source_points = np.column_stack([source_lons_valid, source_lats_valid])
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(target_lats))
        chunk_n_lats = end_idx - start_idx
        
        if i % 10 == 0:
            logging.info(f"Processing chunk {i+1}/{n_chunks} (rows {start_idx}-{end_idx})")
        
        # Create meshgrid only for this chunk of latitudes
        chunk_lats = target_lats[start_idx:end_idx]
        chunk_lon_grid, chunk_lat_grid = np.meshgrid(target_lons, chunk_lats)
        
        # Flatten to get target points for this chunk
        chunk_target_points = np.column_stack([
            chunk_lon_grid.ravel(),
            chunk_lat_grid.ravel()
        ])
        
        # Interpolate for this chunk
        chunk_values = griddata(
            source_points,
            source_values_valid,
            chunk_target_points,
            method=method,
            fill_value=np.nan
        )
        
        # Reshape and assign
        regridded_grid[start_idx:end_idx, :] = chunk_values.reshape(chunk_n_lats, len(target_lons))
    
    logging.info(f"ORCA regridding complete")
    
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
    
    This function prepares ORCA data for regridding by converting it to xarray,
    regridding to the reference grid, and converting back to iris cube.
    
    Args:
        ref_cube: Reference iris cube with target grid
        orca_cube: ORCA grid iris cube to regrid
        
    Returns:
        Regridded iris cube on the same grid as ref_cube
    """
    import iris
    from iris.coords import DimCoord
    
    logging.info("Processing ORCA grid coordinates")
    
    # Get target lat/lon from reference cube
    target_lats = ref_cube.coord('latitude').points
    target_lons = ref_cube.coord('longitude').points
    
    # Convert ORCA cube to xarray for easier handling
    # Handle potential time dimension
    if orca_cube.ndim == 3:  # time, y, x
        regridded_slices = []
        n_times = orca_cube.shape[0]
        logging.info(f"Processing {n_times} time steps")
        
        for time_idx in range(n_times):
            logging.info(f"Processing time step {time_idx + 1}/{n_times}")
            orca_slice = orca_cube[time_idx]
            orca_data = xr.DataArray(
                orca_slice.data,
                coords={
                    'nav_lat': (['y', 'x'], orca_slice.coord('latitude').points),
                    'nav_lon': (['y', 'x'], orca_slice.coord('longitude').points)
                },
                dims=['y', 'x']
            )
            regridded_slice = regrid_orca_to_latlon(orca_data, target_lats, target_lons)
            regridded_slices.append(regridded_slice.values)
        
        regridded_data = np.stack(regridded_slices)
        
        # Create new iris cube with regridded data
        lat_coord = DimCoord(target_lats, standard_name='latitude', units='degrees')
        lon_coord = DimCoord(target_lons, standard_name='longitude', units='degrees')
        time_coord = orca_cube.coord('time')
        
        regridded_cube = iris.cube.Cube(
            regridded_data,
            dim_coords_and_dims=[
                (time_coord, 0),
                (lat_coord, 1),
                (lon_coord, 2)
            ]
        )
    else:  # 2D: y, x
        orca_data = xr.DataArray(
            orca_cube.data,
            coords={
                'nav_lat': (['y', 'x'], orca_cube.coord('latitude').points),
                'nav_lon': (['y', 'x'], orca_cube.coord('longitude').points)
            },
            dims=['y', 'x']
        )
        regridded_data = regrid_orca_to_latlon(orca_data, target_lats, target_lons)
        
        lat_coord = DimCoord(target_lats, standard_name='latitude', units='degrees')
        lon_coord = DimCoord(target_lons, standard_name='longitude', units='degrees')
        
        regridded_cube = iris.cube.Cube(
            regridded_data.values,
            dim_coords_and_dims=[
                (lat_coord, 0),
                (lon_coord, 1)
            ]
        )
    
    # Copy metadata from original cube
    regridded_cube.standard_name = orca_cube.standard_name
    regridded_cube.long_name = orca_cube.long_name
    regridded_cube.var_name = orca_cube.var_name
    regridded_cube.units = orca_cube.units
    
    return regridded_cube
