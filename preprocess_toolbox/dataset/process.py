import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import iris
import iris.analysis
import iris.exceptions
import numpy as np
import rioxarray
import xarray as xr

from affine import Affine
from download_toolbox.interface import DatasetConfig
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from preprocess_toolbox.cli import parse_shape
from preprocess_toolbox.dataset.spatial import (gridcell_angles_from_dim_coords,
                                                invert_gridcell_angles,
                                                rotate_grid_vectors)


def regrid_dataset(ref_file: os.PathLike,
                   process_config: DatasetConfig,
                   coord_processing: callable = None,
                   coord_processing_args: list = None,
                   regrid_processing: callable = None,
                   regrid_processing_args: list = None,
                   ):
    """

    TODO: we need to incorporate OSISAF / SIC grounc truth cube generation into the IceNet library
     as the native files downloaded just aren't suitable. That doesn't belong in here though!
     Or if it is included it should be as a helper utility

    TODO: regrid_processing needs to come from a module:regrid method in icenet.data.regrid.osisaf, for example
     which needs to be specified from the command line

    :param ref_file:
    :param process_config:
    :param coord_processing:
    :param coord_processing_args:
    :param regrid_processing:
    :param regrid_processing_args:
    """
    logging.info("Regridding dataset")

    # Give me strength with Iris, it's hard to tell what it'll return
    ref_cube = iris.load_cube(ref_file)

    for datafile in [_
                     for var_files in process_config.var_files.values()
                     for _ in var_files]:
        (datafile_path, datafile_name) = os.path.split(datafile)

        regrid_source_name = "_regrid_{}".format(datafile_name)
        regrid_datafile = os.path.join(datafile_path, regrid_source_name)
        os.rename(datafile, regrid_datafile)

        logging.debug("Regridding {}".format(regrid_datafile))

        try:
            cube = iris.load_cube(regrid_datafile)

            # TODO: this assumes a lot, and also should be contained in the icenet library by default
            if coord_processing is None:
                if cube.coord_system() is None:
                    logging.warning("We have not detected a coordinate system and have "
                                    "no method to apply, copying from ref_cube for lat/long")
                    cs = ref_cube.coord_system().ellipsoid

                    for coord in ['longitude', 'latitude']:
                        cube.coord(coord).coord_system = cs
            else:
                logging.info("Providing coordinate system transform method being run: {}".format(coord_processing))
                coord_processing_args = tuple() if coord_processing_args is None else coord_processing_args
                cube = coord_processing(ref_cube, cube, *coord_processing_args)
            cube_regridded = cube.regrid(ref_cube, iris.analysis.Linear())

        except iris.exceptions.CoordinateNotFoundError:
            logging.warning("{} has no coordinates...".format(datafile_name))
            continue

        if regrid_processing is not None:
            logging.debug("Calling regrid processing callable: {}".format(regrid_processing))
            regrid_processing_args = tuple() if regrid_processing_args is None else regrid_processing_args
            cube_regridded = regrid_processing(ref_cube, cube_regridded, *regrid_processing_args)

        logging.debug("Saving regridded data to {}... ".format(datafile))
        iris.save(cube_regridded, datafile, fill_value=np.nan)

        if os.path.exists(datafile):
            os.remove(regrid_datafile)


def rotate_dataset(ref_file: os.PathLike,
                   process_config: DatasetConfig,
                   vars_to_rotate: object = ("uas", "vas")):
    """

    :param ref_file:
    :param process_config:
    :param vars_to_rotate:
    """
    if len(vars_to_rotate) != 2:
        raise RuntimeError("Two variables only should be supplied, you gave {}".format(", ".join(vars_to_rotate)))

    ref_cube = iris.load_cube(ref_file)

    angles = gridcell_angles_from_dim_coords(ref_cube)
    invert_gridcell_angles(angles)

    wind_files = {
        vars_to_rotate[0]: sorted(process_config.var_files[vars_to_rotate[0]]),
        vars_to_rotate[1]: sorted(process_config.var_files[vars_to_rotate[1]]),
    }

    # NOTE: we're relying on apply_to having equal datasets
    assert len(wind_files[vars_to_rotate[0]]) == len(wind_files[vars_to_rotate[1]]), \
        "The wind file datasets are unequal in length"

    # validation
    for idx, wind_file_0 in enumerate(wind_files[vars_to_rotate[0]]):
        wind_file_1 = wind_files[vars_to_rotate[1]][idx]

        wd0 = re.sub(r'^{}_'.format(vars_to_rotate[0]), '',
                     os.path.basename(wind_file_0))

        if not wind_file_1.endswith(wd0):
            logging.error("File array is not valid: {}".format(zip(wind_files)))
            raise RuntimeError("{} is not at the end of {}, something is "
                               "wrong".format(wd0, wind_file_1))

    for idx, wind_file_0 in enumerate(wind_files[vars_to_rotate[0]]):
        wind_file_1 = wind_files[vars_to_rotate[1]][idx]

        logging.info("Rotating {} and {}".format(wind_file_0, wind_file_1))

        wind_cubes = dict()
        wind_cubes_r = dict()

        wind_cubes[vars_to_rotate[0]] = iris.load_cube(wind_file_0)
        wind_cubes[vars_to_rotate[1]] = iris.load_cube(wind_file_1)

        try:
            wind_cubes_r[vars_to_rotate[0]], wind_cubes_r[vars_to_rotate[1]] = \
                rotate_grid_vectors(
                    wind_cubes[vars_to_rotate[0]],
                    wind_cubes[vars_to_rotate[1]],
                    angles,
                )
        except iris.exceptions.CoordinateNotFoundError:
            logging.exception("Failure to rotate due to coordinate issues. "
                              "moving onto next file")
            continue

        # Original implementation is in danger of lost updates
        # due to potential lazy loading
        for i, name in enumerate([wind_file_0, wind_file_1]):
            # NOTE: implementation with temp file caused problems on NFS
            # mounted filesystem, so avoiding in place of letting iris do it
            temp_name = os.path.join(os.path.split(name)[0],
                                     "temp.{}".format(
                                         os.path.basename(name)))
            logging.debug("Writing {}".format(temp_name))

            iris.save(wind_cubes_r[vars_to_rotate[i]], temp_name)
            os.replace(temp_name, name)
            logging.debug("Overwritten {}".format(name))

    # merge_files(new_datafile, moved_datafile, self._drop_vars)


def reproject_dataset(
    netcdf_file,
    source_crs = None,
    target_crs = None,
    resolution = None,
    shape = None,
    target_transform = None,
    coarsen = 1,
    interpolate_nans = False,
) -> xr.Dataset:
    """
    Reprojects a source dataset from a source CRS to a target CRS using rioxarray.

    Args:
        netcdf_file: Path to the NetCDF file or an xarray Dataset/DataArray.
        source_crs (optional): Source coordinate reference system (CRS). Defaults to "EPSG:4326".
        target_crs (optional): Target coordinate reference system (CRS). Defaults to "EPSG:6931".
        resolution (optional): Resolution of the target grid. Defaults to None.
        shape (optional): Shape of the target grid. Defaults to None.
        target_transform (optional): Affine transform for the target grid. Defaults to None.
        coarsen (optional): Factor by which to coarsen the dataset. Defaults to 1.
        interpolate_nans (optional): Whether to interpolate missing values (NaNs). Defaults to False.

    Returns:
        xarray.Dataset: Reprojected dataset.
    """
    if isinstance(netcdf_file, xr.Dataset) or isinstance(netcdf_file, xr.DataArray):
        ds = netcdf_file
    else:
        ds = xr.open_dataset(netcdf_file, decode_coords="all")

    source_crs = source_crs if source_crs else "EPSG:4326"
    target_crs = target_crs if target_crs else "EPSG:6931"

    if not hasattr(ds, "spatial_ref"):
        logging.debug(
            f"No spatial reference found in dataset, setting grid to: {source_crs}"
        )
        # This will add a `.spatial_ref`` attribute to the dataset,
        # accessible via `ds.spatial_ref`.
        # Assume that dataset is a lat/lon grid if `source_crs` is not defined
        ds.rio.write_crs(source_crs, inplace=True)

    if not isinstance(shape, tuple):
        shape: tuple[int, int] = parse_shape(shape)

    ds_reprojected = ds.rio.reproject(
        target_crs,
        resolution=resolution,
        shape=shape,
        # TODO: Missing antimeridian slice issue with Polar reprojection when using
        # other resampling methods (e.g., bilinear, cubic)
        resampling=Resampling.nearest,
        nodata=np.nan,
        transform=target_transform,
    )

    if interpolate_nans:
        # Interpolate missing regions (nans), for CANARI, its below equator, might be
        # useful if training mask doesn't align exactly?
        ds_reprojected = ds_reprojected.rio.interpolate_na("nearest")

    if coarsen > 1:
        ds_reprojected = ds_reprojected.coarsen(
            x=coarsen, y=coarsen, boundary="trim"
        ).mean()

    # TODO: Storing grid mapping attributes in the dataset to be CF Compliant

    return ds_reprojected


def reproject_dataset_ease2(
    *args,
    **kwargs,
) -> xr.Dataset:
    """
    Reprojects a dataset to the EASE-Grid 2.0 standard.

    Args:
        *args: Positional arguments to pass to `reproject_dataset`.
        **kwargs: Keyword arguments to pass to `reproject_dataset`. Must include:
            - target_crs (str): Target CRS, must be "EPSG:6931" or "EPSG:6932".
            - shape (tuple[int, int], optional): Shape of the target grid. Defaults to (720, 720).

    Raises:
        ValueError: If `target_crs` or `shape` does not match the EASE-Grid 2.0 standard.

    Returns:
        Reprojected dataset.
    """
    target_crs = kwargs["target_crs"]
    if target_crs is None:
        raise ValueError("target_crs must be specified")
    shape = kwargs.get("shape", (720, 720))
    kwargs["shape"] = shape

    if kwargs.get("resolution", None):
        raise ValueError(
            f"Resolution cannot be specified for EASE-Grid 2.0: {kwargs['resolution']}"
        )

    if not isinstance(shape, tuple):
        shape: tuple[int, int] = parse_shape(shape)

    if target_crs == "EPSG:6931" or target_crs == "EPSG:6932":
        # Define grid parameters for EASE-Grid 2.0 standard grid
        # Reference: https://nsidc.org/data/user-resources/help-center/guide-ease-grids#anchor-25-km-resolution-ease-grids
        # `cell_size` is the grid resolution in meters taken from the table in above link
        if shape == (720, 720):
            cell_size = 25000
        elif shape == (500, 500):
            cell_size = 36000
        else:
            raise ValueError(
                f"shape doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{shape}`)"
            )
        # The x-axis coordinate of the outer edge of the upper-left pixel
        x0 = -9000000.0
        # The y-axis coordinate of the outer edge of the upper-left pixel
        y0 = 9000000.0
    else:
        raise ValueError(
            f"target_crs doesn't match expected EASE-Grid 2.0 standard grid:\n\t(`{target_crs}`)"
        )

    # Create an affine transform for the target grid.
    # from_origin expects (upper-left x, upper-left y, x resolution, y resolution)
    target_transform = from_origin(x0, y0, cell_size, cell_size)

    ds_reprojected = reproject_dataset(
        *args, target_transform=target_transform, **kwargs
    )

    return ds_reprojected


def reproject_file(datafile: str, ease2: bool = False, **kwargs) -> None:
    """
    Reprojects a single NetCDF file.

    Args:
        datafile (str): Path to the NetCDF file to reproject.
        ease2 (optional): Whether to use EASE-Grid 2.0 standard for reprojection. Defaults to False.
        **kwargs: Additional arguments to pass to `reproject_dataset` or `reproject_dataset_ease2`.

    Raises:
        Exception: If an error occurs during reprojection.
    """
    try:
        (datafile_path, datafile_name) = os.path.split(datafile)
        reproject_source_name = f"_reproject_{datafile_name}"
        reproject_datafile = Path(datafile_path) / reproject_source_name
        os.rename(datafile, reproject_datafile)

        logging.debug(f"Reprojecting {reproject_datafile}")

        if ease2:
            ds_reprojected = reproject_dataset_ease2(
                netcdf_file=reproject_datafile, **kwargs
            )
        else:
            ds_reprojected = reproject_dataset(netcdf_file=reproject_datafile, **kwargs)

        logging.debug(f"Saving reprojected data to {datafile}... ")
        ds_reprojected.to_netcdf(datafile)
    except Exception as e:
        print(f"Error reprojecting {datafile}: {e}")
        raise
    finally:
        # Ensure temp file is deleted
        if os.path.exists(reproject_datafile):
            os.remove(reproject_datafile)


def reproject_datasets_from_config(
    process_config: DatasetConfig, ease2=False, workers: int=1, **kwargs
) -> None:
    """
    Reprojects multiple datasets from input config file.

    Args:
        process_config: Configuration object containing dataset file paths.
        ease2 (optional): Whether to use EASE-Grid 2.0 standard for reprojection. Defaults to False.
        workers (optional): Number of parallel workers to use. Defaults to 1.
        **kwargs: Additional arguments to pass to `reproject_file`.
    """
    logging.info("Reprojecting dataset")

    datafiles = [
        _ for var_files in process_config.var_files.values() for _ in var_files
    ]

    logging.info(f"{len(datafiles)} files to reproject")
    if workers > 1:
        logging.info(f"Reprojecting using {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(reproject_file, datafile, ease2, **kwargs) for datafile in datafiles]

        _ = [future.result() for future in futures]
    else:
        logging.info("Reprojecting using one worker")
        for datafile in datafiles:
            reproject_file(datafile, ease2, **kwargs)

    logging.info("Reprojection completed")
