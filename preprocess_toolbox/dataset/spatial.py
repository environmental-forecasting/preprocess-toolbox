import logging
import os

import cartopy.crs as ccrs
import cf_units
import iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate as interpolate

from download_toolbox.interface import DatasetConfig


def spatial_interpolation(da: xr.DataArray,
                          ds_config: DatasetConfig,
                          mask_processor: object = None,
                          masks: list = None,
                          save_comparison_fig: bool = False) -> object:
    """
    TODO: method inherited from icenet2 draft code from Tom, not sure it's generalisable

    Args:
        da:
        ds_config:
        mask_processor:
        masks:
        save_comparison_fig:

    """

    for date in da.time.values:
        mask = None
        if mask_processor is not None:
            logging.debug("Getting masks {} from {}".format(", ".join(masks), mask_processor))
            for mask_type in masks:
                if mask is None:
                    mask = getattr(mask_processor, mask_type)(date)
                else:
                    add_mask = getattr(mask_processor, mask_type)(date)
                    mask[add_mask] = True

        da_el = da.sel(time=date).copy()
        if len(da_el.shape) > 2:
            raise RuntimeError("Spatial interpolation is only available for 2D data: {} - {} spatial dims found".
                               format(len(da_el.shape), "x".format(da_el.shape)))
        x_len, y_len = da_el.shape
        xx, yy = np.meshgrid(np.arange(x_len), np.arange(y_len))

        # Grid cells outside NaN regions
        valid = ~np.isnan(da_el.data)
        if mask is not None:
            valid = valid | mask

        # Interpolate if there is more than one missing grid cell
        if np.sum(~valid) >= 1:
            date_str = pd.to_datetime(date).strftime(ds_config.frequency.date_format)
            logging.info("Interpolating spatial data for {}".format(date_str))

            # Find grid cell locations surrounding NaN regions for bilinear
            # interpolation
            nan_mask = np.ma.masked_array(np.full((x_len, y_len), 0.))
            nan_mask[~valid] = np.ma.masked

            nan_neighbour_arrs = {}
            # C - horizontal, F - vertical
            for order in 'C', 'F':
                # starts and ends indexes of masked element chunks
                slice_ends = np.ma.clump_masked(nan_mask.ravel(order=order))

                nan_neighbour_idxs = []
                nan_neighbour_idxs.extend([s.start - 1 for s in slice_ends])
                nan_neighbour_idxs.extend([s.stop for s in slice_ends])
                nan_neighbour_idxs = [el for el in nan_neighbour_idxs
                                      if 0 <= el < np.prod((x_len, y_len))]

                nan_neighbour_arr_i = np.array(
                    np.full(shape=(x_len, y_len), fill_value=False),
                    order=order)
                nan_neighbour_arr_i.ravel(order=order)[nan_neighbour_idxs] = True
                nan_neighbour_arrs[order] = nan_neighbour_arr_i

            nan_neighbour_arr = nan_neighbour_arrs['C'] + nan_neighbour_arrs['F']
            # Remove artefacts along edge of the grid
            nan_neighbour_arr[:, 0] = \
                nan_neighbour_arr[0, :] = \
                nan_neighbour_arr[:, -1] = \
                nan_neighbour_arr[-1, :] = False

            if np.sum(nan_neighbour_arr) == 1:
                res = np.where(np.array(nan_neighbour_arr) == True)  # noqa: E712
                logging.warning(
                    "Not enough nans for interpolation, extending {}".format(res))

                x_idx, y_idx = res[0][0], res[1][0]
                nan_neighbour_arr[x_idx - 1:x_idx + 2, y_idx] = True
                nan_neighbour_arr[x_idx, y_idx - 1:y_idx + 2] = True
                logging.debug(
                    np.where(np.array(nan_neighbour_arr) == True))  # noqa: E712

            # Perform bilinear interpolation
            x_valid = xx[nan_neighbour_arr]
            y_valid = yy[nan_neighbour_arr]
            values = da_el.data[nan_neighbour_arr]

            x_interp = xx[~valid]
            y_interp = yy[~valid]

            before = da_el.copy()

            try:
                if len(x_valid) or len(y_valid):
                    interp_vals = interpolate.griddata((x_valid, y_valid),
                                                       values,
                                                       (x_interp, y_interp),
                                                       method='linear')
                    da_el.data[~valid] = interp_vals
                else:
                    logging.warning("No valid values to interpolate with on {}".format(date_str))
            except Exception as e:
                logging.warning("Interpolation failed for {}, assignment will not take place: {}".format(date_str, e))
            else:
                da.loc[dict(time=date)] = da_el

                if save_comparison_fig:
                    plt.rcParams['figure.figsize'] = [13, 4]
                    fig, axs = plt.subplots(ncols=3)
                    plt.tight_layout()
                    before.plot.contourf(ax=axs[0], levels=20)
                    da_el.plot.contourf(ax=axs[1], levels=20)
                    (da_el - before).plot.contour(ax=axs[2], levels=2)
                    output_path = os.path.join(ds_config.path,
                                               "_interpd_data",
                                               "display.{}.png".format(date_str))
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    logging.info("Saving interpolation figure for analysis: {}".format(output_path))
                    fig.savefig(output_path)
                    plt.close()
    return da


def assign_lat_lon_coord_system(cube: object):
    """Assign coordinate system to iris cube to allow regridding.

    :param cube:
    """

    # This originated from era5
    cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(6367470.0)
    cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(6367470.0)

    # NOTE: CMIP6 original assignment, but cs in this case is bring assigned to
    # a module call that doesn't exist, let alone contain this member
    # cs = grid_cube.coord_system().ellipsoid
    for coord in ['longitude', 'latitude']:
        if cube.coord(coord).units != cf_units.Unit("degrees"):
            cube.coord(coord).units = cf_units.Unit("degrees")
    #     cmip6_cube.coord(coord).coord_system = cs

    return cube


def rotate_grid_vectors(u_cube: object, v_cube: object, angles: object):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.rotate_grid_vectors`
    that can rotate multiple masked spatial fields in one go by iterating
    over the horizontal spatial axes in slices

    :param u_cube:
    :param v_cube:
    :param angles:
    :return:

    """
    # lists to hold slices of rotated vectors
    u_r_all = iris.cube.CubeList()
    v_r_all = iris.cube.CubeList()

    # get the X and Y dimension coordinates for each source cube
    u_xy_coords = [
        u_cube.coord(axis='x', dim_coords=True),
        u_cube.coord(axis='y', dim_coords=True)
    ]
    v_xy_coords = [
        v_cube.coord(axis='x', dim_coords=True),
        v_cube.coord(axis='y', dim_coords=True)
    ]

    # iterate over X, Y slices of the source cubes, rotating each in turn
    for u, v in zip(u_cube.slices(u_xy_coords, ordered=False),
                    v_cube.slices(v_xy_coords, ordered=False)):
        u_r, v_r = iris.analysis.cartography.rotate_grid_vectors(u, v, angles)
        u_r_all.append(u_r)
        v_r_all.append(v_r)

    # return the slices, merged back together into a pair of cubes
    return u_r_all.merge_cube(), v_r_all.merge_cube()


def gridcell_angles_from_dim_coords(cube: object):
    """
    Author: Tony Phillips (BAS)

    Wrapper for :func:`~iris.analysis.cartography.gridcell_angles`
    that derives the 2D X and Y lon/lat coordinates from 1D X and Y
    coordinates identifiable as 'x' and 'y' axes

    The provided cube must have a coordinate system so that its
    X and Y coordinate bounds (which are derived if necessary)
    can be converted to lons and lats

    :param cube:
    :return:
    """

    # get the X and Y dimension coordinates for the cube
    x_coord = cube.coord(axis='x', dim_coords=True)
    y_coord = cube.coord(axis='y', dim_coords=True)

    # add bounds if necessary
    if not x_coord.has_bounds():
        x_coord = x_coord.copy()
        x_coord.guess_bounds()
    if not y_coord.has_bounds():
        y_coord = y_coord.copy()
        y_coord.guess_bounds()

    # get the grid cell bounds
    x_bounds = x_coord.bounds
    y_bounds = y_coord.bounds
    nx = x_bounds.shape[0]
    ny = y_bounds.shape[0]

    # make arrays to hold the ordered X and Y bound coordinates
    x = np.zeros((ny, nx, 4))
    y = np.zeros((ny, nx, 4))

    # iterate over the bounds (in order BL, BR, TL, TR), mesh them and
    # put them into the X and Y bound coordinates (in order BL, BR, TR, TL)
    c = [0, 1, 3, 2]
    cind = 0
    for yi in [0, 1]:
        for xi in [0, 1]:
            xy = np.meshgrid(x_bounds[:, xi], y_bounds[:, yi])
            x[:, :, c[cind]] = xy[0]
            y[:, :, c[cind]] = xy[1]
            cind += 1

    # convert the X and Y coordinates to longitudes and latitudes
    source_crs = cube.coord_system().as_cartopy_crs()
    target_crs = ccrs.PlateCarree()
    pts = target_crs.transform_points(source_crs, x.flatten(), y.flatten())
    lons = pts[:, 0].reshape(x.shape)
    lats = pts[:, 1].reshape(x.shape)

    # get the angles
    angles = iris.analysis.cartography.gridcell_angles(lons, lats)

    # add the X and Y dimension coordinates from the cube to the angles cube
    angles.add_dim_coord(y_coord, 0)
    angles.add_dim_coord(x_coord, 1)

    # if the cube's X dimension preceeds its Y dimension
    # transpose the angles to match
    if cube.coord_dims(x_coord)[0] < cube.coord_dims(y_coord)[0]:
        angles.transpose()

    return angles


def invert_gridcell_angles(angles: object):
    """
    Author: Tony Phillips (BAS)

    Negate a cube of gridcell angles in place, transforming
    gridcell_angle_from_true_east <--> true_east_from_gridcell_angle
    :param angles:
    """
    angles.data *= -1

    names = ['true_east_from_gridcell_angle', 'gridcell_angle_from_true_east']
    name = angles.name()
    if name in names:
        angles.rename(names[1 - names.index(name)])

