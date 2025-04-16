import logging
import os
import re

import iris
import iris.analysis
import iris.exceptions
import numpy as np

from download_toolbox.interface import DatasetConfig
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
