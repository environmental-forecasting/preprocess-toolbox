from abc import abstractmethod

import logging
import os

import dask.array
import numpy as np
import xarray as xr

from pprint import pformat

from download_toolbox.interface import DataCollection, DatasetConfig


class ProcessingError(RuntimeError):
    pass


class Processor(DataCollection):
    """An abstract base class for data processing classes.

    Provides methods for initialising source data from download-toolbox defined
    configurations, process the data, and saving the processed data to normalised netCDF files.

    TODO: the majority of actual data processing, for the moment, is being isolated in the
     child implementation of NormalisingChannelProcessor whilst I work out what is going
     on regarding the relevant processing and the inheritance hierarchy that support future,
     diverse implementations for alternative data types not based on xarray

    """

    SUFFIXES = ["abs"]

    def __init__(self,
                 dataset_config: DatasetConfig,
                 absolute_vars: list,
                 identifier: str,
                 base_path: os.PathLike = os.path.join(".", "processed"),
                 config_path: os.PathLike = None,
                 dtype: np.typecodes = np.float32,
                 processed_files: dict = None,
                 update_key: str = None,
                 **kwargs) -> None:
        """

        Args:
            dataset_config:
            absolute_vars:
            identifier:
            base_path:
            dtype:
            update_key:
            **kwargs:
        """
        super().__init__(base_path=base_path,
                         config_path=config_path,
                         config_type="processed",
                         identifier=identifier,
                         path_components=[])

        self.config.output_path = "." if config_path is None else config_path

        self._abs_vars = absolute_vars if absolute_vars else []
        self._dtype = dtype

        self._processed_files = dict() if processed_files is None else processed_files

        self._update_key = self.identifier if not update_key else update_key

    def get_data_var_folder(self,
                            var_name: str,
                            append: object = None,
                            missing_error: bool = False) -> os.PathLike:
        """Returns the path for a specific data variable.

        Appends additional folders to the path if specified in the `append` parameter.

        :param var_name: The data variable.
        :param append: Additional folders to append to the path.
        :param missing_error: Flag to specify if missing directories should be treated as an error.

        :return str: The path for the specific data variable.
        """
        if not append:
            append = []

        data_var_path = os.path.join(self.path, *[var_name, *append])

        if not os.path.exists(data_var_path):
            if not missing_error:
                os.makedirs(data_var_path, exist_ok=True)
            else:
                raise OSError("Directory {} is missing and this is "
                              "flagged as an error!".format(data_var_path))

        return data_var_path

    def save_processed_file(self,
                            var_name: str,
                            name: str,
                            data: object,
                            convert: bool = True,
                            overwrite: bool = False) -> str:
        """Save processed data to netCDF file.

        Args:
            var_name: The name of the variable.
            name: The name of the file.
            data: The data to be saved.
            convert: Whether to convert data to the processors data type
            overwrite: Whether to overwrite extant files

        Returns:
            object: The path of the saved netCDF file.

        """
        file_path = os.path.join(self.path, name)
        if overwrite or not os.path.exists(file_path):
            logging.debug("Writing to {}".format(file_path))
            if convert:
                data = data.astype(self._dtype)
            data.to_netcdf(file_path)

        if var_name not in self.processed_files.keys():
            self.processed_files[var_name] = list()

        if file_path not in self.processed_files[var_name]:
            logging.debug("Adding {} file: {}".format(var_name, file_path))
            self.processed_files[var_name].append(file_path)
        # else:
        #     logging.warning("{} already exists in {} processed list".format(file_path, var_name))
        return file_path

    def get_dataset(self,
                    var_names: list = None):
        logging.debug("Finding files for {}".format(", ".join(var_names if var_names is not None else "everything")))

        var_files = [var_filepaths
                     for vn in var_names
                     for var_filepaths in self.processed_files[vn]] \
            if var_names is not None else \
                    [var_filepaths
                     for vn in self.processed_files.keys()
                     for var_filepaths in self.processed_files[vn].values()]

        logging.info("Got {} filenames to open dataset with!".format(len(var_files)))
        logging.debug(pformat(var_files))

        # TODO: where's my parallel mfdataset please!?
        with (dask.config.set(**{'array.slicing.split_large_chunks': True})):
            ds = xr.open_mfdataset(
                var_files,
                combine="nested",
                concat_dim="time",
                coords="minimal",
                compat="override",
                chunks=dict(time=1, ),
            )
            ds = ds.astype(self._dtype)
        return ds

    @abstractmethod
    def process(self):
        pass

    @property
    def abs_vars(self):
        return self._abs_vars

    @property
    def dtype(self):
        return self._dtype

    @property
    def processed_files(self) -> dict:
        """A dict with the processed files organised by variable name."""
        return self._processed_files

    @property
    def update_key(self):
        return self._update_key
