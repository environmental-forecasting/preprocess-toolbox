import argparse
import logging
import operator
import os

from dateutil.relativedelta import relativedelta

import orjson
import pandas as pd
import xarray as xr

from download_toolbox.interface import DatasetConfig, Frequency


def get_config(config_path: os.PathLike):
    with open(config_path, "r") as fh:
        logging.info("Configuration {} being loaded".format(fh.name))
        cfg_data = orjson.loads(fh.read())
    return cfg_data


def get_config_filename(args: argparse.Namespace, prefix: str = "loader"):
    default_loader_config = f"{args.name}.json"

    if prefix is not None:
        default_loader_config = "{}.{}".format(prefix, default_loader_config)

    if (
        "loader_path" in args
        and args.loader_path is not None
        and (os.path.isfile(args.loader_path) or not os.path.exists(args.loader_path))
    ):
        return args.loader_path

    # TODO: this is a bit grim, but to allow different config output paths it's very flexible. refactor
    if args.config is not None and (os.path.isfile(args.config) or not os.path.exists(args.config)):
        logging.warning("{} has been specified, overriding default name {}".format(args.config, args.name))

    return default_loader_config if args.config is None \
        else os.path.join(args.config, default_loader_config) if os.path.isdir(args.config) \
        else args.config


def get_extension_dates(ds_config: DatasetConfig,
                        dates: list,
                        num_steps: int,
                        start_step: int = 0,
                        reverse: bool = False):
    additional_dates, dropped_dates = [], []

    for date in dates:
        for time in range(start_step, num_steps):
            attrs = {"{}s".format(ds_config.frequency.attribute): time}
            op = operator.sub if reverse else operator.add
            extended_date = op(date, relativedelta(**attrs))

            if ds_config.frequency == Frequency.MONTH:
                extended_date = pd.to_datetime(extended_date + pd.offsets.MonthEnd(0)).date()

            # Check we don't know we have data, and also ignore previous occurrences
            if extended_date not in dates and extended_date not in additional_dates:
                extended_date_var_files = [ds_config.var_filepath(var_config, [extended_date])
                                           for var_config in ds_config.variables]
                if all([os.path.exists(df) for df in extended_date_var_files]):
                    # The above will catch those items that fall outside the file output boundary, but not missing
                    # dates within ALL files. This next clause is more expensive, but necessary to catch everything!
                    logging.debug("Files exist, double checking whether {} appears in data itself across {} files".
                                  format(extended_date, len(extended_date_var_files)))

                    # TODO: this won't catch partially available dates where not all files have the date, but some do
                    if pd.Timestamp(extended_date) in xr.open_mfdataset(extended_date_var_files, compat="no_conflicts").time.values:
                        # We only add these dates into the mix if all necessary files exist
                        additional_dates.append(extended_date)
                    else:
                        logging.warning("Nope, {} not in data itself so dropping {}".format(extended_date, date))
                        dropped_dates.append(date)
                        break
                else:
                    # Otherwise, warn that the lag data means this is being dropped
                    logging.warning("{} will be dropped due to missing data {}".
                                    format(date, extended_date))
                    dropped_dates.append(date)
                    break

    return sorted(list(set(additional_dates))), sorted(list(set(dropped_dates)))


def update_config(loader_config: os.PathLike,
                  segment: str,
                  configuration: dict):
    cfg_data = get_config(loader_config)

    if segment not in cfg_data:
        cfg_data[segment] = dict()
    cfg_data[segment].update(configuration)

    with open(loader_config, "w") as fh:
        logging.info("Writing over {}".format(fh.name))
        fh.write(orjson.dumps(cfg_data, option=orjson.OPT_INDENT_2).decode())
