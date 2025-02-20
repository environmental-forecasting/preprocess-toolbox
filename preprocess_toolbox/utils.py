import argparse
import logging
import operator
import os

from dateutil.relativedelta import relativedelta

import orjson

from download_toolbox.interface import DatasetConfig


def get_config(config_path: os.PathLike):
    with open(config_path, "r") as fh:
        logging.info("Configuration {} being loaded".format(fh.name))
        cfg_data = orjson.loads(fh.read())
    return cfg_data


def get_config_filename(args: argparse.Namespace, prefix: str = "loader"):
    default_loader_config = f"{args.name}.json"

    if prefix is not None:
        default_loader_config = "{}.{}".format(prefix, default_loader_config)

    # TODO: this is a bit grim, but to allow different config output paths it's very flexible. refactor
    if args.config is not None and (os.path.isfile(args.config) or not os.path.exists(args.config)):
        logging.warning("{} has been specified, overriding default name {}".format(args.config, args.name))

    return default_loader_config if args.config is None \
        else os.path.join(args.config, default_loader_config) if os.path.isdir(args.config) \
        else args.config


def get_extension_dates(ds_config: DatasetConfig,
                        dates: list,
                        num_steps: int,
                        reverse=False):
    additional_dates, dropped_dates = [], []

    for date in dates:
        for time in range(num_steps):
            attrs = {"{}s".format(ds_config.frequency.attribute): time + 1}
            op = operator.sub if reverse else operator.add
            extended_date = op(date, relativedelta(**attrs))

            if extended_date not in dates:
                if all([os.path.exists(ds_config.var_filepath(var_config, [extended_date]))
                        for var_config in ds_config.variables]):
                    # We only add these dates into the mix if all necessary files exist
                    additional_dates.append(extended_date)
                else:
                    # Otherwise, warn that the lag data means this is being dropped
                    logging.warning("{} will be dropped due to missing data {}".
                                    format(date, extended_date))
                    dropped_dates.append(date)

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
