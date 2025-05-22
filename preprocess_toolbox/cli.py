import argparse
import collections
import datetime as dt
import logging
import re

import pandas as pd

# We do work around and grab things directly from download_toolbox.cli here
from download_toolbox.cli import (date_arg, dates_arg,
                                  csv_arg, csv_of_csv_arg, csv_of_date_args,
                                  int_or_list_arg, BaseArgParser)
from download_toolbox.interface import Frequency


class ProcessingArgParser(BaseArgParser):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument("source", type=str)
        self.add_argument("-p", "--destination-path",
                          help="Folder that any output data collections will be put in",
                          type=str, default="processed_data")

    def add_ref_ds(self):
        self.add_argument("reference", type=str)
        return self

    def add_destination(self, optional: bool = True):
        if optional:
            self.add_argument("destination_id", type=str, nargs="?", default=None)
        else:
            self.add_argument("destination_id", type=str)

        return self

    def add_loader(self):
        self.add_argument("-u",
                          "--update-key",
                          default=None,
                          help="Add update key to processor to avoid overwriting default"
                               "entries in the loader configuration",
                          type=str)
        return self

    def add_concurrency(self):
        self.add_argument("-po",
                          "--parallel-opens",
                          default=False,
                          action="store_true",
                          help="Allow parallel opens and dask implementation")
        return self

    def add_implementation(self):
        self.add_argument("-i",
                          "--implementation",
                          type=str,
                          help="Allow implementation to be specified on command line")
        return self

    def add_splits(self):
        self.add_argument("-ps",
                          "--processing-splits",
                          type=csv_arg,
                          required=False,
                          default=None)
        self.add_argument("-sn",
                          "--split-names",
                          type=csv_arg,
                          required=False,
                          default=None)
        self.add_argument("-ss",
                          "--split_starts",
                          type=csv_of_date_args,
                          required=False,
                          default=[])
        self.add_argument("-se",
                          "--split_ends",
                          type=csv_of_date_args,
                          required=False,
                          default=[])

        self.add_argument("-sh",
                          "--split-head",
                          help="Split dates head, number of time steps to include before",
                          type=int,
                          default=0)
        self.add_argument("-st",
                          "--split-tail",
                          help="Split dates tail, number of time steps to include after",
                          type=int,
                          default=0)
        return self

    def add_vars(self):
        self.add_argument("--abs",
                          help="Comma separated list of absolute vars",
                          type=csv_arg,
                          default=[])
        self.add_argument("--anom",
                          help="Comma separated list of anomoly vars",
                          type=csv_arg,
                          default=[])
        return self

    def add_var_name(self):
        self.add_argument("-n", "--var-names",
                          help="Comma separated list of variable names",
                          type=csv_arg,
                          default=None)
        return self

    def add_trends(self):
        self.add_argument("--trends",
                          help="Comma separated list of abs vars",
                          type=csv_arg,
                          default=[])
        self.add_argument("--trend-lead",
                          help="Time steps in the future for linear trends",
                          type=int_or_list_arg,
                          default=7)
        return self

    def add_reference(self):
        self.add_argument("-r",
                          "--ref",
                          help="Reference loader for normalisations etc",
                          default=None,
                          type=str)
        return self


def process_split_args(args: object,
                       frequency: Frequency) -> dict:
    """

    :param args:
    :param frequency:
    :return:
    """
    if not hasattr(args, "split_names") or args.split_names is None:
        logging.info("No split names in arguments to process...")
        return dict()

    splits = {_: list() for _ in args.split_names}

    for idx, split in enumerate(splits.keys()):
        split_dates = collections.deque()

        for period_start, period_end in zip(args.split_starts[idx], args.split_ends[idx]):
            split_dates += [
                pd.to_datetime(date).date()
                for date in pd.date_range(period_start, period_end, freq=frequency.freq)
            ]
        logging.info("Got {} dates for {}".format(len(split_dates), split))

        splits[split] = sorted(list(split_dates))
    return splits


def parse_shape(value: str) -> tuple:
    """
    Parse a shape argument into a tuple of integers.

    This function takes a string representing shape dimensions.
    If the input is a single value, it is duplicated to form a tuple of
    length two. If multiple values are provided, they are converted into
    a tuple.

    Args:
        value: A string containing one or more integers separated by commas,
            representing shape dimensions.

    Returns:
        A tuple of two integers derived from the input string.

    Examples:
        parse_shape("5") returns (5, 5)
        parse_shape("5,6") returns (5, 6)
    """
    if isinstance(value, int):
        return (value, value)
    else:
        values = value.split(",")

        # If only one value is provided, repeat it to create a tuple of length 2
        if len(values) == 1:
            values.append(values[0])
        values = map(int, values)

    return tuple(values)
