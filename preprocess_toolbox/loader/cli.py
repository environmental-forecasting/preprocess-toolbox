import argparse
import logging
import os

import orjson

from preprocess_toolbox.cli import BaseArgParser
from preprocess_toolbox.utils import get_config_filename, update_config

from download_toolbox.interface import get_dataset_config_implementation, get_implementation


class LoaderArgParser(BaseArgParser):
    """An ArgumentParser specialised to support forecast plot arguments

    The 'allow_*' methods return self to permit method chaining.

    :param suppress_logs:
    """

    def __init__(self,
                 source=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if source:
            self.add_argument("source",
                              help="A complete path to a source configuration",
                              type=str)

        self.add_argument("name",
                          type=str)

    def add_configurations(self):
        self.add_argument("configurations",
                          type=argparse.FileType("r"),
                          nargs="+")
        return self

    def add_sections(self):
        self.add_argument("segments",
                          nargs="+")
        return self


class MetaArgParser(LoaderArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument("ground_truth_dataset")
        self.add_argument("-p", "--destination-path",
                          help="Folder that any output data collections will be put in",
                          type=str, default="processed_data")

    def add_channel(self):
        self.add_argument("channel_name")
        self.add_argument("implementation")
        return self

    def add_property(self):
        self.add_argument("-p", "--property",
                          type=str, default=None)
        return self


def create():
    args = LoaderArgParser().parse_args()

    data = dict(
        identifier=args.name,
        filenames=dict(),
        sources=dict(),
        masks=dict(),
        channels=dict(),
    )
    destination_path = get_config_filename(args)

    if not os.path.exists(destination_path):
        with open(destination_path, "w") as fh:
            fh.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode())
        logging.info("Created a configuration {} to build on".format(destination_path))
    else:
        raise FileExistsError("It's pretty pointless calling init on an existing configuration, "
                              "perhaps delete the file first and go for it")


def copy():
    args = (LoaderArgParser(source=True).
            add_sections().
            parse_args())
    if len(args.segments) < 1:
        raise RuntimeError("No segments supplied ")

    with open(args.source, "r") as fh:
        source_data = orjson.loads(fh.read())

    with open(get_config_filename(args), "r") as fh:
        dest_data = orjson.loads(fh.read())

    for segment in args.segments:
        logging.info("Copying segment {} from {} to {}".format(segment, args.source, args.name))
        dest_data[segment] = source_data[segment]

    logging.info("Outputting {}".format(args.name))
    with open(get_config_filename(args), "w") as fh:
        fh.write(orjson.dumps(dest_data, option=orjson.OPT_INDENT_2).decode())


def add_processed():
    args = (LoaderArgParser().
            add_configurations().
            parse_args())
    cfgs = dict()
    filenames = dict()

    for fh in args.configurations:
        logging.info("Configuration {} being loaded".format(fh.name))
        cfg_data = orjson.loads(fh.read())

        if "data" not in cfg_data:
            raise KeyError("There's no data element in {}, that's not right!".format(fh.name))
        name = ".".join(fh.name.split(".")[1:-1])
        cfgs[name] = cfg_data["data"]
        filenames[name] = fh.name
        fh.close()

    update_config(get_config_filename(args), "filenames", filenames)
    update_config(get_config_filename(args), "sources", cfgs)


def get_channel_info_from_processor(cfg_segment: str):
    args = (MetaArgParser().
            add_channel().
            parse_args())

    proc_impl = get_implementation(args.implementation)
    ds_config = get_dataset_config_implementation(args.ground_truth_dataset)

    if args.config is not None:
        # FIXME: args.config contains the location of the dataset config on render, but
        #   this is not part of this pattern! DS is either ground truth or in derived class,
        #   but this library doesn't care or know of it respectively.
        raise RuntimeError("--config-path is invalid for this CLI endpoint, sorry...")

    processor = proc_impl(ds_config,
                          [args.channel_name,],
                          args.channel_name)
    processor.process()
    update_config(get_config_filename(args),
                  cfg_segment,
                  {args.channel_name: processor.get_config()})


def add_channel():
    get_channel_info_from_processor("channels")


def add_mask():
    get_channel_info_from_processor("masks")
