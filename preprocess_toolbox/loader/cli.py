import argparse
import logging
import os

import orjson

import preprocess_toolbox.utils
from preprocess_toolbox.cli import BaseArgParser, csv_arg
from preprocess_toolbox.loader.utils import update_config
from preprocess_toolbox.utils import get_implementation


from download_toolbox.interface import get_dataset_config_implementation


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
                              type=str)

        self.add_argument("name",
                          type=str)

    def add_configurations(self):
        self.add_argument("configurations",
                          type=argparse.FileType("r"),
                          nargs="+")
        return self

    def add_prefix(self):
        self.add_argument("-p",
                          "--prefix",
                          type=str,
                          default="loader")
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
    args = (LoaderArgParser().
            add_prefix().
            parse_args())

    data = dict(
        identifier=args.name,
        filenames=dict(),
        sources=dict(),
        masks=dict(),
        channels=dict(),
    )
    destination_filename = "{}.{}.json".format(args.prefix, args.name)

    if not os.path.exists(destination_filename):
        with open(destination_filename, "w") as fh:
            fh.write(orjson.dumps(data, option=orjson.OPT_INDENT_2).decode())
        logging.info("Created a configuration {} to build on".format(destination_filename))
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

    with open(args.name, "r") as fh:
        dest_data = orjson.loads(fh.read())

    for segment in args.segments:
        logging.info("Copying segment {} from {} to {}".format(segment, args.source, args.name))
        dest_data[segment] = source_data[segment]

    logging.info("Outputting {}".format(args.name))
    with open(args.name, "w") as fh:
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

    update_config(args.name, "filenames", filenames)
    update_config(args.name, "sources", cfgs)


def get_channel_info_from_processor(cfg_segment: str):
    args = (MetaArgParser().
            add_channel().
            parse_args())

    proc_impl = get_implementation(args.implementation)
    ds_config = get_dataset_config_implementation(args.ground_truth_dataset)
    processor = proc_impl(ds_config,
                          [args.channel_name,],
                          args.channel_name)
    processor.process()
    update_config(args.name, cfg_segment, {args.channel_name: processor.get_config()})


def add_channel():
    get_channel_info_from_processor("channels")


def add_mask():
    get_channel_info_from_processor("masks")
