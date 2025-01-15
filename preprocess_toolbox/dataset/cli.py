import logging

from dateutil.relativedelta import relativedelta

from download_toolbox.interface import get_dataset_config_implementation

from preprocess_toolbox.dataset.process import regrid_dataset, rotate_dataset
from preprocess_toolbox.dataset.spatial import spatial_interpolation
from preprocess_toolbox.dataset.time import process_missing_dates
from preprocess_toolbox.cli import ProcessingArgParser, process_split_args, csv_arg
from preprocess_toolbox.interface import get_processor_from_source
from preprocess_toolbox.processor import NormalisingChannelProcessor
from preprocess_toolbox.utils import get_config, get_implementation


def process_dataset():
    args = (ProcessingArgParser().
            add_concurrency().
            add_destination().
            add_implementation().
            add_reference().
            add_splits().
            add_trends().
            add_vars()).parse_args()
    ds_config = get_dataset_config_implementation(args.source)
    splits = process_split_args(args, frequency=ds_config.frequency)

    implementation = NormalisingChannelProcessor \
        if args.implementation is None \
        else get_implementation(args.implementation)

    proc = implementation(ds_config,
                          args.anom,
                          splits,
                          args.abs,
                          anom_clim_splits=args.processing_splits,
                          identifier=args.destination_id,
                          # TODO: nomenclature is old here, lag and lead make sense in forecasting, but not in here
                          #  so this mapping should be revised throughout the library - we don't necessarily forecast!
                          lag_time=args.split_head,
                          lead_time=args.split_tail,
                          linear_trends=args.trends,
                          linear_trend_steps=args.trend_lead,
                          normalisation_splits=args.processing_splits,
                          parallel_opens=args.parallel_opens or False,
                          ref_procdir=args.ref)
    proc.process()


def init_dataset(args):
    ds_config = get_dataset_config_implementation(args.source)

    if args.destination_id is not None:
        splits = process_split_args(args, frequency=ds_config.frequency)

        if len(splits) > 0:
            logging.info("Processing based on {} provided splits".format(len(splits)))
            split_dates = [date for split in splits.values() for date in split]

            all_files = dict()
            for var_config in ds_config.variables:
                # This is not processing, so we naively extend the range as the split extension args might be set
                # and if they aren't the preprocessing will dump the dates via Processor
                lag = relativedelta({"{}s".format(ds_config.frequency.attribute): args.split_head})
                lead = relativedelta({"{}s".format(ds_config.frequency.attribute): args.split_head})
                min_filepath = ds_config.var_filepath(var_config, [min(split_dates) - lag])
                max_filepath = ds_config.var_filepath(var_config, [max(split_dates) + lead])

                var_files = sorted(ds_config.var_files[var_config.name])
                min_index = var_files.index(min_filepath)
                max_index = var_files.index(max_filepath)
                all_files[var_config.name] = var_files[min_index:max_index+1]
            ds_config.var_files = all_files
        else:
            logging.info("No splits provided, assuming to copy the whole dataset")

        ds_config.copy_to(args.destination_id,
                          base_path=args.destination_path)

    var_names = None if "var_names" not in args else args.var_names
    ds = ds_config.get_dataset(var_names)
    return ds, ds_config


def missing_time():
    args = (ProcessingArgParser().
            add_destination().
            add_var_name().
            add_splits().
            parse_args())
    ds, ds_config = init_dataset(args)

    for var_name in args.var_names:
        logging.info("Processing missing dates for {}".format(var_name))
        ds = process_missing_dates(ds,
                                   ds_config,
                                   var_name)

    ds_config.save_data_for_config(source_ds=ds)


def missing_spatial():
    args = (ProcessingArgParser(suppress_logs=["PIL"]).
            add_destination().
            add_var_name().
            add_splits().
            add_extra_args([
                (("-m", "--mask-configuration"), dict()),
                (("-mp", "--masks"), dict(type=csv_arg)),
            ]).
            parse_args())
    ds, ds_config = init_dataset(args)
    mask_proc = None

    if len(args.masks) > 0:
        proc_config = get_config(args.mask_configuration)["data"]
        mask_proc = get_processor_from_source("masks", proc_config)

    for var_name in args.var_names:
        logging.info("Processing missing dates for {}".format(var_name))
        ds[var_name] = spatial_interpolation(getattr(ds, var_name).compute(),
                                             ds_config,
                                             mask_proc,
                                             args.masks,
                                             save_comparison_fig=False)

    ds_config.save_data_for_config(source_ds=ds)


def regrid():
    args = (ProcessingArgParser().
            add_ref_ds().
            add_destination().
            add_splits().
            parse_args())
    ds, ds_config = init_dataset(args)
    regrid_dataset(args.reference, ds_config)
    ds_config.save_config()


def rotate():
    args = (ProcessingArgParser().
            add_ref_ds().
            add_destination().
            add_splits().
            add_var_name().
            parse_args())
    ds, ds_config = init_dataset(args)
    if args.var_names:
        rotate_dataset(args.reference, ds_config, vars_to_rotate=args.var_names)
    else:
        rotate_dataset(args.reference, ds_config)
    ds_config.save_config()
