import logging

from dateutil.relativedelta import relativedelta

from download_toolbox.interface import get_dataset_config_implementation, get_implementation

from preprocess_toolbox.dataset.process import regrid_dataset, rotate_dataset, reproject_datasets_from_config
from preprocess_toolbox.dataset.spatial import spatial_interpolation
from preprocess_toolbox.dataset.time import process_missing_dates
from preprocess_toolbox.cli import ProcessingArgParser, process_split_args, csv_arg
from preprocess_toolbox.interface import get_processor_from_source
from preprocess_toolbox.processor import NormalisingChannelProcessor
from preprocess_toolbox.utils import get_config


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
                          config_path=args.config,
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
    proc.process(config_path=args.config)


def init_dataset(args):
    ds_config = get_dataset_config_implementation(args.source,
                                                  output_path=args.config)

    if args.destination_id is not None:
        splits = process_split_args(args, frequency=ds_config.frequency)

        if len(splits) > 0:
            logging.info("Processing based on {} provided splits".format(len(splits)))
            split_dates = [date for split in splits.values() for date in split]

            all_files = dict()
            for var_config in ds_config.variables:
                # This is not processing, so we naively extend the range as the split extension args might be set
                # and if they aren't the preprocessing will dump the dates via Processor
                lag = relativedelta(**{"{}s".format(ds_config.frequency.attribute): args.split_head})
                lead = relativedelta(**{"{}s".format(ds_config.frequency.attribute): args.split_tail})
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
            add_extra_args([
                (("-cp", "--coord-method"), dict(default=None, type=str,
                                                 help="Method that processes cube prior to regrid")),
                (("-ca", "--coord-method-args"), dict(default=[], type=csv_arg,
                                                      help="CSV of simple arguments for coord-method")),
                (("-rp", "--regrid-method"), dict(default=None, type=str,
                                                  help="Method that processes cube data after regrid")),
                (("-ra", "--regrid-method-args"), dict(default=[], type=str,
                                                       help="CSV of simple arguments for regrid-method")),
            ]).
            parse_args())
    ds, ds_config = init_dataset(args)

    coord_proc = get_implementation(args.coord_method) if args.coord_method is not None else None
    regrid_proc = get_implementation(args.regrid_method) if args.regrid_method is not None else None

    regrid_dataset(args.reference, ds_config,
                   coord_processing=coord_proc,
                   coord_processing_args=args.coord_method_args,
                   regrid_processing=regrid_proc,
                   regrid_processing_args=args.regrid_method_args)
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


def reproject():
    """
    Reproject a dataset from one CRS to another CRS.

    Example usage:
        preprocess_reproject -v -c ./reproject.era5.day.north.json --workers 8 -ps train \
        -sn train,val,test -ss 2023-1-1,2024-2-1,2024-12-1 -se 2023-12-31,2024-2-14,2024-12-1 \
        -sh 4 -st 1 --source-crs 'EPSG:4326' --target-crs 'EPSG:6931' --shape 500 \
        --ease2 data.aws.day.north.json proc.aws

        This command reprojects an ERA5 lat/lon grid (EPSG:4326) to an EASE Grid 2.0 grid
        (EPSG:6931) with an output shape of (500, 500). The dataset only processes dates
        for the splits defined: 2023-1-1 to 2024-2-1, 2024-2-1 to 2024-12-1 and
        2024-12-1 to 2025-1-1.
        It adds 4 days prior to start and 1 day after due to `-sh` and `-st` flags.
    """
    args = (
        ProcessingArgParser()
        .add_destination()
        .add_splits()
        .add_extra_args(
            [
                (("-w", "--workers"), dict(default=1, type=int)),
                (("-sc", "--source-crs"), dict(
                        default="EPSG:4326",
                        type=str,
                        required=True,
                        help="Source dataset CRS definition: EPSG code (e.g., `EPSG:4326`)",
                )),
                (("-tc", "--target-crs"), dict(
                        default="EPSG:6931",
                        type=str,
                        required=False,
                        help="Target dataset CRS definition: Full cartopy.crs expression (e.g., `EPSG:6931`)",
                )),
                (("-r", "--resolution"), dict(
                        default=None,
                        type=float,
                        required=False,
                        help="Resolution of output grid (in meters or degrees). Can only specify either `--resolution` or `--shape`, not both",
                )),
                (("-s", "--shape"), dict(
                        default="720,720",
                        type=str,
                        required=False,
                        help="Shape of output grid (in pixels, e.g. '720,720'). Can only specify either `--resolution` or `--shape`, not both",
                )),
                (("-e", "--ease2"), dict(
                        action="store_true",
                        help="Enable to output an EASE-Grid 2.0 conformal grid",
                )),
                (("-cn", "--coarsen"), dict(
                        default=1,
                        type=int,
                        help="To coarsen output grid by this integer factor.",
                )),
                (("-in", "--interpolate-nans"), dict(
                        action="store_true",
                        help="Enable nearest neighbour interpolation to fill in missing areas.",
                )),
            ]
        )
        .parse_args()
    )
    # Initially copy across the source data from `./data/` to the destination
    # `./processed_data/`
    ds, ds_config = init_dataset(args)
    # Reproject and overwrite the copied data
    reproject_datasets_from_config(
        ds_config,
        source_crs=args.source_crs,
        target_crs=args.target_crs,
        resolution=args.resolution,
        shape=args.shape,
        ease2=args.ease2,
        coarsen=args.coarsen,
        interpolate_nans=args.interpolate_nans,
        workers=args.workers,
    )
    ds_config.save_config()
