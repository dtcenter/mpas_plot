#!/usr/bin/env python3
"""
Script for plotting MPAS input and/or output in native NetCDF format"
"""
import argparse
import copy
import glob
import logging
import sys
import os
import multiprocessing

import gc, psutil
proc = psutil.Process(os.getpid())


print("Importing uxarray; this may take a while...")
import uxarray as ux
import xarray as xr

import uwtools.api.config as uwconfig

from plot_mpas_netcdf import plotithandler

logger = logging.getLogger(__name__)


# =====================
# User-defined derived functions
# =====================
def diff_prev_timestep(field: ux.UxDataArray, dim: str = "Time") -> ux.UxDataArray:
    """
    Return timestep-to-timestep differences for input field.
    First timestep is filled with zeros.
    """
    # Compute differences along Time
    result = field.diff(dim=dim, n=1)

    return result


def sum_fields(x1, x2):
    return x1 + x2


DERIVED_FUNCTIONS = {
    "diff_prev_timestep": diff_prev_timestep,
    "sum_fields": sum_fields,
}


# =====================
# Recursive variable parsing
# =====================
def get_vars_to_read(var_defs: dict, name: str, seen=None) -> set:
    """
    Recursively return all native variables needed to compute `name`.
    """
    if seen is None:
        seen = set()
    if name in seen:
        return set()
    seen.add(name)

    cfg = var_defs[name]

    if cfg["source"] == "native":
        return {name}
    elif cfg["source"] == "derived":
        inputs = cfg.get("inputs", [])
        result = set()
        for v in inputs:
            result |= get_vars_to_read(var_defs, v, seen)
        return result
    else:
        raise ValueError(f"Unknown source type {cfg['source']} for {name}")

# =====================
# Lazy UxDataset opening
# =====================
def open_ux_subset(gridfile, datafiles, vars_to_keep):

    if isinstance(datafiles, str):
        datafiles = [datafiles]

    datasets = []

    logger.debug(f"Opening gridfile; Memory usage:{proc.memory_info().rss/1024**2} MB")
    # Attempt to read gridfile and handle error gracefully if grid info not available
    if not gridfile:
        # If no gridfile provided (empty string), read grid from first data file
        gf=datafiles[0]
    else:
        gf=gridfile
    try:
        uxgrid=ux.open_grid(gf)
    except Exception as e:
        logger.error(f'Could not read grid information from {gf}')
        if not gridfile:
            logger.error("Specify dataset:gridfile as a file that contains grid information")
            logger.error("For MPAS this is usually a history file or an init.nc file")
        raise e

    keep_set = set(vars_to_keep) | {"xtime"}  # always keep xtime


    xr_ds_list = []
    for f in datafiles:
        logger.debug(f"Opening dataset file {f}\nMemory usage:{proc.memory_info().rss/1024**2} MB")
        ds = xr.open_dataset(f, decode_cf=False, chunks={})  # lazy
        missing = [v for v in vars_to_keep if v not in ds.variables]
        if missing:
            raise KeyError(f"{f} missing required variables: {missing}")

        available_keep = [v for v in keep_set if v in ds.variables]
        xr_ds_list.append(ds[available_keep])

    merged = xr.concat(xr_ds_list, dim="Time", data_vars="minimal", coords="all")

    logger.debug(f"Attaching grid to dataset; Memory usage:{proc.memory_info().rss/1024**2} MB")
    full_dataset = ux.UxDataset.from_xarray(merged, uxgrid=uxgrid)

    return full_dataset

# =====================
# Recursive derived variable computation
# =====================
def compute_derived(var_defs, ds, name):
    cfg = var_defs[name]

    if cfg["source"] == "native":
        return ds[name]

    elif cfg["source"] == "derived":
        func_name = cfg["function"]
        inputs = cfg.get("inputs", [])

        # Recursively get input arrays
        input_arrays = [compute_derived(var_defs, ds, v) for v in inputs]


        # Lookup function
        func = DERIVED_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown derived function: {func_name}")

        # Compute derived variable
        logger.debug(f"Computing derived variable {name} with function {func_name}")
        result = func(*input_arrays)

        # Attach metadata if provided
        attrs = cfg.get("attrs", {})
        if isinstance(result, xr.DataArray):
            result.attrs.update(attrs)

        return result

# =====================
# Load dataset based on user settings in dataset config
# =====================
def load_full_dataset(dsconf):
    files = sorted(glob.glob(dsconf["files"]))
    var_defs = dsconf["vars"]

    logger.debug(f"Determining variables to read from file\nMemory usage:{proc.memory_info().rss/1024**2} MB")

    # 1. Determine all native variables needed
    readvars = set()
    for varname in var_defs:
        readvars |= get_vars_to_read(var_defs, varname)

    # If no gridfile provided, set to empty string and handle in open_ux_subset()
    if not dsconf.get("gridfile"):
        dsconf["gridfile"]=""
    # 2. Open UxDataset lazily
    ds = open_ux_subset(dsconf["gridfile"], files, list(readvars))

    logger.debug(f"Compute derived variables\nMemory usage:{proc.memory_info().rss/1024**2} MB")
    # 3. Compute derived variables and add to ds
    for varname, cfg in var_defs.items():
        if cfg["source"] == "derived":
            ds[varname] = compute_derived(var_defs, ds, varname)

    return ds


def setupargs(config_d: dict,uxds: ux.UxDataset):
    """
    Sets up the argument list for plotit to allow for parallelization with Python starmap
    """
    args = []

    for var in config_d["dataset"]["vars"]:
        # Update each variable's plot settings dictionary
        plotdict=copy.copy(config_d["plot"])
        if update_dict:=config_d["dataset"]["vars"][var].get("plot"):
            plotdict.update(update_dict)
        config_d["dataset"]["vars"][var]["plot"]=plotdict

        vardict=config_d["dataset"]["vars"][var]
        # Plot all levels by default
        if not vardict.get("lev"):
            vardict["lev"]="all"
        vardict["vertcoord"]=vardict.get("vertcoord","nVertLevels")
        if vardict["lev"] in [ ["all"], "all" ]:
            if vardict["vertcoord"] in uxds[var].dims:
                levels = range(0,len(uxds[var][vardict["vertcoord"]]))
            else:
                logger.debug(f"{var} has no vertical coordinate, plotting only level")
                levels = [0]
        elif isinstance(vardict["lev"], list):
            levels = vardict["lev"]
        elif isinstance(vardict["lev"], int):
            levels = [vardict["lev"]]
        else:
            raise TypeError(f"Invalid level {vardict['lev']} specified for variable {var}")

        for lev in levels:
            args.append( (config_d,uxds,var,lev) )

    return args


def setup_config(config: str, default: str="default_options.yaml") -> dict:
    """
    Function for reading in dictionary of configuration settings, and performing basic checks
    on those settings

    Args:
        config  (str) : The full path of the user config file
        default (str) : The full path of the default config file

    Returns:
        dict: A dictionary of the configuration settings after applying defaults and user settings,
              as well as some basic consistency checks
    """
    logger.debug(f"Reading defaults file {default}")
    try:
        expt_config = uwconfig.get_yaml_config(config=default)
    except Exception as e:
        logger.critical(e)
        logger.critical(f"Error reading {config}, check above error trace for details")
        sys.exit(1)
    logger.debug(f"Reading options file {config}")
    try:
        user_config = uwconfig.get_yaml_config(config=config)
    except Exception as e:
        logger.critical(e)
        logger.critical(f"Error reading {config}, check above error trace for details")
        sys.exit(1)

    # Update the dict read from defaults file with the dict read from user config file
    expt_config.update_values(user_config)

    if not expt_config["dataset"].get("gridfile"):
        expt_config["dataset"]["gridfile"]=""

    # Perform consistency checks
    if not expt_config["data"].get("lev"):
        logger.debug("Level not specified in config, will use level 0 if multiple found")
        expt_config["data"]["lev"]=0

    if isinstance(expt_config["plot"]["title"],str):
        raise TypeError("plot:title should be a dictionary, not a string\n"\
                        "Adjust your config.yaml accordingly. See default_options.yaml for details.")

    logger.debug("Expanding references to other variables and Jinja templates")
    expt_config.dereference()
    return expt_config


def setup_logging(logfile: str = "log.mpas_plot", debug: bool = False):
    """
    Sets up logging, printing high-priority (INFO and higher) messages to screen, and printing all
    messages with detailed timing and routine info in the specified text file.

    If debug = True, print all messages to both screen and log file.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if called more than once
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console_formatter = logging.Formatter("%(levelname)-8s %(message)s")
    console.setFormatter(console_formatter)

    # File handler
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s %(name)s.%(funcName)s %(levelname)-8s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(file_formatter)

    # Add handlers
    root_logger.addHandler(console)
    root_logger.addHandler(fh)


    # Suppress debug prints from matplotlib
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    root_logger.debug("Logging configured")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for plotting a custom field on the native MPAS grid from native NetCDF format files"
    )
    parser.add_argument('-c', '--config', type=str, default='config_custom.yaml',
                        help='File used to specify plotting options')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Script will be run in debug mode with more verbose output')
    parser.add_argument('-p', '--procs', type=int, default=1,
                        help='Number of processors for generating plots in parallel')

    args = parser.parse_args()

    setup_logging(debug=args.debug)

    # Load settings from config file
    logger.info('Loading user config settings')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    expt_config=setup_config(args.config)

    # Load all data to plot as a single dataset
    logger.info('Loading data from netcdf files')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    dataset=load_full_dataset(expt_config["dataset"])

    logger.debug(f'{dataset=}')

    # Set up plotit() arguments
    logger.info('Setting up plot tasks')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    plotargs=setupargs(expt_config,dataset)

    logger.info('Submitting to starmap')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    logger.debug(f"{plotargs=}")
    # Make the plots!
    if args.procs > 1:
        logger.info(f"Plotting in parallel with {args.procs} tasks")
    # This is needed to avoid some kind of file handle clobbering mumbo-jumbo with netCDF
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=args.procs) as pool:
        pool.starmap(plotithandler, plotargs)

    logger.info("Done plotting all figures!")
