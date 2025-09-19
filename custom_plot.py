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

from plot_mpas_netcdf import setup_logging, setup_config, set_map_projection, plotithandler
from file_read import load_dataset, invalid_vars

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

    print("Open gridfile, RSS MB:", proc.memory_info().rss/1024**2)
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

    print("Reading datafiles, RSS MB:", proc.memory_info().rss/1024**2)

    xr_ds_list = []
    for f in datafiles:
        print(f"For file {f}")
        print("Open dataset, RSS MB:", proc.memory_info().rss/1024**2)
        ds = xr.open_dataset(f, decode_cf=False, chunks={})  # lazy
        missing = [v for v in vars_to_keep if v not in ds.variables]
        if missing:
            raise KeyError(f"{f} missing required variables: {missing}")

        available_keep = [v for v in keep_set if v in ds.variables]
        print("Append dataset, RSS MB:", proc.memory_info().rss/1024**2)
        xr_ds_list.append(ds[available_keep])

    print("Merge dataset, RSS MB:", proc.memory_info().rss/1024**2)
    merged = xr.concat(xr_ds_list, dim="Time", data_vars="minimal", coords="all")

    print("Attach grid to dataset, RSS MB:", proc.memory_info().rss/1024**2)
    full_dataset = ux.UxDataset.from_xarray(merged, uxgrid=uxgrid)

    return full_dataset

# =====================
# Recursive derived variable computation
# =====================
def compute_derived(var_defs, ds, name):
    cfg = var_defs[name]
    print(f"{cfg=}")

    if cfg["source"] == "native":
        return ds[name]

    elif cfg["source"] == "derived":
        func_name = cfg["function"]
        inputs = cfg.get("inputs", [])
        print(f"{func_name=}")
        print(f"{inputs=}")

        # Recursively get input arrays
        input_arrays = [compute_derived(var_defs, ds, v) for v in inputs]
        print(f"{input_arrays=}")


        # Lookup function
        func = DERIVED_FUNCTIONS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown derived function: {func_name}")

        # Compute derived variable
        result = func(*input_arrays)
        print(f"{result=}")

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

    print("1. Determine all native variables needed, RSS MB:", proc.memory_info().rss/1024**2)
    # 1. Determine all native variables needed
    readvars = set()
    for varname in var_defs:
        readvars |= get_vars_to_read(var_defs, varname)

    # If no gridfile provided, set to empty string and handle in open_ux_subset()
    if not dsconf.get("gridfile"):
        dsconf["gridfile"]=""
    print("2. Open UxDataset lazily, RSS MB:", proc.memory_info().rss/1024**2)
    # 2. Open UxDataset lazily
    ds = open_ux_subset(dsconf["gridfile"], files, list(readvars))

    print("3. Compute derived variables and add to ds, RSS MB:", proc.memory_info().rss/1024**2)
    # 3. Compute derived variables and add to ds
    for varname, cfg in var_defs.items():
        if cfg["source"] == "derived":
            ds[varname] = compute_derived(var_defs, ds, varname)
            print(f"{varname=}")
            print(f"{ds[varname]=}")

    print("4. RSS MB:", proc.memory_info().rss/1024**2)
    print(f"{ds=}")

    return ds


def setupargs(config_d: dict,uxds: ux.UxDataset):
    """ 
    Sets up the argument list for plotit to allow for parallelization with Python starmap
    """ 
    
    args = [] 

    # Set up map projection properties
    proj=set_map_projection(expt_config["plot"]["projection"])

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
        if vardict["lev"] in [ ["all"], "all" ]:
            if "nVertLevels" in uxds[var].dims:
                levels = range(0,len(uxds[var]["nVertLevels"]))
            else:
                levels = [0]
        elif isinstance(vardict["lev"], list):
            levels = vardict["lev"]
        elif isinstance(vardict["lev"], int):
            levels = [vardict["lev"]]
        else:
            raise TypeError(f"Invalid level {vardict['lev']} specified for variable {var}")

        print(f"{var=}\n{levels=}")
        for lev in levels:
            args.append( (config_d,uxds,var,lev,proj) )

    return args

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
    print("RSS MB:", proc.memory_info().rss/1024**2)
    expt_config=setup_config(args.config)

    # Load all data to plot as a single dataset
    logger.info('Loading data from netcdf files')
    print("RSS MB:", proc.memory_info().rss/1024**2)
    dataset=load_full_dataset(expt_config["dataset"])

    logger.debug(f'{dataset=}')

    # Set up plotit() arguments
    logger.info('Setting up plot tasks')
    print("RSS MB:", proc.memory_info().rss/1024**2)
    plotargs=setupargs(expt_config,dataset)

    logger.info('Submitting to starmap')
    print("RSS MB:", proc.memory_info().rss/1024**2)
    logger.debug(f"{plotargs=}")
    # Make the plots!
    if args.procs > 1:
        logger.info(f"Plotting in parallel with {args.procs} tasks")
    # This is needed to avoid some kind of file handle clobbering mumbo-jumbo with netCDF
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=args.procs) as pool:
        pool.starmap(plotithandler, plotargs)

#    proj=set_map_projection(expt_config["plot"]["projection"])
#    plotithandler(expt_config,dataset,dataset,"rainnc",0,"test",proj)

#    proj=set_map_projection(expt_config["dataset"]["vars"]["rainnc"]["plot"]["projection"])
#    plotithandler(expt_config["dataset"]["vars"]["rainnc"]["plot"],dataset,dataset,"rainnc",1,"test",proj)
    logger.info("Done plotting all figures!")
