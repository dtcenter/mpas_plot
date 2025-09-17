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
from multiprocessing import Pool


import pandas as pd  # optional, for nice datetime handling

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

#    # Pad with zeros for the first timestep
#    first = xr.zeros_like(field.isel({dim: 0}))
#    result = xr.concat([first, diffs], dim=dim)
#
#    # Preserve grid reference if available
#    if hasattr(field, "uxgrid"):
#        result = result.assign_coords(field.coords)
#        result.uxgrid = field.uxgrid

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

    # Attempt to read gridfile and handle error gracefully if grid info not available
    if not gridfile:
        # If no gridfile provided (empty string), read grid from first data file
        gf=datafiles[0]
    else:
        gf=gridfile
    try:
        ux.open_grid(gf)
    except Exception as e:
        logger.error(f'Could not read grid information from {gf}')
        if not gridfile:
            logger.error("Specify dataset:gridfile as a file that contains grid information")
            logger.error("For MPAS this is usually a history file or an init.nc file")
        raise e

    for f in datafiles:
        # Open data file lazily (metadata only, no data read yet)
        ds_raw = xr.open_dataset(f, decode_cf=False, chunks={})

        # Check that all requested vars exist
        missing = [v for v in vars_to_keep if v not in ds_raw.variables]
        if missing:
            raise KeyError(
                f"File {f} is missing required variables: {missing}. "
                f"Available variables: {list(ds_raw.variables)}"
            )

        ds_raw.close()

        # Identify variables to drop
        keep_vars = set(vars_to_keep) | {"xtime"}  # always keep xtime
        drop_vars = [v for v in ds_raw.data_vars if v not in keep_vars]

        ds = ux.open_dataset(
            gridfile,
            f,
            drop_variables=drop_vars,
            chunks={},
            decode_cf=False
        )

        datasets.append(ds)

    # Concatenate along Time
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        merged_xr = xr.concat(datasets, dim="Time", data_vars="minimal", coords="all")
        full_dataset = ux.UxDataset.from_xarray(merged_xr, uxgrid=datasets[0].uxgrid)

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

    # 1. Determine all native variables needed
    readvars = set()
    for varname in var_defs:
        readvars |= get_vars_to_read(var_defs, varname)

    # If no gridfile provided, set to empty string and handle in open_ux_subset()
    if not dsconf.get("gridfile"):
        dsconf["gridfile"]=""
    # 2. Open UxDataset lazily
    ds = open_ux_subset(dsconf["gridfile"], files, list(readvars))

    # 3. Compute derived variables and add to ds
    for varname, cfg in var_defs.items():
        if cfg["source"] == "derived":
            ds[varname] = compute_derived(var_defs, ds, varname)
            print(f"{varname=}")
            print(f"{ds[varname]=}")

    ds.load()

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
    expt_config=setup_config(args.config)

    # Load all data to plot as a single dataset
    logger.info('Loading data from netcdf files')
    dataset=load_full_dataset(expt_config["dataset"])

    logger.debug(f'{dataset=}')

    # Set up plotit() arguments
    logger.info('Setting up plot tasks')
    plotargs=setupargs(expt_config,dataset)

    logger.debug(f"{plotargs=}")
    # Make the plots!
    if args.procs > 1:
        logger.info(f"Plotting in parallel with {args.procs} tasks")
    with Pool(processes=args.procs) as pool:
        pool.starmap(plotithandler, plotargs)

#    proj=set_map_projection(expt_config["plot"]["projection"])
#    plotithandler(expt_config,dataset,dataset,"rainnc",0,"test",proj)

#    proj=set_map_projection(expt_config["dataset"]["vars"]["rainnc"]["plot"]["projection"])
#    plotithandler(expt_config["dataset"]["vars"]["rainnc"]["plot"],dataset,dataset,"rainnc",1,"test",proj)
    logger.info("Done plotting all figures!")
