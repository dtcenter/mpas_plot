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
import signal
import traceback
import time
import re
from datetime import datetime, timedelta

import gc, psutil
proc = psutil.Process(os.getpid())

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs


import uxarray as ux
import xarray as xr

import uwtools.api.config as uwconfig
import custom_functions
from cross_section import plot_cross_section, cross_section_vars

from plot_functions import (set_map_projection, set_patterns_and_outfile, get_data_extent,
                            get_colormap, finish_plot, deep_merge)


logger = logging.getLogger(__name__)

# Custom error class for ensuring subprocess exceptions are handled correctly
class PlotError(Exception):
    pass


def get_vars_to_read(var_defs: dict, name: str, seen=None) -> set:
    """
    Recursively return all native variables needed to compute `name`.
    """
    if seen is None:
        seen = set()
    if name in seen:
        return set()
    seen.add(name)

    if not (cfg:=var_defs.get(name)):
        return {name}

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


def open_ux_subset(gridfile, datafiles, vars_to_keep, optional_vars=()):

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

    # optional_vars are read if they are present, but are not an error if they are not
    keep_set = set(vars_to_keep) | set(optional_vars) | {"xtime"}  # always keep xtime if present

    xr_ds_list = []
    for i, f in enumerate(datafiles):
        logger.debug(f"Opening dataset file {f}\nMemory usage:{proc.memory_info().rss/1024**2} MB")
        ds = xr.open_dataset(f, decode_cf=False, chunks={})  # lazy
        missing = [v for v in vars_to_keep if v not in ds.variables]
        if missing:
            raise KeyError(f"{f} missing required variables: {missing}")

        # Normalize time to xtime if not already present
        if "xtime" not in ds.variables:
            timestring = _extract_timestring(ds, f, i)
            ds["xtime"] = xr.DataArray(
                [[c for c in timestring.ljust(64)[:64]]],
                dims=["Time", "StrLen"]
            )
            logger.debug(f"Synthesized xtime={timestring} for {f}")

        available_keep = [v for v in keep_set if v in ds.variables]
        xr_ds_list.append(ds[available_keep])

    merged = xr.concat(xr_ds_list, dim="Time", data_vars="minimal", coords="all")

    # Variables such as zgrid are usually only in the static/init file, so fall back to
    # reading them from the gridfile. This must happen before the grid is attached below,
    # because that is what maps the nCells dimension onto the grid's n_face.
    for v in optional_vars:
        if v not in merged.variables:
            griddata = xr.open_dataset(gf, decode_cf=False, chunks={})
            if v in griddata.variables:
                logger.debug(f"Reading {v} from gridfile {gf}")
                merged[v] = griddata[v]
            else:
                logger.warning(f"Could not find {v} in the data files or in {gf}")

    logger.debug(f"Attaching grid to dataset; Memory usage:{proc.memory_info().rss/1024**2} MB")
    full_dataset = ux.UxDataset.from_xarray(merged, uxgrid=uxgrid)

    return full_dataset

# =====================
# Recursive derived variable computation
# =====================
def compute_derived(var_defs, ds, name):
    if not (cfg:=var_defs.get(name)):
        return ds[name]

    if cfg["source"] == "native":
        return ds[name]

    elif cfg["source"] == "derived":
        func_name = cfg["function"]
        inputs = cfg.get("inputs", [])

        # Recursively get input arrays
        input_arrays = [compute_derived(var_defs, ds, v) for v in inputs]


        # Lookup function
        func = custom_functions.DERIVED_FUNCTIONS.get(func_name)
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
def load_full_dataset(dsconf, optional_vars=()):
    if isinstance(dsconf["files"],list):
        files = sorted(dsconf["files"])
    else:
        files = sorted(glob.glob(dsconf["files"]))
        if not files:
            raise FileNotFoundError(dsconf["files"])
    dsconf["files"]=files
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
    ds = open_ux_subset(dsconf["gridfile"], dsconf["files"], list(readvars),
                        optional_vars=optional_vars)

    logger.debug(f"Compute derived variables\nMemory usage:{proc.memory_info().rss/1024**2} MB")
    # 3. Compute derived variables and add to ds
    for varname, cfg in var_defs.items():
        if cfg["source"] == "derived":
            ds[varname] = compute_derived(var_defs, ds, varname)

    return ds


def plotithandler(config_d: dict,uxds: ux.UxDataset,var: str,lev: int,timeint: int, timestring: str) -> None:
    """
    A wrapper for plotit() that handles errors for Python multiprocessing, as well as preprocessing the
    UxDataSet into a UxDataArray with just the variable and timestep we want to plot
    """

    # Since this is a spawn process, we need to re-initialize logging
    logger = logging.getLogger(__name__)

    logger.info(f"Starting plotit() for {var=}, {lev=}")

    if var not in list(uxds.data_vars.keys()):
        msg = f"{var=} is not a valid variable\n\n(you never should have made it this far though)\n\n{uxds.data_vars}"
        raise ValueError(msg)

    field=uxds[var]

    if timestring:
        ftime_dt = datetime.strptime(timestring.strip(), "%Y-%m-%d_%H:%M:%S")

    # timeint was set to -1 in setup_args if variable has no time dimension
    if timeint == -1:
        timeint=0
        plotfield=field
    else:
        plotfield=field.isel(Time=timeint)
    plottype = config_d['dataset']['vars'][var]['plot'].get("type","map")
    plotfunc = PLOT_FUNCTIONS.get(plottype)
    if plotfunc is None:
        raise ValueError(f"Unknown plot type: {plottype}\n\nValid options are {list(PLOT_FUNCTIONS)}")
    try:
        plotfunc(config_d['dataset']['vars'][var],plotfield,var,lev,
                 config_d['dataset']['files'][0],ftime_dt,uxds=uxds)
    except Exception as e:
        logger.error(f'Could not plot variable {var}, level {lev}')
        logger.debug(f"Arguments to plotit():\n{config_d['dataset']['vars'][var]=}\n{plotfield=}\n"\
                      f"{var=}\n{lev=}\n{config_d['dataset']['files'][0]=}\n{ftime_dt=}")
        raise PlotError(traceback.format_exc()) from None

def plotit(vardict: dict,uxda: ux.UxDataArray,var: str,lev: int,filepath: str,ftime,
           uxds: ux.UxDataset=None) -> None:
    """
    The main program that makes the plot(s)
    Args:
        vardict      (dict): A dictionary containing experiment settings specific to the variable being plotted
        uxds (ux.UxDataArray): A ux.UxDataArray object containing the data to be plotted and grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        ftime    (datetime): The forecast valid time as a datetime object
        uxds (ux.UxDataset): The full dataset, for plot types that need other variables;
                             unused here, but part of the shared plot-type signature

    Returns:
        None
    """

    plotdict=vardict["plot"]
    plotstart = time.time()

    # Make vertical coordinate a keyword argument
    vertargs={vardict["vertcoord"]: lev}
    if vardict["vertcoord"] in uxda.dims:
        varslice = uxda.isel(**vertargs)
    else:
        if lev > 0:
            logger.error(f"Variable {var} only has one vertical level; can not plot {lev=}")
            return
        varslice = uxda

    if "n_face" not in uxda.dims:
        logger.warning(f"Variable {var} not face-centered, will interpolate to faces")
        varslice = varslice.remap.inverse_distance_weighted(uxda.uxgrid,
                                                          remap_to='face centers', k=3)
        logger.debug(f"Data slice after interpolation:\n{varslice=}")

    logger.debug(f"{varslice=}")

    # Set up map projection properties
    logger.debug(plotdict["projection"])
    proj=set_map_projection(plotdict["projection"])

    # Create figure and plot axes
    fig, ax = plt.subplots(1, 1, figsize=(plotdict["figwidth"],
                           plotdict["figheight"]), dpi=plotdict["dpi"],
                           constrained_layout=True,
                           subplot_kw=dict(projection=proj))

    # Create figure and plot axes

    #Plot political boundaries if requested
    if plotdict.get("boundaries"):
        pb=plotdict["boundaries"]
        if pb.get("enable"):
            # Users can set these values to scalars or lists; if scalar provided, re-format to list with three identical values
            for setting in ["color", "linewidth", "scale"]:
                if type(pb[setting]) is not list:
                    pb[setting]=[pb[setting],pb[setting],pb[setting]]
            if pb["detail"]==2:
                if pb["scale"][2]!='10m':
                    logger.info(f"scale={pb['scale'][2]} not supported for county-level boundaries")
                    logger.info("Setting scale to 10m")
                    pb["scale"][2]='10m'
                ax.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                               scale=pb["scale"][2],edgecolor=pb["color"][2],
                               facecolor='none',linewidth=pb["linewidth"][2], name='admin_2_counties'))
            if pb["detail"]>0:
                ax.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                               scale=pb["scale"][1],edgecolor=pb["color"][1],
                               facecolor='none',linewidth=pb["linewidth"][1], name='admin_1_states_provinces'))
            ax.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                           scale=pb["scale"][0],edgecolor=pb["color"][0],
                           facecolor='none',linewidth=pb["linewidth"][0], name='admin_0_countries'))
    #Plot coastlines if requested
    if plotdict.get("coastlines"):
        pcl=plotdict["coastlines"]
        if pcl.get("enable"):
            ax.add_feature(cfeature.NaturalEarthFeature(category='physical',color=pcl["color"],facecolor='none',
                           linewidth=pcl["linewidth"], scale=pcl["scale"], name='coastline'))
    #Plot lakes if requested
    if plotdict.get("lakes"):
        pl=plotdict["lakes"]
        if pl.get("enable"):
            ax.add_feature(cfeature.NaturalEarthFeature(category='physical',edgecolor=pl["color"],facecolor='none',
                           linewidth=pl["linewidth"], scale=pl["scale"], name='lakes'))


    # Handle color mapping
    cmap = get_colormap(plotdict)


    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")

    # Set axes extent based on data extent and/or user settings
    if None in plotdict["projection"]["lonrange"] or None in plotdict["projection"]["latrange"]:
        logger.info('One or more latitude/longitude range values were not set; plotting full projection')
        extent=get_data_extent(varslice)
        # If auto-calculated extent is greater than globe, reset to full globe
        if (extent[1] - extent[0]) > 360:
            extent[1]=180
            extent[0]=-180
        if (extent[3] - extent[2]) > 180:
            extent[3]=90
            extent[2]=-90
    else:
        extent=[plotdict["projection"]["lonrange"][0], plotdict["projection"]["lonrange"][1], plotdict["projection"]["latrange"][0], plotdict["projection"]["latrange"][1]]
    logger.debug(f'Domain extent: {extent}')

    is_full_globe = (extent[0] <= -180 and extent[1] >= 180 and extent[2] <= -90 and extent[3] >= 90)
    if is_full_globe:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Create image with polycollection or raster
    if plotdict["polycollection"]:
        try:
            if plotdict["periodic_bdy"]:
                logger.info("Creating polycollection with periodic_bdy=True")
                logger.info("NOTE: This option can be very slow for large domains")
                pc=varslice.to_polycollection(periodic_elements='split')
            else:
                raise Exception("periodic_bdy: False does not work (known UXarray bug). Set periodic_bdy: True")
                pc=varslice.to_polycollection(periodic_elements='exclude')
            logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
        except ValueError as e:
            logger.critical(e)
            msg=f"Variable {var} may not have standard vertical levels (nVertLevels):\n"
            msg+=f"dimensions={varslice.dims}\n"
            msg+="Check documentation for how to handle variables with non-standard vertical levels"
            raise PlotError(msg)

        pc.set_antialiased(False)

        pc.set_edgecolor(plotdict['edges']['color'])
        pc.set_linewidth(plotdict['edges']['width'])
        pc.set_transform(ccrs.Geodetic())

        logger.debug("Adding collection to plot axes")
        if plotdict["projection"]["projection"] != "PlateCarree":
            logger.info(f"Interpolating to {plotdict['projection']['projection']} projection; this may take a while...")
        if None in plotdict["projection"]["lonrange"] or None in plotdict["projection"]["latrange"]:
            img = ax.add_collection(pc, autolim=True)
            ax.autoscale()
        else:
            img = ax.add_collection(pc)
    else:
        try:
            raster = varslice.to_raster(ax=ax, pixel_ratio=plotdict['pixel_ratio'])
        except ValueError as e:
            logger.critical(e)
            msg=f"Variable {var} may not have standard vertical levels (nVertLevels):\n"
            msg+=f"dimensions={varslice.dims}\n"
            msg+="Check documentation for how to handle variables with non-standard vertical levels"
            raise PlotError(msg)

        img = ax.imshow(
            raster,
            cmap=cmap,
            origin="lower",
            extent=ax.get_xlim() + ax.get_ylim(),
        )

    finish_plot(fig,ax,img,var,lev,filepath,uxda,ftime,plotdict,plotstart)


# =====================
# The available plot types, selected by the plot:type setting
# =====================
PLOT_FUNCTIONS = {
    "map": plotit,
    "cross_section": plot_cross_section,
}


def _extract_timestring(ds: xr.Dataset, filepath: str, file_index: int = 0) -> str:
    """
    Extract a '%Y-%m-%d_%H:%M:%S' timestring from a single raw xarray Dataset.
    Tries: CF Time variable, filename parsing, dummy fallback.
    Only called from open_ux_subset() when xtime is absent.
    file_index is used to generate unique dummy timestamps when all else fails.
    """
    # CF-compliant Time variable with units attribute
    if "Time" in ds.variables and "units" in ds["Time"].attrs:
        units = ds["Time"].attrs["units"]
        match = re.match(r'(\w+) since (\d{4}-\d{2}-\d{2}[T _]\d{2}:\d{2}:\d{2})', units)
        if match:
            unit_type = match.group(1).lower()
            base_time = datetime.strptime(
                match.group(2).replace("T", " ").replace("_", " "), "%Y-%m-%d %H:%M:%S"
            )
            multipliers = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
            if unit_type in multipliers:
                offset_secs = float(ds["Time"].isel(Time=0).values) * multipliers[unit_type]
                valid_time = base_time + timedelta(seconds=offset_secs)
                logger.debug(f"Decoded time from CF Time variable with units: {units}")
                return valid_time.strftime("%Y-%m-%d_%H:%M:%S")

    # Parse time from filename (e.g. diag.2025-06-07_00.00.00.nc)
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2})\.(\d{2})\.(\d{2})', os.path.basename(filepath))
    if match:
        timestring = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
        logger.info(f"No time variable found; extracted time from filename: {timestring}")
        return timestring

    dummy = datetime(1900, 1, 1) + timedelta(hours=file_index)
    logger.warning(f"Could not determine valid time for {filepath}; using dummy time value {dummy.strftime('%Y-%m-%d_%H:%M:%S')}")
    return dummy.strftime("%Y-%m-%d_%H:%M:%S")


def setup_args(config_d: dict,uxds: ux.UxDataset):
    """
    Sets up the argument list for plotit to allow for parallelization with Python starmap
    """
    args = []

    for var in config_d["dataset"]["vars"]:
        # Update each variable's plot settings dictionary
        plotdict=copy.copy(config_d["plot"])
        if update_dict:=config_d["dataset"]["vars"][var].get("plot"):
            plotdict=deep_merge(plotdict,update_dict)
        config_d["dataset"]["vars"][var]["plot"]=plotdict

        vardict=config_d["dataset"]["vars"][var]
        # Plot all levels by default
        if vardict.get("lev") is None:
            vardict["lev"]="all"
        vardict["vertcoord"]=vardict.get("vertcoord","nVertLevels")
        # Cross-sections fan out over named transects rather than vertical levels. The
        # transect name travels in the "lev" slot, so that the {lev} substitution still
        # gives each transect its own output file without any further plumbing.
        if vardict["plot"].get("type") == "cross_section":
            levels = list(vardict["plot"]["cross_section"]["transects"])
            if not levels:
                raise ValueError(f"No plot:cross_section:transects defined for variable {var}")
        elif vardict["lev"] in [ ["all"], "all" ]:
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

        if "Time" in uxds[var].dims:
            for lev in levels:
                for i in range(uxds.sizes["Time"]):
                    timestring = "".join(uxds["xtime"].isel(Time=i).values.astype(str)).strip()
                    args.append( (config_d,uxds,var,lev,i,timestring) )
        else:
            logger.debug(f"{var} has no time dimension")
            timestring = "".join(uxds["xtime"].isel(Time=0).values.astype(str)).strip()
            for lev in levels:
                args.append( (config_d,uxds,var,lev,-1,timestring) )

    return args


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
        raise SystemExit
    logger.debug(f"Reading options file {config}")
    try:
        user_config = uwconfig.get_yaml_config(config=config)
    except Exception as e:
        logger.critical(e)
        logger.critical(f"Error reading {config}, check above error trace for details")
        raise SystemExit

    if user_config.get("data"):
        logger.error(f"Found 'data' section in {config}, which is invalid.")
        raise ValueError("You may be using an old config file format; see config_plot.yaml.example")
    # Update the dict read from defaults file with the dict read from user config file
    expt_config.update_from(user_config)

    if not expt_config["dataset"].get("gridfile"):
        expt_config["dataset"]["gridfile"]=""

    # Perform consistency checks

    if isinstance(expt_config["plot"]["title"],str):
        raise TypeError("plot:title should be a dictionary, not a string\n"\
                        "Adjust your config.yaml accordingly. See default_options.yaml for details.")

    if not (None in expt_config["plot"]["projection"]["lonrange"] or None in expt_config["plot"]["projection"]["latrange"]):
        lon0=expt_config["plot"]["projection"]["lonrange"][0]
        lon1=expt_config["plot"]["projection"]["lonrange"][1]
        lat0=expt_config["plot"]["projection"]["latrange"][0]
        lat1=expt_config["plot"]["projection"]["latrange"][1]
        if lon0 >= lon1:
            raise ValueError(f"plot:projection:lonrange first value {lon0} >= second value {lon1}")
        if lat0 >= lat1:
            raise ValueError(f"plot:projection:latrange first value {lat0} >= second value {lat1}")

    logger.debug("Expanding references to other variables and Jinja templates")
    expt_config.dereference()
    return expt_config


def worker_init(debug=False):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_logging(debug=debug)


if __name__ == "__main__":

    scriptstart = time.time()
    parser = argparse.ArgumentParser(
        description="Script for producing plot files from MPAS input and/or output in native NetCDF format"
    )
    parser.add_argument('-c', '--config', type=str, default='config_plot.yaml',
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
    dataset=load_full_dataset(expt_config["dataset"],
                              optional_vars=cross_section_vars(expt_config))

    logger.debug(f'{dataset=}')

    # Set up plotit() arguments
    logger.info('Setting up plot tasks')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    plotargs=setup_args(expt_config,dataset)

    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    logger.debug(f"{plotargs=}")
    # Make the plots!
    if args.procs > 1:
        logger.info(f"Plotting in parallel with {args.procs} tasks")
        # This is needed to avoid some kind of file handle clobbering mumbo-jumbo with netCDF
        multiprocessing.set_start_method("spawn")
        try:
            with multiprocessing.Pool(processes=args.procs,initializer=worker_init,initargs=(args.debug,)) as pool:
                result = pool.starmap_async(plotithandler, plotargs)
                while True:
                    try:
                        result.get(timeout=1)
                        break
                    except multiprocessing.TimeoutError:
                        continue
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received; terminating workers...")
            pool.terminate()
            pool.join()
            sys.exit(0)
    else:
        i=0
        for instance in plotargs:
            plotithandler(*plotargs[i])
            i+=1

    logger.info(f"Done plotting all figures! Total time {time.time()-scriptstart} seconds")
