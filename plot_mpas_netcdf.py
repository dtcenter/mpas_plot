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
import traceback
import time
from datetime import datetime

import gc, psutil
proc = psutil.Process(os.getpid())

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs


print("Importing uxarray; this may take a while...")
import uxarray as ux
import xarray as xr

import uwtools.api.config as uwconfig
import custom_functions

from plot_functions import set_map_projection, set_patterns_and_outfile


logger = logging.getLogger(__name__)

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
def load_full_dataset(dsconf):
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
    ds = open_ux_subset(dsconf["gridfile"], dsconf["files"], list(readvars))

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
        try:
            plotit(config_d['dataset']['vars'][var],field,var,lev,config_d['dataset']['files'][0],ftime_dt)
        except Exception as e:
            logger.error(f'Could not plot variable {var}, level {lev}')
            logger.debug(f"Arguments to plotit():\n{config_d['dataset']['vars'][var]}\n{field=}\n"\
                         f"{var=}\n{lev=}\n{config_d['dataset']['files'][0]=}\n{ftime_dt=}"\
                         f"{config_d['dataset']['vars'][var]['plot']=}")
            logger.error(f"{traceback.print_tb(e.__traceback__)}:")
            logger.error(f"{type(e).__name__}:")
            logger.error(e)
    else:
        try:
            plotit(config_d['dataset']['vars'][var],field.isel(Time=timeint),var,lev,config_d['dataset']['files'][timeint],ftime_dt)
        except Exception as e:
            logger.error(f'Could not plot variable {var}, level {lev}, time {timeint}')
            logger.debug(f"Arguments to plotit():\n{config_d['dataset']['vars'][var]}\n{field.isel(Time=timeint)=}\n"\
                         f"{var=}\n{lev=}\n{config_d['dataset']['files'][timeint]=}\n{ftime_dt=}"\
                         f"{config_d['dataset']['vars'][var]['plot']=}")
            logger.error(f"{traceback.print_tb(e.__traceback__)}:")
            logger.error(f"{type(e).__name__}:")
            logger.error(e)


def plotit(vardict: dict,uxda: ux.UxDataArray,var: str,lev: int,filepath: str,ftime) -> None:
    """
    The main program that makes the plot(s)
    Args:
        vardict      (dict): A dictionary containing experiment settings specific to the variable being plotted
        uxds (ux.UxDataArray): A ux.UxDataArray object containing the data to be plotted and grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        ftime    (datetime): The forecast valid time as a datetime object

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

    logger.info(f"{varslice=}")

    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    if plotdict["periodic_bdy"]:
        logger.info("Creating polycollection with periodic_bdy=True")
        logger.info("NOTE: This option can be very slow for large domains")
        pc=varslice.to_polycollection(periodic_elements='split')
    else:
        pc=varslice.to_polycollection()
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")

    pc.set_antialiased(False)

    # Handle color mapping
    cmapname=plotdict["colormap"]
    if cmapname in plt.colormaps():
        cmap=mpl.colormaps[cmapname]
        pc.set_cmap(plotdict["colormap"])
    elif os.path.exists(colorfile:=f"colormaps/{cmapname}.yaml"):
        cmap_settings = uwconfig.get_yaml_config(config=colorfile)
        #Overwrite additional settings specified in colormap file
        logger.info(f"Color map {cmapname} selected; using custom settings from {colorfile}")
        for setting in cmap_settings:
            if setting == "colors":
                # plot:colors is a list of color values for the custom colormap and is handled separately
                continue
            logger.debug(f"Overwriting config {setting} with custom value {cmap_settings[setting]} from {colorfile}")
            plotdict[setting]=cmap_settings[setting]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name="custom",colors=cmap_settings["colors"])
    else:
        raise ValueError(f"Requested color map {cmapname} is not valid")

    if not plotdict["plot_over"]:
        cmap.set_over(alpha=0)
    if not plotdict["plot_under"]:
        cmap.set_under(alpha=0)
    pc.set_cmap(cmap)

    # Set up map projection properties
    logger.debug(plotdict["projection"])
    proj=set_map_projection(plotdict["projection"])

    fig, ax = plt.subplots(1, 1, figsize=(plotdict["figwidth"],
                           plotdict["figheight"]), dpi=plotdict["dpi"],
                           constrained_layout=True,
                           subplot_kw=dict(projection=proj))

    # Check the valid file formats supported for this figure
    validfmts=fig.canvas.get_supported_filetypes()

    logger.debug(f"{plotdict['projection']['lonrange']=}\n{plotdict['projection']['latrange']=}")
    if None in plotdict["projection"]["lonrange"] or None in plotdict["projection"]["latrange"]:
        logger.info('One or more latitude/longitude range values were not set; plotting full projection')
    else:
        ax.set_extent([plotdict["projection"]["lonrange"][0], plotdict["projection"]["lonrange"][1], plotdict["projection"]["latrange"][0], plotdict["projection"]["latrange"][1]], crs=ccrs.PlateCarree())

    if None not in [ plotdict["vmin"], plotdict["vmax"]]:
        pc.set_clim(plotdict["vmin"],plotdict["vmax"])

    #Plot political boundaries if requested
    if plotdict.get("boundaries"):
        pb=plotdict["boundaries"]
        if pb.get("enable"):
            # Users can set these values to scalars or lists; if scalar provided, re-format to list with three identical values
            for setting in ["color", "linewidth", "scale"]:
                if type(pb[setting]) is not list:

                    pb[setting]=[pb[setting],pb[setting],pb[setting]]
            if pb["detail"]==2:
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

    # Create a dict of substitutable patterns to make string substitutions easier, and determine output filename
    patterns,outfile,fmt = set_patterns_and_outfile(validfmts,var,lev,filepath,uxda,ftime,plotdict)

    pc.set_edgecolor(plotdict['edges']['color'])
    pc.set_linewidth(plotdict['edges']['width'])
    pc.set_transform(ccrs.PlateCarree())

    logger.debug("Adding collection to plot axes")
    if plotdict["projection"]["projection"] != "PlateCarree":
        logger.info(f"Interpolating to {plotdict['projection']['projection']} projection; this may take a while...")
    if None in plotdict["projection"]["lonrange"] or None in plotdict["projection"]["latrange"]:
        coll = ax.add_collection(pc, autolim=True)
        ax.autoscale()
    else:
        coll = ax.add_collection(pc)

    logger.debug("Configuring plot title")
    if plottitle:=plotdict["title"].get("text"):
        plt.title(plottitle.format_map(patterns), wrap=True, fontsize=plotdict["title"]["fontsize"])
    else:
        logger.warning("No 'text' field for title specified, creating plot with no title")

    logger.debug("Configuring plot colorbar")
    if plotdict.get("colorbar"):
        if plotdict.get("colorbar").get("enable"):
            cb = plotdict["colorbar"]
            cbar = plt.colorbar(coll,ax=ax,orientation=cb["orientation"])
            if cb.get("label"):
                cbar.set_label(cb["label"].format_map(patterns), fontsize=cb["fontsize"])
                cbar.ax.tick_params(labelsize=cb["fontsize"])

    # Make sure any subdirectories exist before we try to write the file
    if os.path.dirname(outfile):
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
    logger.debug(f"Saving plot {outfile}")
    plt.savefig(outfile,format=fmt)
    plt.close(fig)
    logger.info(f"Done saving plot {outfile}. Plot generation {time.time()-plotstart} seconds")


def deep_merge(dict1, dict2):
    """Update dictionary 1 with dictionary 2, including nested dictionaries."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result


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

        # Extract time strings
        # If multiple timesteps in a dataset, loop over times
        if "Time" in uxds[var].dims:
            times=[]
            for i in range(uxds.sizes["Time"]):
                logger.debug(f"Plotting time step {i}")
                if "xtime" in uxds:
                    times.append("".join(uxds["xtime"].isel(Time=i).values.astype(str)))
                else:
                    logger.warning(f"'xtime' variable not found in input file, using dummy time value")
                    times.append("1900-01-01_00:00:00")
            for lev in levels:
                i=0
                for timestring in times:
                    args.append( (config_d,uxds,var,lev,i,timestring) )
                    i+=1

        else:
            logger.debug(f"{var} has no time dimension")
            if "xtime" in uxds:
                logger.debug("Using first xtime value in file")
                timestring="".join(uxds["xtime"].isel(Time=0).values.astype(str))
            else:
                logger.warning(f"'xtime' variable not found in input file, using dummy time value")
                timestring="1900-01-01_00:00:00"

            for lev in levels:
                args.append( (config_d,uxds,var,lev,-1,timestring) )

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
    expt_config.update_from(user_config)

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


def worker_init(debug=False):
    setup_logging(debug=debug)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for plotting a custom field on the native MPAS grid from native NetCDF format files"
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
    dataset=load_full_dataset(expt_config["dataset"])

    logger.debug(f'{dataset=}')

    # Set up plotit() arguments
    logger.info('Setting up plot tasks')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    plotargs=setup_args(expt_config,dataset)

    logger.info('Submitting to starmap')
    logger.debug(f"Memory usage:{proc.memory_info().rss/1024**2} MB")
    logger.debug(f"{plotargs=}")
    # Make the plots!
    if args.procs > 1:
        logger.info(f"Plotting in parallel with {args.procs} tasks")
    # This is needed to avoid some kind of file handle clobbering mumbo-jumbo with netCDF
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=args.procs,initializer=worker_init,initargs=(args.debug,)) as pool:
        pool.starmap(plotithandler, plotargs)

    logger.info("Done plotting all figures!")
