#!/usr/bin/env python3
"""
Script for plotting MPAS input and/or output in native NetCDF format"
"""
import argparse
import copy
import glob
import logging
import os
import sys
import time
import traceback
from multiprocessing import Pool

print("Importing uxarray; this may take a while...")
import uxarray as ux
import matplotlib as mpl
#This is needed to solve memory leak with large numbers of plots
#https://github.com/matplotlib/matplotlib/issues/20300
mpl.use('agg')
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs

import uwtools.api.config as uwconfig


def load_dataset(fn: str, gf: str = "") -> tuple[ux.UxDataset,ux.Grid]:
    """
    Program loads the dataset from the specified MPAS NetCDF data file and grid file and returns
    ux.UxDataset and ux.Grid objects. If grid file not specified, it is assumed to be the same as
    the data file.
    """

    logging.info(f"Reading data from {fn}")
    if gf:
        logging.info(f"Reading grid from {gf}")
    else:
        gf=fn
    return ux.open_dataset(gf,fn),ux.open_grid(gf)


def plotitparallel(config_d: dict,uxds: ux.UxDataset,grid: ux.Grid,filepath: str,variable: list=[],level: list=[]) -> None:
    """
    The a wrapper for plotit() used for calling it recursively and in parallel with use of starmap
    Args:
        config_d     (dict): A dictionary containing experiment settings
        uxds (ux.UxDataset): A ux.UxDataset object containing the data to be plotted
        grid      (ux.Grid): A ux.Grid object containing the unstructured grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        variable     (list): A one-item list, the variable name to plot. Used to overwrite the
                             value in data:var (used for recursion over all variables)
        level        (list): A one-item list, the vertical level to plot. If provided, overwrites
                             the value in data:lev (used for recursion over all levels)

    Returns:
        None

    """

    newconf = copy.deepcopy(config_d)
    if variable:
        newconf["data"]["var"]=variable
    if level:
        newconf["data"]["lev"]=level

    logging.debug(f'Trying to plot level {newconf["data"]["lev"]} for variable {newconf["data"]["var"]}')
    try:
        plotit(newconf,uxds,grid,filepath,1)
    except Exception as e:
        logging.warning(f'Could not plot variable {newconf["data"]["var"]}, level {newconf["data"]["lev"]}')
        logging.warning(f"{traceback.print_tb(e.__traceback__)}:")
        logging.warning(f"{type(e).__name__}:")
        logging.warning(e)



def plotit(config_d: dict,uxds: ux.UxDataset,grid: ux.Grid,filepath: str,parproc: int) -> None:
    """
    The main program that makes the plot(s)
    Args:
        config_d     (dict): A dictionary containing experiment settings
        uxds (ux.UxDataset): A ux.UxDataset object containing the data to be plotted
        grid      (ux.Grid): A ux.Grid object containing the unstructured grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        parproc       (int): The number of processors available for generating plots in parallel

    Returns:
        None
    """

    filename=os.path.basename(filepath)
    #filename minus extension
    fnme=os.path.splitext(filename)[0]

    logging.debug(f"Available data variables:\n{list(uxds.data_vars.keys())}")

    # To plot all variables, call plotit() recursively, trapping errors
    if config_d["data"]["var"]=="all":
        args = []
        for var in uxds:
            # Create argument tuples for each call to plotit() for use with starmap
            args.append( (config_d,uxds,grid,filepath,[var]) )
        if parproc > 1:
            with Pool(processes=parproc) as pool:
                pool.starmap(plotitparallel, args)
        else:
            i=0
            for instance in args:
                # '*' operator unpacks tuple for use as args
                plotitparallel(*args[i])
                i+=1
    # To plot all levels, call plotit() recursively, trapping errors
    elif config_d["data"]["lev"]=="all":
        args = []
        if "nVertLevels" in uxds[config_d["data"]["var"]].dims:
            levs = range(0,len(uxds[config_d["data"]["var"]]["nVertLevels"]))
        else:
            levs = [0]
        for lev in levs:
            # Create argument tuples for each call to plotit() for use with starmap
            args.append( (config_d,uxds,grid,filepath,config_d["data"]["var"],[lev]) )
        if parproc > 1:
            with Pool(processes=parproc) as pool:
                pool.starmap(plotitparallel, args)
        else:
            i=0
            for instance in args:
                plotitparallel(*args[i])
                i+=1
    else:
        # Convert variables that may be a list into one if they aren't already
        if not isinstance(config_d["data"]["var"], list):
            config_d["data"]["var"] = [ config_d["data"]["var"] ]
        if not isinstance(config_d["data"]["lev"], list):
            config_d["data"]["lev"] = [ config_d["data"]["lev"] ]
        for var in config_d["data"]["var"]:
            plotstart = time.time()
            if var not in list(uxds.data_vars.keys()):
                msg = f"{var=} is not a valid variable in {filepath}\n\n{uxds.data_vars}"
                raise ValueError(msg)
            logging.info(f"Plotting variable {var}")
            logging.debug(f"{uxds[var]=}")
            field=uxds[var]

            # If multiple timesteps in a file, only plot the first for now
            if "Time" in field.dims:
                logging.info("Plotting first time step")
                field=field.isel(Time=0)

            # Parse multiple levels for 3d fields
            # "sliced" is a dictionary of 2d slices of data we will plot. We use a dictionary
            # instead of a list because the levels may not necessarily be contiguous or monotonic
            sliced = {}
            if "nVertLevels" in field.dims:
                if config_d["data"]["lev"]:
                    levs=config_d["data"]["lev"]
                logging.info(f'Plotting vertical level(s) {levs}')
                for lev in levs:
                    sliced[lev]=field.isel(nVertLevels=lev)
            else:
                levs = [0]
                sliced[0]=field

            for lev in levs:
                logging.debug(f"For level {lev}, data slice to plot:\n{sliced[lev]}")

                if "n_face" not in field.dims:
                    logging.warning(f"Variable {var} not face-centered, will interpolate to faces")
                    sliced[lev] = sliced[lev].remap.inverse_distance_weighted(grid,
                                                                      remap_to='face centers', k=3)
                    logging.debug(f"Data slice after interpolation:\n{sliced[lev]=}")

                if config_d["plot"]["periodic_bdy"]:
                    logging.info("Creating polycollection with periodic_bdy=True")
                    logging.info("NOTE: This option can be very slow for large domains")
                    pc=sliced[lev].to_polycollection(periodic_elements='split')
                else:
                    pc=sliced[lev].to_polycollection()

                pc.set_antialiased(False)

                pc.set_cmap(config_d["plot"]["colormap"])
                pc.set_clim(config_d["plot"]["vmin"],config_d["plot"]["vmax"])

                fig, ax = plt.subplots(1, 1, figsize=(config_d["plot"]["figwidth"],
                                       config_d["plot"]["figheight"]), dpi=config_d["plot"]["dpi"],
                                       constrained_layout=True,
                                       subplot_kw=dict(projection=ccrs.PlateCarree()))


                ax.set_xlim((config_d["plot"]["lonrange"][0],config_d["plot"]["lonrange"][1]))
                ax.set_ylim((config_d["plot"]["latrange"][0],config_d["plot"]["latrange"][1]))

                #Plot coastlines if requested
                if config_d["plot"]["coastlines"]:
                    ax.add_feature(cfeature.NaturalEarthFeature(category='physical',
                                   **config_d["plot"]["coastlines"], name='coastline'))
                if config_d["plot"]["boundaries"]:
                    if config_d["plot"]["boundaries"]["detail"]==0:
                        name='admin_0_countries'
                    elif config_d["plot"]["boundaries"]["detail"]==1:
                        name='admin_1_states_provinces'
                    elif config_d["plot"]["boundaries"]["detail"]==2:
                        logging.info("Counties only available at 10m resolution")
                        config_d["plot"]["boundaries"]["scale"]='10m'
                        name='admin_2_counties'
                    else:
                        raise ValueError(f'Invalid value for {config_d["plot"]["boundaries"]["detail"]=}')
                    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural',
                                   scale=config_d["plot"]["boundaries"]["scale"], facecolor='none',
                                   linewidth=0.2, name=name))

                #Set file format based on filename or manual settings
                validfmts=fig.canvas.get_supported_filetypes()
                outfile=config_d['plot']['filename']
                if "." in os.path.basename(outfile):
                    #output filename and extension
                    outfnme,fmt=os.path.splitext(outfile)
                    fmt=fmt[1:]
                    if config_d["plot"]["format"] is not None:
                        if fmt != config_d["plot"]["format"]:
                            raise ValueError(f"plot:format is inconsistent with plot:filename\n" +
                                             f"{config_d['plot']['format']=}\n" +
                                             f"{config_d['plot']['filename']=}")
                else:
                    outfnme=outfile
                    if config_d["plot"]["format"] is not None:
                        fmt=config_d["plot"]["format"]
                    else:
                        logging.warning("No output file format specified; defaulting to PNG")
                        fmt='png'

                if fmt not in validfmts:
                    raise ValueError(f"Invalid file format requested: {fmt}\n" +
                                     f"Valid formats are:\n{validfmts}")

                # Create a dict of substitutable patterns to make string substitutions easier
                # using the python string builtin method format_map()
                patterns = {
                    "var": var,
                    "lev": lev,
                    "units": field.attrs["units"],
                    "varln": field.attrs["long_name"],
                    "filename": filename,
                    "fnme": fnme,
                }
                if field.coords.get("Time"):
                    patterns.update({
                        "date": field.coords['Time'].dt.strftime('%Y-%m-%d').item(),
                        "time": field.coords['Time'].dt.strftime('%H:%M:%S').item()
                    })
                else:
                    patterns.update({
                        "date": "no_Time_dimension",
                        "time": "no_Time_dimension"
                    })

                # Check if the file already exists, if so act according to plot:exists setting
                outfnme=outfnme.format_map(patterns)
                outfile=f"{outfnme.format_map(patterns)}.{fmt}"
                if os.path.isfile(outfile):
                    if config_d["plot"]["exists"]=="overwrite":
                        logging.info(f"Overwriting existing file {outfile}")
                    elif config_d["plot"]["exists"]=="abort":
                        raise FileExistsError(f"{outfile}\n"
                              "to change this behavior see plot:exists setting in config file")
                    elif config_d["plot"]["exists"]=="rename":
                        logging.info(f"File exists: {outfile}")
                        i=0
                        # I love when I get to use the walrus operator :D
                        while os.path.isfile(outfile:=f"{outfnme}-{i}.{fmt}"):
                            logging.debug(f"File exists: {outfile}")
                            i+=1
                        logging.info(f"Saving to {outfile} instead")
                    else:
                        raise ValueError(f"Invalid option: {config_d['plot']['exists']}")

                coll = ax.add_collection(pc)

                if plottitle:=config_d["plot"]["title"].get("text"):
                    plt.title(plottitle.format_map(patterns), wrap=True, fontsize=config_d["plot"]["title"]["fontsize"])
                else:
                    logging.warning("No text field for title specified, creating plot with no title")

                # Handle colorbar
                if config_d["plot"].get("colorbar"):
                    cb = config_d["plot"]["colorbar"]
                    cbar = plt.colorbar(coll,ax=ax,orientation=cb["orientation"])
                    if cb.get("label"):
                        cbar.set_label(cb["label"].format_map(patterns), fontsize=cb["fontsize"])
                        cbar.ax.tick_params(labelsize=cb["fontsize"])

                # Make sure any subdirectories exist before we try to write the file
                if os.path.dirname(outfile):
                    os.makedirs(os.path.dirname(outfile),exist_ok=True)
                logging.debug(f"Saving plot {outfile}")
                plt.savefig(outfile,format=fmt)
                plt.close(fig)
                logging.debug(f"Done. Plot generation {time.time()-plotstart} seconds")



def setup_logging(logfile: str = "log.generate_FV3LAM_wflow", debug: bool = False) -> logging.Logger:
    """
    Sets up logging, printing high-priority (INFO and higher) messages to screen, and printing all
    messages with detailed timing and routine info in the specified text file.

    If debug = True, print all messages to both screen and log file.
    """
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console = logging.StreamHandler()
    fh = logging.FileHandler(logfile)

    # Set the log level for each handler
    if debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)  # Log DEBUG and above to the file

    formatter = logging.Formatter("%(name)-22s %(levelname)-8s %(message)s")

    # Set format for file handler
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console)
    logger.addHandler(fh)

    logger.debug("Logging set up successfully")

    return logger


def setup_config(config: str, default: str="default_options.yaml") -> dict:
    """
    Function for reading in dictionary of configuration settings, and performing basic checks
    on those settings

    Args:
        config  (str) : The full path of the user config file
        default (str) : The full path of the default config file
        debug   (bool): Enable extra output for debugging
    Returns:
        dict: A dictionary of the configuration settings after applying defaults and user settings,
              as well as some basic consistency checks
    """
    logging.debug(f"Reading defaults file {default}")
    try:
        expt_config = uwconfig.get_yaml_config(config=default)
    except Exception as e:
        logging.critical(e)
        logging.critical(f"Error reading {config}, check above error trace for details")
        sys.exit(1)
    logging.debug(f"Reading options file {config}")
    try:
        user_config = uwconfig.get_yaml_config(config=config)
    except Exception as e:
        logging.critical(e)
        logging.critical(f"Error reading {config}, check above error trace for details")
        sys.exit(1)

    # Update the dict read from defaults file with the dict read from user config file
    expt_config.update_values(user_config)

    # Perform consistency checks
    if not expt_config["data"].get("lev"):
        logging.debug("Level not specified in config, will use level 0 if multiple found")
        expt_config["data"]["lev"]=0

    logging.debug("Expanding references to other variables and Jinja templates")
    expt_config.dereference()
    return expt_config

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Script for plotting MPAS input and/or output in native NetCDF format"
    )
    parser.add_argument('-c', '--config', type=str, default='config_plot.yaml',
                        help='File used to specify plotting options')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Script will be run in debug mode with more verbose output')
    parser.add_argument('-p', '--procs', type=int, default=1,
                        help='Number of processors for generating plots in parallel')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


    # Load settings from config file
    expt_config=setup_config(args.config)

    if os.path.isfile(expt_config["data"]["filename"]):
        files = [expt_config["data"]["filename"]]
    elif glob.glob(expt_config["data"]["filename"]):
        files = sorted(glob.glob(expt_config["data"]["filename"]))
    elif isinstance(expt_config["data"]["filename"], list):
        files = expt_config["data"]["filename"]
    else:
        raise FileNotFoundError(f"Invalid filename(s) specified:\n{expt_config['data']['filename']}")

    if not expt_config["data"].get("gridfile"):
        expt_config["data"]["gridfile"]=""

    for f in files:
        # Open specified file and load dataset
        dataset,grid=load_dataset(f,expt_config["data"]["gridfile"])

        logging.debug(f"{dataset=}")
        logging.debug(f"{grid=}")
        # Make the plots!
        plotit(expt_config,dataset,grid,f,args.procs)
