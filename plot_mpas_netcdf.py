#!/usr/bin/env python3
"""
Script for plotting MPAS input and/or output in native NetCDF format"
"""
import argparse
import copy
import glob
import inspect
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


def load_dataset(logger,fn: str, gf: str = "") -> tuple[ux.UxDataset,ux.Grid]:
    """
    Program loads the dataset from the specified MPAS NetCDF data file and grid file and returns
    ux.UxDataset and ux.Grid objects. If grid file not specified, it is assumed to be the same as
    the data file.
    """

    logger.info(f"Reading data from {fn}")
    if gf:
        logger.info(f"Reading grid from {gf}")
    else:
        gf=fn
    return ux.open_dataset(gf,fn),ux.open_grid(gf)


def plotit(logger,config_d: dict,uxds: ux.UxDataset,grid: ux.Grid,filepath: str,proj,parproc: int) -> None:
    """
    The main program that makes the plot(s)
    Args:
        config_d     (dict): A dictionary containing experiment settings
        uxds (ux.UxDataset): A ux.UxDataset object containing the data to be plotted
        grid      (ux.Grid): A ux.Grid object containing the unstructured grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        proj (cartopy.crs.proj): A cartopy projection
        parproc       (int): The number of processors available for generating plots in parallel

    Returns:
        None
    """

    filename=os.path.basename(filepath)
    #filename minus extension
    fnme=os.path.splitext(filename)[0]

    logger.debug(f"Available data variables:\n{list(uxds.data_vars.keys())}")

    for var in config_d["data"]["var"]:
        plotstart = time.time()
        if var not in list(uxds.data_vars.keys()):
            msg = f"{var=} is not a valid variable in {filepath}\n\n{uxds.data_vars}"
            raise ValueError(msg)
        logger.info(f"Plotting variable {var}")
        logger.debug(f"{uxds[var]=}")
        field=uxds[var]

        # If multiple timesteps in a file, only plot the first for now
        if "Time" in field.dims:
            logger.info("Plotting first time step")
            field=field.isel(Time=0)

        # Parse multiple levels for 3d fields
        # "sliced" is a dictionary of 2d slices of data we will plot. We use a dictionary
        # instead of a list because the levels may not necessarily be contiguous or monotonic
        sliced = {}
        if "nVertLevels" in field.dims:
            if config_d["data"]["lev"]:
                levs=config_d["data"]["lev"]
            logger.info(f'Plotting vertical level(s) {levs}')
            for lev in levs:
                sliced[lev]=field.isel(nVertLevels=lev)
        else:
            if len(config_d["data"]["lev"])>1:
                logger.warning(f"{var} has no vertical dimension; only plotting one level")
            elif config_d["data"]["lev"][0]>0:
                logger.warning(f"{var} has no vertical dimension; can not plot level {config_d['data']['lev'][0]} > 0")
                continue
            levs = [0]
            sliced[0]=field

        for lev in levs:
            logger.debug(f"For level {lev}, data slice to plot:\n{sliced[lev]}")

            if "n_face" not in field.dims:
                logger.warning(f"Variable {var} not face-centered, will interpolate to faces")
                sliced[lev] = sliced[lev].remap.inverse_distance_weighted(grid,
                                                                  remap_to='face centers', k=3)
                logger.debug(f"Data slice after interpolation:\n{sliced[lev]=}")

            if config_d["plot"]["periodic_bdy"]:
                logger.info("Creating polycollection with periodic_bdy=True")
                logger.info("NOTE: This option can be very slow for large domains")
                pc=sliced[lev].to_polycollection(periodic_elements='split')
            else:
                pc=sliced[lev].to_polycollection()

            pc.set_antialiased(False)

            pc.set_cmap(config_d["plot"]["colormap"])
            pc.set_clim(config_d["plot"]["vmin"],config_d["plot"]["vmax"])

            fig, ax = plt.subplots(1, 1, figsize=(config_d["plot"]["figwidth"],
                                   config_d["plot"]["figheight"]), dpi=config_d["plot"]["dpi"],
                                   constrained_layout=True,
                                   subplot_kw=dict(projection=proj))


            logger.debug(config_d["plot"]["projection"])
            logger.debug(f"{config_d['plot']['projection']['lonrange']=}\n{config_d['plot']['projection']['latrange']=}")
            if None in config_d["plot"]["projection"]["lonrange"] or None in config_d["plot"]["projection"]["latrange"]:
                logger.info('One or more latitude/longitude range values were not set; plotting full projection')
            else:
                ax.set_extent([config_d["plot"]["projection"]["lonrange"][0], config_d["plot"]["projection"]["lonrange"][1], config_d["plot"]["projection"]["latrange"][0], config_d["plot"]["projection"]["latrange"][1]], crs=ccrs.PlateCarree())

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
                    logger.info("Counties only available at 10m resolution")
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
                    logger.warning("No output file format specified; defaulting to PNG")
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
                "proj": config_d["plot"]["projection"]["projection"],
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
                    logger.info(f"Overwriting existing file {outfile}")
                elif config_d["plot"]["exists"]=="abort":
                    raise FileExistsError(f"{outfile}\n"
                          "to change this behavior see plot:exists setting in config file")
                elif config_d["plot"]["exists"]=="rename":
                    logger.info(f"File exists: {outfile}")
                    i=0
                    # I love when I get to use the walrus operator :D
                    while os.path.isfile(outfile:=f"{outfnme}-{i}.{fmt}"):
                        logger.debug(f"File exists: {outfile}")
                        i+=1
                    logger.info(f"Saving to {outfile} instead")
                else:
                    raise ValueError(f"Invalid option: {config_d['plot']['exists']}")

            pc.set_edgecolor(config_d['plot']['edges']['color'])
            pc.set_linewidth(config_d['plot']['edges']['width'])
            pc.set_transform(ccrs.PlateCarree())

            logger.debug("Adding collection to plot axes")
            if config_d["plot"]["projection"]["projection"] != "PlateCarree":
                logger.info(f"Interpolating to {config_d['plot']['projection']['projection']} projection; this may take a while...")
            if None in config_d["plot"]["projection"]["lonrange"] or None in config_d["plot"]["projection"]["latrange"]:
                coll = ax.add_collection(pc, autolim=True)
                ax.autoscale()
            else:
                coll = ax.add_collection(pc)

            logger.debug("Configuring plot title")
            if plottitle:=config_d["plot"]["title"].get("text"):
                plt.title(plottitle.format_map(patterns), wrap=True, fontsize=config_d["plot"]["title"]["fontsize"])
            else:
                logger.warning("No text field for title specified, creating plot with no title")

            logger.debug("Configuring plot colorbar")
            if config_d["plot"].get("colorbar"):
                cb = config_d["plot"]["colorbar"]
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
            logger.debug(f"Done. Plot generation {time.time()-plotstart} seconds")


def set_map_projection(logger,confproj) -> ccrs.Projection:
    """
    Creates and returns a map projection based on the dictionary confproj, which contains the user
    settings for the desired map projection. Raises descriptive exception if invalid settings are
    specified.
    """

    proj=confproj["projection"]

    # Some projections have limits on the lat/lon range that can be plotted when specifying a map subset
    if not ( None in confproj['latrange']):
        if proj in ["Mercator","Miller","Mollweide","TransverseMercator"]:
            if confproj["latrange"][0]<-80:
                logger.warning(f"{proj} can not be plotted near poles, capping south latitude at -80˚")
                confproj["latrange"][0]=-79.999
            if confproj["latrange"][1]>80:
                logger.warning(f"{proj} can not be plotted near poles, capping north latitude at 80˚")
                confproj["latrange"][1]=80
        if proj in ["AlbersEqualArea","AzimuthalEquidistant","Gnomonic","Orthographic","Geostationary","LambertAzimuthalEqualArea","LambertConformal","NearsidePerspective","TransverseMercator"]:
            if confproj["latrange"][1]-confproj["latrange"][0] > 179:
                logger.debug(f"{confproj['latrange']=}")
                raise ValueError(f"{proj} projection limited to less than one hemisphere\n"\
                                  "change plot:projection:latrange to a smaller range")
    if not ( None in confproj['lonrange']):
        if proj in ["EckertI","EckertII","EckertIII","EckertIV","EckertV","EckertVI","EqualEarth","Mollweide","Sinusoidal"]:
            if confproj["lonrange"][1]-confproj["lonrange"][0] > 359:
                logger.warning(f"{proj} can not plot full globe, setting maximum longitude to minimum + 359˚")
                confproj["lonrange"][1]=confproj["lonrange"][0] + 359
        if proj in ["NorthPolarStereo","SouthPolarStereo","Stereographic"]:
            if confproj["lonrange"][1]-confproj["lonrange"][0] > 340:
                logger.warning(f"{proj} can not plot full globe, setting maximum longitude to minimum + 340˚")
                confproj["lonrange"][1]=confproj["lonrange"][0] + 340
        if proj in ["EquidistantConic"]:
            if confproj["lonrange"][1]-confproj["lonrange"][0] > 270:
                raise ValueError(f"{proj} projection limited to showing 3/4 of sphere\n"\
                                  "change plot:projection:lonrange to a smaller range")
        if proj in ["AlbersEqualArea","AzimuthalEquidistant","Gnomonic","Orthographic","Geostationary","LambertAzimuthalEqualArea","LambertConformal","NearsidePerspective","TransverseMercator"]:
            if confproj["lonrange"][1]-confproj["lonrange"][0] > 179:
                logger.debug(f"{confproj['lonrange']=}")
                raise ValueError(f"{proj} projection limited to less than one hemisphere\n"\
                                  "change plot:projection:lonrange to a smaller range")

    # Set some short var names
    proj=confproj["projection"]
    clat=confproj["central_lat"]
    clon=confproj["central_lon"]
    lat0=confproj["latrange"][0]
    lat1=confproj["latrange"][1]
    lon0=confproj["lonrange"][0]
    lon1=confproj["lonrange"][1]

    # If projection parameters are unset, set some sane defaults
    if clon is None:
        if None in confproj['lonrange']:
            clon = 0
        else:
            if lon0<lon1:
                clon = (lon0+lon1)/2
            else:
                clon = (lon0+lon1+360)/2
                if clon>180:
                    clon = clon-360
    if clat is None:
        if None in confproj['latrange']:
            clat = 0
        else:
            clat = (lat0+lat1)/2

    # Get all projection names and classes from cartopy.crs
    projection_dict = {}
    for pname, pcls in vars(ccrs).items():
        if inspect.isclass(pcls) and issubclass(pcls, ccrs.Projection) and pcls is not ccrs.Projection:
            if pname in ["AlbersEqualArea","EquidistantConic","LambertConformal"]:
                if pname == proj:
                    for setting in ["satellite_height"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                if None in confproj["standard_parallels"]:
                    projection_dict[pname] = pcls(central_latitude=clat,central_longitude=clon)
                else:
                    sp1,sp2=confproj["standard_parallels"]
                    projection_dict[pname] = pcls(central_latitude=clat,central_longitude=clon,standard_parallels=(sp1, sp2))
            elif pname in ["Geostationary"]:
                if pname == proj:
                    for setting in ["central_lat","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                if confproj["satellite_height"] is None:
                    projection_dict[pname] = pcls(central_longitude=clon)
                else:
                    projection_dict[pname] = pcls(central_longitude=clon,satellite_height=confproj["satellite_height"])
            elif pname in ["NearsidePerspective"]:
                if pname == proj:
                    for setting in ["standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                if confproj["satellite_height"] is None:
                    projection_dict[pname] = pcls(central_latitude=clat,central_longitude=clon)
                else:
                    projection_dict[pname] = pcls(central_latitude=clat,central_longitude=clon,satellite_height=confproj["satellite_height"])
            elif pname in ["AzimuthalEquidistant","Gnomonic","LambertAzimuthalEqualArea","ObliqueMercator","Orthographic","Stereographic","TransverseMercator"]:
                if pname == proj:
                    for setting in ["satellite_height","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                projection_dict[pname] = pcls(central_latitude=clat,central_longitude=clon)
            elif pname in ["Aitoff","EckertI","EckertII","EckertIII","EckertIV","EckertV","EckertVI","EqualEarth","Gnomonic","Hammer","InterruptedGoodeHomolosine","LambertCylindrical","Mercator","Miller","Mollweide","NorthPolarStereo","PlateCarree","Robinson","Sinusoidal","SouthPolarStereo"]:
                if pname == proj:
                    for setting in ["central_lat","satellite_height","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                projection_dict[pname] = pcls(central_longitude=clon)
            else:
                # Handle projections that require no args
                try:
                    if pname == proj:
                        for setting in ["central_lat","central_lon","satellite_height","standard_parallels"]:
                            if confproj[setting] is not None:
                                logger.info(f"{proj} does not use {setting}; ignoring")

                    projection_dict[pname] = pcls()  # Instantiate with default args
                except (TypeError,AttributeError):
                    # Skip non-projections, like base classes for other projections
                    continue

    for valid in projection_dict:
        if proj == valid:
            logger.debug(f"Setting up {valid} projection")
            logger.debug(f"{projection_dict[valid]=}\n{type(projection_dict[valid])}")
            return projection_dict[valid]

    raise ValueError(f"Invalid projection {proj} specified; valid options are:\n{list(projection_dict.keys())}")

def setup_logging(logfile: str = "log.mpas_plot", debug: bool = False) -> logging.Logger:
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

    formatter = logging.Formatter("%(asctime)s %(funcName)-16s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")

    # Set format for file handler
    fh = logging.FileHandler(logfile, mode='w')
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console)
    logger.addHandler(fh)

    logger.debug("Logging set up successfully")

    return logger


def setup_config(logger: logging.Logger, config: str, default: str="default_options.yaml") -> dict:
    """
    Function for reading in dictionary of configuration settings, and performing basic checks
    on those settings

    Args:
        config  (str) : The full path of the user config file
        default (str) : The full path of the default config file
        logger        : logging.Logger object

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

    # Perform consistency checks
    if not expt_config["data"].get("lev"):
        logger.debug("Level not specified in config, will use level 0 if multiple found")
        expt_config["data"]["lev"]=0

    # Check for old dictionary formats/deprecated options
    if expt_config["plot"].get("latrange") or expt_config["plot"].get("lonrange"):
        raise TypeError("plot:latrange and plot:lonrange have been moved to\n"\
                       "plot:projection:latrange and plot:projection:lonrange respectively\n"\
                       "Adjust your config.yaml accordingly. See default_options.yaml for details.")

    if isinstance(expt_config["plot"]["title"],str):
        raise TypeError("plot:title should be a dictionary, not a string\n"\
                        "Adjust your config.yaml accordingly. See default_options.yaml for details.")

    logger.debug("Expanding references to other variables and Jinja templates")
    expt_config.dereference()
    return expt_config

if __name__ == "__main__":

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

    logger=setup_logging(debug=args.debug)


    # Load settings from config file
    expt_config=setup_config(logger,args.config)

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
        dataset,grid=load_dataset(logger,f,expt_config["data"]["gridfile"])

        # Set up map projection properties
        proj=set_map_projection(logger,expt_config["plot"]["projection"])

        logger.debug(f"{dataset=}")
        logger.debug(f"{grid=}")
        logger.debug(f"{proj=}")

        # Make the plots!
        plotit(logger,expt_config,dataset,grid,f,proj,args.procs)
