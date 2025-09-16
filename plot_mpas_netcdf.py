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

from file_read import load_dataset

logger = logging.getLogger(__name__)

def setupargs(config_d: dict,uxds: ux.UxDataset,grid: ux.Grid,filepath: str):
    """
    Sets up the argument list for plotit to allow for parallelization with Python starmap
    """

    args = []
    variables = []
    levels = []

    if config_d["data"]["var"] in [ ["all"], "all" ]:
        for var in uxds:
            variables.append(var)
    else:
        for var in config_d["data"]["var"]:
            variables.append(var)

    if config_d["data"]["lev"] in [ ["all"], "all" ]:
        # Since there's no good way to query vertical levels in general, we query each variable until we find a 3d one, if there is one
        levels = [0]
        for var in variables:
            if "nVertLevels" in uxds[var].dims:
                levels = range(0,len(uxds[var]["nVertLevels"]))
    else:
        for lev in config_d["data"]["lev"]:
            levels.append(lev)

    # Set up map projection properties
    proj=set_map_projection(expt_config["plot"]["projection"])

    # Construct list of argument tuples for plotit()
    for var in variables:
        for lev in levels:
            args.append( (config_d,uxds,grid,var,lev,filepath,proj) )
    return args


def plotithandler(config_d: dict,uxds: ux.UxDataset,grid: ux.Grid,var: str,lev: int,filepath: str,proj) -> None:
    """
    A wrapper for plotit() that handles errors for Python multiprocessing, as well as preprocessing the
    UxDataSet into a UxDataArray with just the variable and timestep we want to plot
    """

    logger.info(f"Starting plotit() for {var=}, {lev=}")

    if var not in list(uxds.data_vars.keys()):
        msg = f"{var=} is not a valid variable in {filepath}\n\n{uxds.data_vars}"
        raise ValueError(msg)

    field=uxds[var]


#    if "Time" in uxds.dims and uxds.sizes.get("Time", 0) > 1:
#        logger.info("Dataset has multiple timesteps — looping over Time")
#        for i in range(uxds.sizes["Time"]):
#            # select the i-th time as an xarray.Dataset
#            ds_i = uxds.isel(Time=i)
#
#            # re-wrap as UxDataset so grid info stays attached
#            uxds = ux.UxDataset.from_xarray(ds_i, uxgrid=uxds.uxgrid)
#
#    else:
#        logger.info("Single timestep — plotting directly")

    # If multiple timesteps in a dataset, loop over times
    if "Time" in field.dims:
        for i in range(uxds.sizes["Time"]):
            logger.info(f"Plotting time step {i}")
            try:
                plotit(config_d,field.isel(Time=i),grid,var,lev,filepath,proj)
            except Exception as e:
                logger.error(f'Could not plot variable {var}, level {lev}, time {i}')
                logger.debug(f"Arguments to plotit():\n{config_d=}\n{field.isel(Time=i)=}\n{grid=}\n"\
                              f"{var=}\n{lev=}\n{filepath=}\n{proj=}")
                logger.error(f"{traceback.print_tb(e.__traceback__)}:")
                logger.error(f"{type(e).__name__}:")
                logger.error(e)


def plotit(config_d: dict,uxda: ux.UxDataArray,grid: ux.Grid,var: str,lev: int,filepath: str,proj) -> None:
    """
    The main program that makes the plot(s)
    Args:
        config_d     (dict): A dictionary containing experiment settings
        uxds (ux.UxDataArray): A ux.UxDataArray object containing the data to be plotted
        grid      (ux.Grid): A ux.Grid object containing the unstructured grid information
        filepath      (str): The filename of the input data that was read into the ux objects
        proj (cartopy.crs.proj): A cartopy projection

    Returns:
        None
    """

    plotstart = time.time()

    if "nVertLevels" in uxda.dims:
        varslice = uxda.isel(nVertLevels=lev)
    else:
        if lev > 0:
            logger.error(f"Variable {var} only has one vertical level; can not plot {lev=}")
            return
        varslice = uxda

    if "n_face" not in uxda.dims:
        logger.warning(f"Variable {var} not face-centered, will interpolate to faces")
        varslice = varslice.remap.inverse_distance_weighted(grid,
                                                          remap_to='face centers', k=3)
        logger.debug(f"Data slice after interpolation:\n{varslice=}")

    if config_d["plot"]["periodic_bdy"]:
        logger.info("Creating polycollection with periodic_bdy=True")
        logger.info("NOTE: This option can be very slow for large domains")
        pc=varslice.to_polycollection(periodic_elements='split')
    else:
        pc=varslice.to_polycollection()

    pc.set_antialiased(False)

    # Handle color mapping
    cmapname=config_d["plot"]["colormap"]
    if cmapname in plt.colormaps():
        cmap=mpl.colormaps[cmapname]
        pc.set_cmap(config_d["plot"]["colormap"])
    elif os.path.exists(colorfile:=f"colormaps/{cmapname}.yaml"):
        cmap_settings = uwconfig.get_yaml_config(config=colorfile)
        #Overwrite additional settings specified in colormap file
        logger.info(f"Color map {cmapname} selected; using custom settings from {colorfile}")
        for setting in cmap_settings:
            if setting == "colors":
                # plot:colors is a list of color values for the custom colormap and is handled separately
                continue
            logger.debug(f"Overwriting config {setting} with custom value {cmap_settings[setting]} from {colorfile}")
            config_d["plot"][setting]=cmap_settings[setting]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name="custom",colors=cmap_settings["colors"])
    else:
        raise ValueError(f"Requested color map {cmapname} is not valid")

    if not config_d["plot"]["plot_over"]:
        cmap.set_over(alpha=0)
    if not config_d["plot"]["plot_under"]:
        cmap.set_under(alpha=0)
    pc.set_cmap(cmap)

    fig, ax = plt.subplots(1, 1, figsize=(config_d["plot"]["figwidth"],
                           config_d["plot"]["figheight"]), dpi=config_d["plot"]["dpi"],
                           constrained_layout=True,
                           subplot_kw=dict(projection=proj))

    # Check the valid file formats supported for this figure
    validfmts=fig.canvas.get_supported_filetypes()


    logger.debug(config_d["plot"]["projection"])
    logger.debug(f"{config_d['plot']['projection']['lonrange']=}\n{config_d['plot']['projection']['latrange']=}")
    if None in config_d["plot"]["projection"]["lonrange"] or None in config_d["plot"]["projection"]["latrange"]:
        logger.info('One or more latitude/longitude range values were not set; plotting full projection')
    else:
        ax.set_extent([config_d["plot"]["projection"]["lonrange"][0], config_d["plot"]["projection"]["lonrange"][1], config_d["plot"]["projection"]["latrange"][0], config_d["plot"]["projection"]["latrange"][1]], crs=ccrs.PlateCarree())

    if None not in [ config_d["plot"]["vmin"], config_d["plot"]["vmax"]]:
        pc.set_clim(config_d["plot"]["vmin"],config_d["plot"]["vmax"])

    #Plot political boundaries if requested
    if config_d["plot"].get("boundaries"):
        pb=config_d["plot"]["boundaries"]
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
    if config_d["plot"].get("coastlines"):
        pcl=config_d["plot"]["coastlines"]
        if pcl.get("enable"):
            ax.add_feature(cfeature.NaturalEarthFeature(category='physical',color=pcl["color"],facecolor='none',
                           linewidth=pcl["linewidth"], scale=pcl["scale"], name='coastline'))
    #Plot lakes if requested
    if config_d["plot"].get("lakes"):
        pl=config_d["plot"]["lakes"]
        if pl.get("enable"):
            ax.add_feature(cfeature.NaturalEarthFeature(category='physical',edgecolor=pl["color"],facecolor='none',
                           linewidth=pl["linewidth"], scale=pl["scale"], name='lakes'))


    # Create a dict of substitutable patterns to make string substitutions easier, and determine output filename
    patterns,outfile,fmt = set_patterns_and_outfile(validfmts,var,lev,filepath,uxda,config_d["plot"])

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
        logger.warning("No 'text' field for title specified, creating plot with no title")

    logger.debug("Configuring plot colorbar")
    if config_d["plot"].get("colorbar"):
        if config_d["plot"].get("colorbar").get("enable"):
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
    logger.info(f"Done saving plot {outfile}. Plot generation {time.time()-plotstart} seconds")


def set_patterns_and_outfile(valid, var, lev, filepath, field, plotdict):
    """
    Create and return a dictionary of substituting patterns to make string substitutions easier
    in filenames and other text fields based on user input, using the python string builtin method
    format_map().
    Also, return string for the output filename and file format based on input files and user settings.
    """
    # Output plot filename
    outfile=plotdict['filename']
    if "." in os.path.basename(outfile):
        #output filename and extension
        outfnme,fmt=os.path.splitext(outfile)
        fmt=fmt[1:]
        if plotdict["format"] is not None:
            if fmt != plotdict["format"]:
                raise ValueError(f"plot:format is inconsistent with plot:filename\n" +
                                 f"{plotdict['format']=}\n" +
                                 f"{plotdict['filename']=}")
    else:
        outfnme=outfile
        if plotdict["format"] is not None:
            fmt=plotdict["format"]
        else:
            logger.warning("No output file format specified; defaulting to PNG")
            fmt='png'

    if fmt not in valid:
        raise ValueError(f"Invalid file format requested: {fmt}\n" +
                         f"Valid formats are:\n{valid}")

    # Input data filename
    filename=os.path.basename(filepath)
    #filename minus extension
    fnme=os.path.splitext(filename)[0]

    pattern_dict = {
        "var": var,
        "lev": lev,
        "units": "no_Units",
        "varln": "no_long_name",
        "filename": filename,
        "fnme": fnme,
        "proj": plotdict["projection"]["projection"],
        "date": "no_Time_dimension",
        "time": "no_Time_dimension"
    }
    if field.attrs.get("units"):
        pattern_dict.update({
            "units": field.attrs["units"],
        })
    if field.attrs.get("long_name"):
        pattern_dict.update({
            "varln": field.attrs["long_name"]
        })
#    if field.coords.get("Time"):
#        pattern_dict.update({
#            "date": field.coords['Time'].dt.strftime('%Y-%m-%d').item(),
#            "time": field.coords['Time'].dt.strftime('%H:%M:%S').item()
#        })


    # Check if the output file already exists, if so act according to plot:exists setting
    outfnme=outfnme.format_map(pattern_dict)
    outfile=f"{outfnme.format_map(pattern_dict)}.{fmt}"
    if os.path.isfile(outfile):
        if plotdict["exists"]=="overwrite":
            logger.info(f"Overwriting existing file {outfile}")
        elif plotdict["exists"]=="abort":
            raise FileExistsError(f"{outfile}\n"
                  "to change this behavior see plot:exists setting in config file")
        elif plotdict["exists"]=="rename":
            logger.info(f"File exists: {outfile}")
            i=0
            # I love when I get to use the walrus operator :D
            while os.path.isfile(outfile:=f"{outfnme}-{i}.{fmt}"):
                logger.debug(f"File exists: {outfile}")
                i+=1
            logger.info(f"Saving to {outfile} instead")
        else:
            raise ValueError(f"Invalid option: config_d['plot']['exists']={plotdict['exists']}")


    return pattern_dict, outfile, fmt


def set_map_projection(confproj) -> ccrs.Projection:
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
    valid= []
    for pname, pcls in vars(ccrs).items():
        if inspect.isclass(pcls) and issubclass(pcls, ccrs.Projection) and pcls is not ccrs.Projection:
            valid.append(pname)
            if pname == proj:
                if pname in ["AlbersEqualArea","EquidistantConic","LambertConformal"]:
                    for setting in ["satellite_height"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                    if None in confproj["standard_parallels"]:
                        return pcls(central_latitude=clat,central_longitude=clon)
                    else:
                        sp1,sp2=confproj["standard_parallels"]
                        return pcls(central_latitude=clat,central_longitude=clon,standard_parallels=(sp1, sp2))
                elif pname in ["Geostationary"]:
                    for setting in ["central_lat","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                    if confproj["satellite_height"] is None:
                        return pcls(central_longitude=clon)
                    else:
                        return pcls(central_longitude=clon,satellite_height=confproj["satellite_height"])
                elif pname in ["NearsidePerspective"]:
                    for setting in ["standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                    if confproj["satellite_height"] is None:
                        return pcls(central_latitude=clat,central_longitude=clon)
                    else:
                        return pcls(central_latitude=clat,central_longitude=clon,satellite_height=confproj["satellite_height"])
                elif pname in ["AzimuthalEquidistant","Gnomonic","LambertAzimuthalEqualArea","ObliqueMercator","Orthographic","Stereographic","TransverseMercator"]:
                    for setting in ["satellite_height","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                    return pcls(central_latitude=clat,central_longitude=clon)
                elif pname in ["Aitoff","EckertI","EckertII","EckertIII","EckertIV","EckertV","EckertVI","EqualEarth","Gnomonic","Hammer","InterruptedGoodeHomolosine","LambertCylindrical","Mercator","Miller","Mollweide","NorthPolarStereo","PlateCarree","Robinson","Sinusoidal","SouthPolarStereo"]:
                    for setting in ["central_lat","satellite_height","standard_parallels"]:
                        if confproj[setting] is not None:
                            logger.info(f"{proj} does not use {setting}; ignoring")
                    return pcls(central_longitude=clon)
                else:
                    # Handle projections that require no args
                    try:
                        for setting in ["central_lat","central_lon","satellite_height","standard_parallels"]:
                            if confproj[setting] is not None:
                                logger.info(f"{proj} does not use {setting}; ignoring")

                        return pcls()  # Instantiate with default args
                    except (TypeError,AttributeError):
                        # Skip non-projections, like base classes for other projections
                        continue

    raise ValueError(f"Invalid projection {proj} specified; valid options are:\n{valid}")

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

    setup_logging(debug=args.debug)


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


        logger.debug(f"{dataset=}")
        logger.debug(f"{grid=}")

        logger.debug(f"Available data variables:\n{list(dataset.data_vars.keys())}")

        # Set up plotit() arguments
        plotargs=setupargs(expt_config,dataset,grid,f)
        logger.debug(f"{plotargs=}")
        # Make the plots!
        if args.procs > 1:
            logger.info(f"Plotting in parallel with {args.procs} tasks")
        with Pool(processes=args.procs) as pool:
            pool.starmap(plotithandler, plotargs)

    logger.info("Done plotting all figures!")
