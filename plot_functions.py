#!/usr/bin/env python3
"""
Module for various plotting functions
"""
import inspect
import logging
import os
import time
import traceback

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import uwtools.api.config as uwconfig

logger = logging.getLogger(__name__)


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


def get_colormap(plotdict):
    """
    Resolve the plot:colormap setting into a Matplotlib colormap object.

    The name may be either a standard Matplotlib colormap or the name of a custom colormap
    file under colormaps/; custom colormap files may also override other plot settings, so
    plotdict is updated in place.

    Args:
        plotdict (dict): The plot settings for the variable being plotted; modified in place
    Returns:
        A Matplotlib colormap object
    """
    cmapname=plotdict["colormap"]
    if cmapname in plt.colormaps():
        if bins:=plotdict.get("colorbins"):
            cmap=plt.get_cmap(cmapname, bins)
        else:
            cmap=mpl.colormaps[cmapname]
    elif os.path.exists(colorfile:=f"colormaps/{cmapname}.yaml"):
        cmap_settings = uwconfig.get_yaml_config(config=colorfile)
        #Overwrite additional settings specified in colormap file
        logger.info(f"Color map {cmapname} selected; using custom settings from {colorfile}")
        for setting in cmap_settings:
            if setting == "colors":
                # plot:colors is a list of color values for the custom colormap and is handled separately
                continue
            logger.debug(f"Overwriting config {setting} with custom value {cmap_settings[setting]} from {colorfile}")
            if isinstance(plotdict.get(setting),dict):
                plotdict[setting]=deep_merge(plotdict[setting],cmap_settings[setting])
            else:
                plotdict[setting]=cmap_settings[setting]
        if not (colorbins:=plotdict.get("colorbins")):
            colorbins=256
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name="custom",colors=cmap_settings["colors"],N=colorbins)
    else:
        raise ValueError(f"Requested color map {cmapname} is not valid")

    if not plotdict["plot_over"]:
        cmap.set_over(alpha=0)
    if not plotdict["plot_under"]:
        cmap.set_under(alpha=0)

    return cmap


def finish_plot(fig, ax, img, var, lev, filepath, field, ftime, plotdict, plotstart):
    """
    Apply the color limits, title, and colorbar settings shared by all plot types, then write
    the figure to disk and close it.

    Args:
        fig    (Figure): The figure to write
        ax       (Axes): The axes the data was plotted on
        img (ScalarMappable): The object returned by the plotting call, for clim and colorbar
        var       (str): Name of the variable being plotted
        lev  (int | str): Vertical level, or the transect name for cross-section plots
        filepath  (str): The filename of the input data that was read into the ux objects
        field (ux.UxDataArray): The field being plotted, for the {maxval}/{minval} patterns
        ftime (datetime): The forecast valid time
        plotdict (dict): The plot settings for the variable being plotted
        plotstart (float): time.time() when plotting began, for the timing log message
    """
    img.set_clim(vmin=plotdict["vmin"],vmax=plotdict["vmax"])

    # Check the valid file formats supported for this figure
    validfmts=fig.canvas.get_supported_filetypes()
    # Create a dict of substitutable patterns to make string substitutions easier, and determine output filename
    patterns,outfile,fmt = set_patterns_and_outfile(validfmts,var,lev,filepath,field,ftime,plotdict)

    logger.debug("Configuring plot title")
    if plottitle:=plotdict["title"].get("text"):
        ax.set_title(plottitle.format_map(patterns), wrap=True, fontsize=plotdict["title"]["fontsize"])
    else:
        logger.warning("No 'text' field for title specified, creating plot with no title")

    logger.debug("Configuring plot colorbar")
    if plotdict.get("colorbar"):
        if plotdict.get("colorbar").get("enable"):
            cb = plotdict["colorbar"]
            cbar = fig.colorbar(img,ax=ax,orientation=cb["orientation"])
            if cb.get("label"):
                cbar.set_label(cb["label"].format_map(patterns), fontsize=cb["fontsize"])
                cbar.ax.tick_params(labelsize=cb["fontsize"])
            if dl:=cb.get("datalabels"):
                bounds = sorted(dl.keys())
                labels = list(dl.values()) # text labels
                cbar.set_ticks(bounds)
                cbar.set_ticklabels(labels)

    # Make sure any subdirectories exist before we try to write the file
    if os.path.dirname(outfile):
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
    logger.debug(f"Saving plot {outfile}")
    fig.savefig(outfile,format=fmt)
    plt.close(fig)
    logger.info(f"Done saving plot {outfile}. Plot generation {time.time()-plotstart} seconds")


def check_plot_settings(plotdict, where):
    """
    Check the plot settings that would otherwise fail much later and much less clearly,
    deep inside Matplotlib or Cartopy. Called once for the top-level plot settings and
    once for each variable's plot-type block, since a block can override any of these.

    Args:
        plotdict (dict): The plot settings to check
        where     (str): The config path these settings came from, for error messages
    """
    if isinstance(plotdict["title"],str):
        raise TypeError(f"{where}title should be a dictionary, not a string\n"\
                        "Adjust your config.yaml accordingly. See default_options.yaml for details.")

    # Each range is checked on its own: a reversed range is a mistake whether or not the
    # other one happens to be set, and leaving it to be caught later means it is not caught
    # at all, since a range with a None in it is simply ignored when the extent is chosen.
    proj=plotdict["projection"]
    for rng in ("lonrange", "latrange"):
        if None in proj[rng]:
            continue
        first,second=proj[rng]
        if first >= second:
            raise ValueError(f"{where}projection:{rng} first value {first} >= second value {second}")


def set_patterns_and_outfile(valid, var, lev, filepath, field, ftime, plotdict):
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

    # max and min values for plotted field
    maxval=float(field.max().compute())
    minval=float(field.min().compute())

    # The plot type is recorded in the plot settings by normalize_plot_blocks(). One
    # variable can now produce more than one kind of figure, so {ptype} is what keeps
    # their filenames apart, and {levlabel} is what lets a single title string read
    # correctly for both a model level and a transect.
    ptype=plotdict.get("type","model_level")
    levlabel=f"{lev} cross-section" if ptype=="cross_section" else f"level {lev}"

    pattern_dict = {
        "var": var,
        "lev": lev,
        "ptype": ptype,
        "levlabel": levlabel,
        "units": "no_Units",
        "varln": "no_long_name",
        "filename": filename,
        "fnme": fnme,
        "proj": plotdict["projection"]["projection"],
        "date": "no_Time_dimension",
        "time": "no_Time_dimension",
        "maxval": f"{maxval:.2f}",
        "minval": f"{minval:.2f}",
    }
    if field.attrs.get("units"):
        pattern_dict.update({
            "units": field.attrs["units"],
        })
    if field.attrs.get("long_name"):
        pattern_dict.update({
            "varln": field.attrs["long_name"]
        })
    if ftime:
        pattern_dict.update({
            "date": ftime.strftime('%Y-%m-%d'),
            "time": ftime.strftime('%H:%M:%S')
        })


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
            raise ValueError(f"Invalid option: plotdict['exists']={plotdict['exists']}")


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


def get_data_extent_raster(raster, lon_bounds=(-180, 180), lat_bounds=(-90, 90)):
    """
    Computes data extent from image raster for automatic zooming to data domain

    Parameters
    ----------
    raster : np.ndarray
        2D raster array with NaNs outside valid region
    lon_bounds : tuple(float, float)
        Longitude range corresponding to full raster width
    lat_bounds : tuple(float, float)
        Latitude range corresponding to full raster height

    Returns
    -------
    extent : list [lon_min, lon_max, lat_min, lat_max]
    """
    valid = ~np.isnan(raster)
    if not np.any(valid):
        # no data at all
        return lon_bounds + lat_bounds

    # pixel indices of valid data
    ys, xs = np.where(valid)

    # convert indices to lon/lat using proportional scaling
    nrows, ncols = raster.shape
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    x_min = lon_min + (xs.min() / ncols) * (lon_max - lon_min)
    x_max = lon_min + (xs.max() / ncols) * (lon_max - lon_min)
    y_min = lat_max - (ys.max() / nrows) * (lat_max - lat_min)
    y_max = lat_max - (ys.min() / nrows) * (lat_max - lat_min)

    pad_fraction=0.05
    dx = (x_max - x_min) * pad_fraction
    dy = (y_max - y_min) * pad_fraction
    # y dimension is flipped for some reason
    return [x_min - dx, x_max + dx, -y_max - dy, -y_min + dy]


def get_data_extent(uxda, pad_fraction=0.05):
    """Return (lon_min, lon_max, lat_min, lat_max) in degrees, with buffer."""
    try:
        if "n_face" in uxda.dims:
            lons = getattr(uxda.uxgrid, "node_lon", None)
            lats = getattr(uxda.uxgrid, "node_lat", None)
        else:
            lons = uxda.lon
            lats = uxda.lat

        lon_min = np.nanmin(lons)
        lon_max = np.nanmax(lons)
        lat_min = np.nanmin(lats)
        lat_max = np.nanmax(lats)

        dx = (lon_max - lon_min) * pad_fraction
        dy = (lat_max - lat_min) * pad_fraction

        return [lon_min - dx, lon_max + dx, lat_min - dy, lat_max + dy]
    except Exception as e:
        raise RuntimeError(f"Could not determine lat/lon bounds: {e}")
