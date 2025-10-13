#!/usr/bin/env python3
"""
Module for various plotting functions
"""
import inspect
import logging
import os
import traceback

import numpy as np
import cartopy.crs as ccrs

logger = logging.getLogger(__name__)


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
