#!/usr/bin/env python3
"""
Module for plotting vertical cross-sections of MPAS data

A cross-section is the orthogonal slice to the horizontal maps produced by plotit(): rather
than taking one vertical level over the whole domain, it takes a path across the domain and
plots every vertical level along it.
"""
import logging
import time

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

from plot_functions import get_colormap, finish_plot

logger = logging.getLogger(__name__)

# Mean Earth radius in km, for reporting distance along a transect
EARTH_RADIUS_KM = 6371.229

# Points used to draw a transect on a map. This only has to be enough for the line to look
# like the great-circle arc it is; it has nothing to do with how densely the data is sampled.
LINE_STEPS = 100

XAXIS_LABELS = {
    "distance": "Distance along transect (km)",
    "lat": "Latitude",
    "lon": "Longitude",
}


def cross_section_vars(config_d: dict) -> set:
    """
    Returns the set of extra variables needed by any requested cross-section plots; this is
    normally just the height variable (zgrid). Returns an empty set if no cross-sections were
    requested, so that configs that do not use them read exactly the data they did before.

    Args:
        config_d (dict): The full experiment configuration, after normalize_plot_blocks()
    Returns:
        A set of variable names
    """
    needed = set()
    for var, vardict in config_d["dataset"]["vars"].items():
        if "_blocks" not in vardict:
            raise RuntimeError(f"The settings for {var} have not been resolved into plot-type "
                               "blocks; normalize_plot_blocks() has to run before this")
        for ptype, settings in vardict["_blocks"].items():
            if ptype != "cross_section":
                continue
            xsdict = settings["cross_section"]
            if xsdict["vertical"] in ("auto", "height"):
                needed.add(xsdict["zvar"])
    return needed


def transect_points(waypoints, steps):
    """
    Converts a list of [latitude, longitude] waypoints into evenly spaced sample points along
    the great-circle arcs joining them.

    Args:
        waypoints (list): A list of at least two [lat, lon] pairs, in degrees
        steps      (int): Approximate total number of sample points along the whole path
    Returns:
        A tuple (lon, lat, dist_km) of 1-D arrays; dist_km is the cumulative distance along
        the transect, for use as the horizontal plot axis
    """
    if waypoints is None or len(waypoints) < 2:
        raise ValueError(f"A transect needs at least two [lat, lon] waypoints, got {waypoints}")

    pts = np.asarray(waypoints, dtype=float)
    rlat, rlon = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    verts = np.column_stack([np.cos(rlat) * np.cos(rlon),
                             np.cos(rlat) * np.sin(rlon),
                             np.sin(rlat)])

    # Central angle of each segment, which is also its length in radians of arc
    angles = 2.0 * np.arcsin(np.linalg.norm(np.diff(verts, axis=0), axis=1) / 2.0)
    total = angles.sum()
    if total <= 0.0:
        raise ValueError(f"Transect waypoints are coincident: {waypoints}")

    xyz, dist, travelled = [], [], 0.0
    for i, omega in enumerate(angles):
        # Points are shared out between segments in proportion to their length, so that the
        # sample spacing is uniform across the whole path rather than within each segment
        n = max(2, int(round(steps * omega / total)))
        frac = np.linspace(0.0, 1.0, n)
        if omega > 0.0:
            # Spherical linear interpolation along the great circle
            seg = ((np.sin((1.0 - frac)[:, None] * omega) * verts[i]
                    + np.sin(frac[:, None] * omega) * verts[i + 1]) / np.sin(omega))
        else:
            seg = np.repeat(verts[i][None, :], n, axis=0)
        segdist = travelled + frac * omega * EARTH_RADIUS_KM
        # Drop the first point of every segment after the first, which duplicates the
        # last point of the previous one
        first = 0 if not xyz else 1
        xyz.append(seg[first:])
        dist.append(segdist[first:])
        travelled = segdist[-1]

    xyz = np.vstack(xyz)
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    return lon, lat, np.concatenate(dist)


def locate_cells(uxgrid, lon, lat):
    """
    Finds the grid cell containing each sample point.

    NOTE: this must be called from within the plotting worker, never when setting up the plot
    tasks. Locating points builds a search tree that is cached on the grid object, and since
    the grid is pickled into every task for parallel plotting, building it early would copy
    that tree into every task.

    Args:
        uxgrid (ux.Grid): The grid to search
        lon, lat (array): Sample point coordinates, in degrees
    Returns:
        An array of cell indices, with -1 for points that fall outside the mesh
    """
    faces, counts = uxgrid.get_faces_containing_point(np.column_stack([lon, lat]))
    cells = np.where(counts > 0, np.asarray(faces)[:, 0], -1)

    outside = int(np.count_nonzero(cells < 0))
    if outside == cells.size:
        raise ValueError("No points along this transect fall inside the model domain; check "
                         "the transect waypoints against the extent of your mesh")
    if outside:
        logger.warning(f"{outside} of {cells.size} transect points fall outside the model "
                       "domain and will be left blank")
    return cells


def sample_along_transect(uxda, cells, mask=True):
    """
    Selects the data column at each sample point along a transect.

    Args:
        uxda (ux.UxDataArray): A face-centered field
        cells        (array): Cell index of each sample point, as returned by locate_cells()
        mask          (bool): Whether to blank out points that fall outside the mesh. This
                              should be True for the field being plotted, but False when
                              sampling the heights that position it: Matplotlib will not
                              accept a non-finite plot coordinate, and since the data at
                              those points is blank anyway, nothing is drawn there.
    Returns:
        An xr.DataArray whose first dimension, "steps", is the sample point
    """
    inside = cells >= 0
    # Index with a clamped copy so the selection is valid everywhere, then mask afterwards
    picker = xr.DataArray(np.where(inside, cells, 0), dims="steps")
    sampled = uxda.isel(n_face=picker)

    # uxarray keeps the grid's own dimension name on the result rather than taking the name
    # of the indexer, so the sampled axis usually comes back as n_face with one entry per
    # sample point. Move it to the front and rename it, so that what comes out of here is a
    # plain transect array whose first dimension is always the sample point.
    stepdim = "n_face" if "n_face" in sampled.dims else "steps"
    sampled = sampled.transpose(stepdim, ...)
    dims = ("steps",) + tuple(d for d in sampled.dims if d != stepdim)
    out = xr.DataArray(np.asarray(sampled.values, dtype=float), dims=dims)
    if mask:
        out = out.where(xr.DataArray(inside, dims="steps"))
    # Rebuilding the array from its values leaves the metadata behind, and the title and
    # colorbar are written from it -- without this the plot has no units and no long name
    out.attrs = dict(uxda.attrs)
    return out


def level_heights(zprofile, nlev):
    """
    Matches sampled interface heights to the vertical levels of the field being plotted.

    MPAS stores zgrid on layer interfaces (nVertLevelsP1) while most fields are on layer
    mid-points (nVertLevels); w is one of the exceptions and lives on the interfaces. Which
    case applies is decided by size rather than by dimension name, so that fields with
    unusual vertical coordinates still work.

    Args:
        zprofile (array): Interface heights along the transect, shape (steps, nVertLevelsP1)
        nlev       (int): Number of vertical levels in the field being plotted
    Returns:
        A tuple (heights, surface); heights is None if the field's vertical coordinate does
        not correspond to these interfaces
    """
    surface = zprofile[:, 0]
    if nlev == zprofile.shape[1] - 1:
        return 0.5 * (zprofile[:, :-1] + zprofile[:, 1:]), surface
    if nlev == zprofile.shape[1]:
        return zprofile, surface
    return None, surface


def draw_transects(ax, settings):
    """
    Draw the paths of cross-section transects onto a map, so that a horizontal plot can show
    where a vertical section through it was taken.

    Args:
        ax     (GeoAxes): The map axes to draw on
        settings  (dict): The plot:transects settings. "paths" holds the transect geometry,
                          which normalize_plot_blocks() resolves from the cross_section
                          settings and puts here
    """
    paths = settings.get("paths") or {}
    wanted = settings.get("names", "all")
    if wanted != "all":
        if isinstance(wanted, str):
            wanted = [wanted]
        if missing := [n for n in wanted if n not in paths]:
            raise ValueError(f"plot:transects:names asks for {missing}, which is not defined "
                             "in cross_section:transects")
        paths = {n: paths[n] for n in wanted}

    if not paths:
        logger.warning("plot:transects:enable is set, but no transects are defined; put them "
                       "in the top-level cross_section:transects")
        return

    for name, waypoints in paths.items():
        lon, lat, _ = transect_points(waypoints, LINE_STEPS)
        # Drawn in the geodetic frame and already densified, so the line follows the same
        # great-circle arcs the section is sampled along, in any projection
        ax.plot(lon, lat, transform=ccrs.Geodetic(), color=settings["color"],
                linewidth=settings["linewidth"], linestyle=settings["linestyle"], zorder=10)
        if settings.get("label"):
            ax.text(lon[0], lat[0], f" {name}", transform=ccrs.PlateCarree(),
                    color=settings["color"], fontsize=settings["fontsize"],
                    ha="left", va="center", zorder=10)
        logger.debug(f"Drew transect {name} from ({lat[0]:.2f}, {lon[0]:.2f}) to "
                     f"({lat[-1]:.2f}, {lon[-1]:.2f})")


def plot_cross_section(vardict: dict, uxda, var: str, lev, filepath: str, ftime,
                       uxds=None) -> None:
    """
    Plots a vertical cross-section of a variable along a named transect.

    This has the same signature as plotit(), so that the two can be dispatched identically;
    see PLOT_FUNCTIONS in plot_mpas_netcdf.py.

    Args:
        vardict      (dict): Experiment settings specific to the variable being plotted
        uxda (ux.UxDataArray): The data to be plotted, with grid information
        var           (str): Name of the variable being plotted
        lev           (str): Name of the transect to plot along
        filepath      (str): The filename of the input data that was read into the ux objects
        ftime    (datetime): The forecast valid time as a datetime object
        uxds (ux.UxDataset): The full dataset, needed to read the height variable
    """
    plotstart = time.time()
    plotdict = vardict["plot"]
    xsdict = vardict["cross_section"]

    transect = xsdict["transects"].get(lev)
    if transect is None:
        raise ValueError(f"Transect {lev} is not defined in cross_section:transects")

    if "n_face" not in uxda.dims:
        logger.warning(f"Variable {var} not face-centered, will interpolate to faces")
        uxda = uxda.remap.inverse_distance_weighted(uxda.uxgrid, remap_to='face centers', k=3)

    lon, lat, dist = transect_points(transect, xsdict["steps"])
    cells = locate_cells(uxda.uxgrid, lon, lat)
    data = sample_along_transect(uxda, cells)

    vertcoord = vardict["vertcoord"]
    if vertcoord not in data.dims:
        raise ValueError(f"Variable {var} has no vertical dimension {vertcoord}; a "
                         "cross-section can only be plotted for a three-dimensional field")
    nlev = data.sizes[vertcoord]

    # Vertical axis: geometric height where we can get it, model level otherwise
    heights, surface = None, None
    if xsdict["vertical"] in ("auto", "height"):
        zvar = xsdict["zvar"]
        if uxds is not None and zvar in uxds.variables:
            zprofile = sample_along_transect(uxds[zvar], cells, mask=False).values
            heights, surface = level_heights(zprofile, nlev)
            if heights is None:
                logger.debug(f"{zvar} has {zprofile.shape[1]} interfaces, which does not match "
                             f"{nlev} levels of {var}")
        if heights is None:
            if xsdict["vertical"] == "height":
                raise ValueError(f"cross_section:vertical is 'height' but {zvar} is not "
                                 f"usable for {var}; it must be present in the data files or "
                                 "in dataset:gridfile, and match the variable's levels")
            logger.warning(f"Could not use {zvar} for the vertical axis of {var}; plotting "
                           "against model level instead. Set cross_section:vertical to "
                           "'height' to make this an error rather than a fallback.")

    if heights is None:
        yvals = np.repeat(np.arange(nlev)[None, :], dist.size, axis=0)
        ylabel = "Model level"
    else:
        yvals = heights
        ylabel = "Height (m)"

    xcoord = {"distance": dist, "lat": lat, "lon": lon}[xsdict["xaxis"]]
    xvals = np.repeat(np.asarray(xcoord)[:, None], nlev, axis=1)

    cmap = get_colormap(plotdict)
    fig, ax = plt.subplots(1, 1, figsize=(plotdict["figwidth"], plotdict["figheight"]),
                           dpi=plotdict["dpi"], constrained_layout=True)
    # A 2-D vertical coordinate lets the terrain-following levels be drawn as they actually
    # are, with no interpolation onto a regular height grid
    img = ax.pcolormesh(xvals, yvals, data.values, cmap=cmap,
                        shading=xsdict["shading"])

    ax.set_xlabel(XAXIS_LABELS[xsdict["xaxis"]], fontsize=plotdict["title"]["fontsize"])
    ax.set_ylabel(ylabel, fontsize=plotdict["title"]["fontsize"])
    ax.tick_params(labelsize=plotdict["title"]["fontsize"])
    if xsdict["ymin"] is not None or xsdict["ymax"] is not None:
        ax.set_ylim(xsdict["ymin"], xsdict["ymax"])

    if surface is not None and xsdict["terrain"]["enable"]:
        # Heights are left unmasked so that Matplotlib has a finite coordinate everywhere,
        # but the terrain itself is only real where the transect is inside the mesh
        ax.fill_between(xcoord, ax.get_ylim()[0], np.where(cells >= 0, surface, np.nan),
                        color=xsdict["terrain"]["color"], zorder=5)

    finish_plot(fig, ax, img, var, lev, filepath, data, ftime, plotdict, plotstart)
