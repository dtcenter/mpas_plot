import logging

import xarray as xr
import uxarray as ux

logger = logging.getLogger(__name__)

def load_dataset(fn: str, gf: str = "") -> tuple[ux.UxDataset,ux.Grid]:
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


def invalid_vars(fn: str,variables: list) -> tuple[set,list]:
    """Check if the provided file contains all the variables in the provided list. Returns a set of
    all missing variables (an empty set if all variables are present)"""

    # Use Xarray to do a "lazy load" to check valid variables in provided MPAS netCDF file; this
    # avoids huge memory hits for very large files.
    with xr.open_dataset(fn, decode_cf=False) as data:
        grid_vars = set(data.variables)

    if invalid:=set(variables) - grid_vars:
        logger.error("Invalid variable(s) requested")
        logger.error(f"Variables in {fn} are:\n{grid_vars}")
        raise ValueError(f"The following variables are not in {fn}:\n{invalid}")
