import logging

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

