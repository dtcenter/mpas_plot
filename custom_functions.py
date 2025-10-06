#!/usr/bin/env python3
"""
Module for various plotting functions
"""
import logging
import uxarray as ux

logger = logging.getLogger(__name__)


# =====================
# User-defined derived functions
# =====================
def diff_prev_timestep(field: ux.UxDataArray, dim: str = "Time") -> ux.UxDataArray:
    """
    Return timestep-to-timestep differences for input field.
    First timestep is filled with zeros.
    """
    # Compute differences along Time
    result = field.diff(dim=dim, n=1)

    return result


def sum_fields(x1, x2):
    return x1 + x2


def vert_max(field: ux.UxDataArray, dim: str = "nVertLevels") -> ux.UxDataArray:
    """
    Take a 3d input field and return a 2d field representing the maximum value for each vertical column.
    The "dim" input is the name of the vertical dimension to operate over.
    """

    if dim not in field.dims:
        raise ValueError(
            f"Vertical dimension '{dim}' not found in data array. "
            f"Available dimensions: {list(field.dims)}"
        )

    # Compute the maximum using xarray’s reduction
    vertmax = field.max(dim=dim, keep_attrs=True)

    # Preserve UxArray-specific grid metadata if present
#    if hasattr(field, "uxgrid"):
#        max_da = ux.UxDataArray(max_da, uxgrid=field.uxgrid)

    return vertmax


DERIVED_FUNCTIONS = {
    "diff_prev_timestep": diff_prev_timestep,
    "sum_fields": sum_fields,
    "vert_max": vert_max,
}

