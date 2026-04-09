"""
Terrain parameter derivation from Digital Elevation Models (DEMs).

Computes slope, aspect, profile curvature, plan curvature, and the
Topographic Wetness Index (TWI) from gridded elevation data.

All gradient computations use second-order finite differences and account
for cell size in both the *x* and *y* directions.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, xr.DataArray]


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------

def calculate_slope(
    dem: ArrayLike,
    cell_size: float = 1.0,
    units: str = "degrees",
) -> ArrayLike:
    """Compute terrain slope from a DEM.

    Uses second-order centred finite differences (the Horn method).

    Parameters
    ----------
    dem : array-like
        2-D elevation array (rows = latitude, columns = longitude).
    cell_size : float
        Grid cell size in the same linear units as the elevation
        (e.g. metres).  For geographic coordinates, pass approximate
        metres per degree.
    units : str
        Output units: ``"degrees"`` or ``"radians"``.

    Returns
    -------
    array-like
        Slope values in the requested units.  Same shape as *dem*.
    """
    elev = _to_float2d(dem)

    dy, dx = np.gradient(elev, cell_size)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))

    if units == "degrees":
        result = np.degrees(slope_rad)
    elif units == "radians":
        result = slope_rad
    else:
        raise ValueError(f"Unsupported units '{units}'. Use 'degrees' or 'radians'.")

    out = _restore_type(result, dem, name="slope")
    logger.info("Computed slope (units=%s, shape=%s).", units, _shape_str(out))
    return out


# ---------------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------------

def calculate_aspect(
    dem: ArrayLike,
    cell_size: float = 1.0,
    units: str = "degrees",
) -> ArrayLike:
    """Compute terrain aspect (downslope direction) from a DEM.

    Aspect is measured clockwise from north (0 = N, 90 = E, 180 = S,
    270 = W).  Flat areas are assigned -1.

    Parameters
    ----------
    dem : array-like
        2-D elevation array.
    cell_size : float
        Grid cell size.
    units : str
        ``"degrees"`` or ``"radians"``.

    Returns
    -------
    array-like
        Aspect values.  Same shape as *dem*.
    """
    elev = _to_float2d(dem)

    dy, dx = np.gradient(elev, cell_size)

    # atan2(-dx, dy) gives angle from north, clockwise positive
    aspect_rad = np.arctan2(-dx, dy)
    # Convert from [-pi, pi] to [0, 2*pi]
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2 * np.pi, aspect_rad)

    # Mark flat areas
    flat = (dx == 0) & (dy == 0)
    aspect_rad = np.where(flat, np.nan, aspect_rad)

    if units == "degrees":
        result = np.degrees(aspect_rad)
        result = np.where(flat, -1.0, result)
    elif units == "radians":
        result = np.where(flat, -1.0, aspect_rad)
    else:
        raise ValueError(f"Unsupported units '{units}'. Use 'degrees' or 'radians'.")

    out = _restore_type(result, dem, name="aspect")
    logger.info("Computed aspect (units=%s, shape=%s).", units, _shape_str(out))
    return out


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------

def calculate_curvature(
    dem: ArrayLike,
    cell_size: float = 1.0,
) -> ArrayLike:
    """Compute profile curvature (combined second derivatives).

    Positive values indicate concave-up (converging) terrain; negative
    values indicate convex-up (diverging) terrain.

    Parameters
    ----------
    dem : array-like
        2-D elevation array.
    cell_size : float
        Grid cell size.

    Returns
    -------
    array-like
        Curvature values.  Same shape as *dem*.
    """
    elev = _to_float2d(dem)

    # Second derivatives
    dyy = np.gradient(np.gradient(elev, cell_size, axis=0), cell_size, axis=0)
    dxx = np.gradient(np.gradient(elev, cell_size, axis=1), cell_size, axis=1)

    curvature = -(dxx + dyy)

    out = _restore_type(curvature, dem, name="curvature")
    logger.info("Computed curvature (shape=%s).", _shape_str(out))
    return out


# ---------------------------------------------------------------------------
# Topographic Wetness Index
# ---------------------------------------------------------------------------

def topographic_wetness_index(
    dem: ArrayLike,
    cell_size: float = 1.0,
    min_slope: float = 0.001,
) -> ArrayLike:
    """Compute the Topographic Wetness Index (TWI).

    .. math::
        TWI = \\ln\\left(\\frac{a}{\\tan(\\beta)}\\right)

    where *a* is the specific contributing area (approximated here as the
    cell area divided by the contour length) and *beta* is the local
    slope angle.

    Parameters
    ----------
    dem : array-like
        2-D elevation array.
    cell_size : float
        Grid cell size in metres.
    min_slope : float
        Minimum slope (radians) to avoid division by zero in flat areas.

    Returns
    -------
    array-like
        TWI values.  Same shape as *dem*.

    Notes
    -----
    This implementation uses a simple gradient-based slope and a uniform
    contributing area per cell.  For research-grade TWI, consider using a
    D-infinity flow-direction algorithm for *a*.
    """
    elev = _to_float2d(dem)

    dy, dx = np.gradient(elev, cell_size)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    slope_rad = np.maximum(slope_rad, min_slope)

    # Approximate specific contributing area as cell area / contour length.
    # With a uniform-flow assumption, a = cell_size (area / contour length).
    specific_area = np.full_like(elev, cell_size)

    twi = np.log(specific_area / np.tan(slope_rad))

    out = _restore_type(twi, dem, name="twi")
    logger.info("Computed TWI (shape=%s).", _shape_str(out))
    return out


# ---------------------------------------------------------------------------
# Flow accumulation (simplified D8)
# ---------------------------------------------------------------------------

def flow_accumulation_d8(
    dem: ArrayLike,
) -> np.ndarray:
    """Compute D8 flow accumulation from a DEM.

    Each cell flows to its steepest downslope neighbour.  The result is
    the count of upstream cells draining through each pixel.

    Parameters
    ----------
    dem : array-like
        2-D elevation array.

    Returns
    -------
    numpy.ndarray
        Integer flow accumulation grid.
    """
    elev = _to_float2d(dem)
    nrows, ncols = elev.shape
    acc = np.ones((nrows, ncols), dtype=np.int64)

    # Neighbour offsets (row, col) and distance weights
    offsets = [
        (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        (0, -1, 1.0),                    (0, 1, 1.0),
        (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414),
    ]

    # Flatten and sort by elevation (highest first)
    flat_elev = elev.ravel()
    order = np.argsort(-flat_elev)

    for idx in order:
        r, c = divmod(int(idx), ncols)
        max_drop = 0.0
        target = None
        for dr, dc, dist in offsets:
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc_ < ncols:
                drop = (elev[r, c] - elev[nr, nc_]) / dist
                if drop > max_drop:
                    max_drop = drop
                    target = (nr, nc_)
        if target is not None:
            acc[target[0], target[1]] += acc[r, c]

    logger.info("Computed D8 flow accumulation (shape=%s).", elev.shape)
    return acc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float2d(arr: ArrayLike) -> np.ndarray:
    """Ensure *arr* is a 2-D float64 numpy array."""
    if isinstance(arr, xr.DataArray):
        vals = arr.values
    else:
        vals = np.asarray(arr)
    vals = vals.astype(np.float64)
    if vals.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {vals.shape}.")
    return vals


def _restore_type(
    values: np.ndarray,
    reference: ArrayLike,
    name: str = "result",
) -> ArrayLike:
    """Wrap *values* back into the type of *reference*."""
    if isinstance(reference, xr.DataArray):
        return xr.DataArray(
            data=values,
            dims=reference.dims,
            coords=reference.coords,
            name=name,
        )
    return values


def _shape_str(arr: ArrayLike) -> str:
    if isinstance(arr, xr.DataArray):
        return str(arr.shape)
    return str(np.asarray(arr).shape)
