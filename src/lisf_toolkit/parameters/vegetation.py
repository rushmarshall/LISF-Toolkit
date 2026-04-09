"""
Vegetation index computation from surface reflectance data.

Supports:
    - Normalised Difference Vegetation Index (NDVI)
    - Enhanced Vegetation Index (EVI)
    - Leaf Area Index (LAI) estimation from NDVI

All functions operate on :class:`numpy.ndarray` or
:class:`xarray.DataArray` inputs and return arrays of the same type.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, xr.DataArray]

# Physical limits used for output clamping
NDVI_RANGE = (-1.0, 1.0)
EVI_RANGE = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------

def calculate_ndvi(
    nir: ArrayLike,
    red: ArrayLike,
    *,
    nodata: Optional[float] = None,
    clamp: bool = True,
) -> ArrayLike:
    """Compute the Normalised Difference Vegetation Index.

    .. math::
        NDVI = \\frac{NIR - Red}{NIR + Red}

    Parameters
    ----------
    nir : array-like
        Near-infrared reflectance band.
    red : array-like
        Red reflectance band.
    nodata : float, optional
        Value representing missing data.  Pixels matching *nodata* in either
        input are set to NaN in the output.
    clamp : bool
        Clamp output to the physical range [-1, 1].

    Returns
    -------
    array-like
        NDVI values.  Returns the same type as the input arrays.
    """
    nir_f = _to_float(nir)
    red_f = _to_float(red)

    if nodata is not None:
        mask = (nir_f == nodata) | (red_f == nodata)
        nir_f = np.where(mask, np.nan, nir_f)
        red_f = np.where(mask, np.nan, red_f)

    denom = nir_f + red_f
    ndvi = np.where(denom == 0, np.nan, (nir_f - red_f) / denom)

    if clamp:
        ndvi = np.clip(ndvi, *NDVI_RANGE)

    result = _restore_type(ndvi, nir, name="ndvi")
    logger.info("Computed NDVI (shape=%s).", _shape_str(result))
    return result


# ---------------------------------------------------------------------------
# EVI
# ---------------------------------------------------------------------------

def calculate_evi(
    nir: ArrayLike,
    red: ArrayLike,
    blue: ArrayLike,
    *,
    gain: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
    nodata: Optional[float] = None,
    clamp: bool = True,
) -> ArrayLike:
    """Compute the Enhanced Vegetation Index.

    .. math::
        EVI = G \\times \\frac{NIR - Red}{NIR + C_1 \\times Red - C_2 \\times Blue + L}

    Parameters
    ----------
    nir : array-like
        Near-infrared reflectance band.
    red : array-like
        Red reflectance band.
    blue : array-like
        Blue reflectance band.
    gain : float
        Gain factor *G* (default 2.5).
    c1, c2 : float
        Aerosol resistance coefficients (defaults 6.0 and 7.5).
    l : float
        Canopy background adjustment (default 1.0).
    nodata : float, optional
        Missing-data sentinel value.
    clamp : bool
        Clamp output to [-1, 1].

    Returns
    -------
    array-like
        EVI values.
    """
    nir_f = _to_float(nir)
    red_f = _to_float(red)
    blue_f = _to_float(blue)

    if nodata is not None:
        mask = (nir_f == nodata) | (red_f == nodata) | (blue_f == nodata)
        nir_f = np.where(mask, np.nan, nir_f)
        red_f = np.where(mask, np.nan, red_f)
        blue_f = np.where(mask, np.nan, blue_f)

    denom = nir_f + c1 * red_f - c2 * blue_f + l
    evi = np.where(denom == 0, np.nan, gain * (nir_f - red_f) / denom)

    if clamp:
        evi = np.clip(evi, *EVI_RANGE)

    result = _restore_type(evi, nir, name="evi")
    logger.info("Computed EVI (shape=%s).", _shape_str(result))
    return result


# ---------------------------------------------------------------------------
# LAI estimation
# ---------------------------------------------------------------------------

def estimate_lai(
    ndvi: ArrayLike,
    *,
    k_ext: float = 0.5,
    ndvi_soil: float = 0.05,
    ndvi_veg: float = 0.95,
) -> ArrayLike:
    """Estimate Leaf Area Index from NDVI using Beer's law inversion.

    .. math::
        f_{veg} = \\frac{NDVI - NDVI_{soil}}{NDVI_{veg} - NDVI_{soil}}

        LAI = -\\frac{\\ln(1 - f_{veg})}{k_{ext}}

    Parameters
    ----------
    ndvi : array-like
        Normalised Difference Vegetation Index values.
    k_ext : float
        Light extinction coefficient (default 0.5 for broadleaf).
    ndvi_soil : float
        NDVI value representing bare soil.
    ndvi_veg : float
        NDVI value representing full vegetation cover.

    Returns
    -------
    array-like
        Estimated LAI values.  Non-physical inputs yield NaN.
    """
    ndvi_f = _to_float(ndvi)

    fveg = (ndvi_f - ndvi_soil) / (ndvi_veg - ndvi_soil)
    fveg = np.clip(fveg, 0.001, 0.999)  # avoid log(0) and log(negative)

    lai = -np.log(1.0 - fveg) / k_ext
    lai = np.where(np.isfinite(lai), lai, np.nan)

    result = _restore_type(lai, ndvi, name="lai")
    logger.info("Estimated LAI from NDVI (shape=%s).", _shape_str(result))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_float(arr: ArrayLike) -> np.ndarray:
    """Convert an array-like to a float64 numpy array."""
    if isinstance(arr, xr.DataArray):
        return arr.values.astype(np.float64)
    return np.asarray(arr, dtype=np.float64)


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
    """Return a human-readable shape string."""
    if isinstance(arr, xr.DataArray):
        return str(arr.shape)
    return str(np.asarray(arr).shape)
