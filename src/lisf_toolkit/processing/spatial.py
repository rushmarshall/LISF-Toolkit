"""
Spatial processing utilities for geospatial raster datasets.

Provides functions for regridding, resampling, spatial clipping, land/water
masking, and zonal statistics.  All operations accept and return
:class:`xarray.Dataset` or :class:`xarray.DataArray` objects with
``rioxarray`` CRS metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def regrid(
    dataset: xr.Dataset,
    target_resolution: float,
    method: str = "bilinear",
    target_crs: Optional[str] = None,
) -> xr.Dataset:
    """Regrid a dataset to a new spatial resolution.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset with spatial coordinates (``latitude`` / ``longitude``
        or ``y`` / ``x``).
    target_resolution : float
        Desired grid spacing in the units of the dataset CRS (typically
        decimal degrees for EPSG:4326).
    method : str
        Interpolation method.  One of ``"bilinear"``, ``"nearest"``, or
        ``"cubic"``.
    target_crs : str, optional
        Target coordinate reference system (e.g. ``"EPSG:4326"``).  When
        *None*, the source CRS is preserved.

    Returns
    -------
    xarray.Dataset
        Regridded dataset.

    Raises
    ------
    ValueError
        If *method* is not supported or coordinates are missing.
    """
    import rioxarray  # noqa: F401 -- activates .rio accessor

    valid_methods = ("bilinear", "nearest", "cubic")
    if method not in valid_methods:
        raise ValueError(f"Unsupported method '{method}'. Choose from {valid_methods}.")

    if target_crs is not None:
        dataset = dataset.rio.reproject(target_crs)

    lat_name = _find_coord(dataset, ("latitude", "lat", "y"))
    lon_name = _find_coord(dataset, ("longitude", "lon", "x"))

    lat_min = float(dataset[lat_name].min())
    lat_max = float(dataset[lat_name].max())
    lon_min = float(dataset[lon_name].min())
    lon_max = float(dataset[lon_name].max())

    new_lat = np.arange(lat_min, lat_max + target_resolution, target_resolution)
    new_lon = np.arange(lon_min, lon_max + target_resolution, target_resolution)

    resampling_map = {"bilinear": "linear", "nearest": "nearest", "cubic": "cubic"}
    interp_kwargs = {
        lat_name: new_lat,
        lon_name: new_lon,
        "method": resampling_map[method],
    }

    result = dataset.interp(**interp_kwargs)
    logger.info(
        "Regridded to %.4f deg resolution (%d x %d).",
        target_resolution,
        len(new_lat),
        len(new_lon),
    )
    return result


# ---------------------------------------------------------------------------
# Clipping / masking
# ---------------------------------------------------------------------------

def clip_to_shapefile(
    dataset: xr.Dataset,
    shapefile: Union[str, Path],
    all_touched: bool = True,
    crs: str = "EPSG:4326",
) -> xr.Dataset:
    """Clip a dataset to the boundary of a vector shapefile.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input raster dataset with CRS metadata (via ``rioxarray``).
    shapefile : str or Path
        Path to a shapefile (``.shp``) or GeoJSON file.
    all_touched : bool
        If *True*, include all pixels touched by the geometry.
    crs : str
        CRS of the shapefile if it differs from the dataset.

    Returns
    -------
    xarray.Dataset
        Clipped dataset with pixels outside the boundary set to NaN.
    """
    import geopandas as gpd
    import rioxarray  # noqa: F401

    gdf = gpd.read_file(shapefile)
    if gdf.crs is not None and gdf.crs.to_epsg() != int(crs.split(":")[1]):
        gdf = gdf.to_crs(crs)

    geometry = gdf.geometry.values

    result = dataset.rio.clip(geometry, all_touched=all_touched)
    logger.info("Clipped dataset to shapefile: %s", shapefile)
    return result


def apply_land_mask(
    dataset: xr.Dataset,
    mask: xr.DataArray,
    land_value: int = 1,
) -> xr.Dataset:
    """Apply a land/water mask to a dataset.

    Pixels where *mask != land_value* are set to NaN.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input raster dataset.
    mask : xarray.DataArray
        Land/water mask array (same spatial extent as *dataset*).
    land_value : int
        Value in *mask* that represents land.

    Returns
    -------
    xarray.Dataset
        Masked dataset with ocean/water pixels set to NaN.
    """
    land = mask == land_value
    result = dataset.where(land)
    n_masked = int((~land).sum())
    logger.info("Applied land mask: %d pixel(s) masked out.", n_masked)
    return result


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_target(
    source: xr.DataArray,
    target: xr.DataArray,
    method: str = "nearest",
) -> xr.DataArray:
    """Resample *source* to match the grid of *target*.

    Parameters
    ----------
    source : xarray.DataArray
        Array to resample.
    target : xarray.DataArray
        Array whose spatial grid defines the target.
    method : str
        Interpolation method (``"nearest"`` or ``"linear"``).

    Returns
    -------
    xarray.DataArray
        Resampled array aligned to *target*.
    """
    lat_name = _find_coord(target, ("latitude", "lat", "y"))
    lon_name = _find_coord(target, ("longitude", "lon", "x"))
    result = source.interp(
        {lat_name: target[lat_name], lon_name: target[lon_name]},
        method=method,
    )
    logger.info("Resampled source to target grid (%s).", method)
    return result


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def zonal_stats(
    dataset: xr.DataArray,
    zones: xr.DataArray,
    stats: Sequence[str] = ("mean", "std", "min", "max", "count"),
) -> Dict[int, Dict[str, float]]:
    """Compute zonal statistics over labelled zones.

    Parameters
    ----------
    dataset : xarray.DataArray
        Raster data values.
    zones : xarray.DataArray
        Integer zone labels (same spatial shape as *dataset*).
    stats : sequence of str
        Statistics to compute for each zone.

    Returns
    -------
    dict
        Mapping of zone label to a dictionary of statistic name/value pairs.
    """
    data_vals = dataset.values.ravel()
    zone_vals = zones.values.ravel()

    valid = ~np.isnan(data_vals)
    data_vals = data_vals[valid]
    zone_vals = zone_vals[valid]

    unique_zones = np.unique(zone_vals[~np.isnan(zone_vals)]).astype(int)
    results: Dict[int, Dict[str, float]] = {}

    stat_funcs = {
        "mean": np.nanmean,
        "std": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
        "count": lambda a: float(np.sum(~np.isnan(a))),
        "median": np.nanmedian,
        "sum": np.nansum,
    }

    for z in unique_zones:
        subset = data_vals[zone_vals == z]
        results[int(z)] = {}
        for s in stats:
            fn = stat_funcs.get(s)
            if fn is None:
                raise ValueError(f"Unknown statistic '{s}'. Choose from {sorted(stat_funcs)}.")
            results[int(z)][s] = float(fn(subset))

    logger.info("Computed zonal statistics for %d zone(s).", len(results))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_coord(ds: Union[xr.Dataset, xr.DataArray], candidates: Tuple[str, ...]) -> str:
    """Find the first matching coordinate name in *ds*."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"Could not find any of {candidates} in dataset coordinates: "
        f"{list(ds.coords)}"
    )
