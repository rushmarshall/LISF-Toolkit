"""
Temporal processing utilities for time-series geospatial data.

Provides functions for temporal aggregation (daily, monthly, seasonal),
rolling-window statistics, gap detection, and interpolation-based gap filling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def aggregate(
    dataset: xr.Dataset,
    freq: str = "ME",
    method: str = "mean",
    time_dim: str = "time",
    min_periods: int = 1,
) -> xr.Dataset:
    """Aggregate a dataset to a coarser temporal frequency.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset with a time dimension.
    freq : str
        Target frequency using pandas offset aliases.  Common choices:
        ``"D"`` (daily), ``"ME"`` (month-end), ``"QE"`` (quarter-end),
        ``"YE"`` (year-end).
    method : str
        Aggregation method: ``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
        ``"median"``.
    time_dim : str
        Name of the time dimension.
    min_periods : int
        Minimum number of valid observations required per group.

    Returns
    -------
    xarray.Dataset
        Temporally aggregated dataset.

    Raises
    ------
    ValueError
        If *method* is not supported.
    """
    valid_methods = ("mean", "sum", "min", "max", "median")
    if method not in valid_methods:
        raise ValueError(f"Unsupported method '{method}'. Choose from {valid_methods}.")

    resampler = dataset.resample({time_dim: freq})

    dispatch = {
        "mean": lambda r: r.mean(dim=time_dim, min_count=min_periods),
        "sum": lambda r: r.sum(dim=time_dim, min_count=min_periods),
        "min": lambda r: r.min(dim=time_dim),
        "max": lambda r: r.max(dim=time_dim),
        "median": lambda r: r.median(dim=time_dim),
    }

    result = dispatch[method](resampler)
    n_steps_in = dataset.sizes.get(time_dim, 0)
    n_steps_out = result.sizes.get(time_dim, 0)
    logger.info(
        "Aggregated %d -> %d time steps (freq=%s, method=%s).",
        n_steps_in,
        n_steps_out,
        freq,
        method,
    )
    return result


def monthly_climatology(
    dataset: xr.Dataset,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute the multi-year monthly climatology.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset spanning multiple years.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        Climatology with a ``month`` coordinate (1--12).
    """
    result = dataset.groupby(f"{time_dim}.month").mean(dim=time_dim)
    logger.info("Computed monthly climatology (12 months).")
    return result


def compute_anomalies(
    dataset: xr.Dataset,
    climatology: Optional[xr.Dataset] = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute anomalies relative to a climatology.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    climatology : xarray.Dataset, optional
        Pre-computed climatology.  If *None*, the climatology is derived
        from *dataset* itself.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        Anomaly dataset (values minus climatological mean).
    """
    if climatology is None:
        climatology = monthly_climatology(dataset, time_dim=time_dim)
    anomalies = dataset.groupby(f"{time_dim}.month") - climatology
    logger.info("Computed anomalies relative to climatology.")
    return anomalies


# ---------------------------------------------------------------------------
# Rolling window statistics
# ---------------------------------------------------------------------------

def rolling_stat(
    dataset: xr.Dataset,
    window: int,
    method: str = "mean",
    time_dim: str = "time",
    center: bool = True,
    min_periods: int = 1,
) -> xr.Dataset:
    """Apply a rolling-window statistic along the time dimension.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    window : int
        Number of time steps in the rolling window.
    method : str
        ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``.
    time_dim : str
        Name of the time dimension.
    center : bool
        Centre the rolling window.
    min_periods : int
        Minimum number of valid observations in a window.

    Returns
    -------
    xarray.Dataset
    """
    valid = ("mean", "sum", "min", "max", "std")
    if method not in valid:
        raise ValueError(f"Unsupported method '{method}'. Choose from {valid}.")

    roller = dataset.rolling({time_dim: window}, center=center, min_periods=min_periods)
    result = getattr(roller, method)()
    logger.info("Applied rolling %s (window=%d) along %s.", method, window, time_dim)
    return result


# ---------------------------------------------------------------------------
# Gap detection and filling
# ---------------------------------------------------------------------------

def detect_gaps(
    dataset: xr.Dataset,
    time_dim: str = "time",
) -> Dict[str, List[Tuple[str, str]]]:
    """Detect temporal gaps in a dataset.

    A *gap* is defined as a missing time step between the first and last
    valid observation for each variable.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset with a regular time dimension.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    dict
        Mapping of variable name to a list of ``(gap_start, gap_end)`` ISO
        date-string tuples.
    """
    times = pd.DatetimeIndex(dataset[time_dim].values)
    if len(times) < 2:
        return {}

    freq = pd.infer_freq(times)
    if freq is None:
        delta = times[1] - times[0]
    else:
        delta = pd.tseries.frequencies.to_offset(freq)

    gaps: Dict[str, List[Tuple[str, str]]] = {}

    for var in dataset.data_vars:
        arr = dataset[var]
        spatial_dims = [d for d in arr.dims if d != time_dim]
        if spatial_dims:
            has_data = (~arr.isnull()).any(dim=spatial_dims).values
        else:
            has_data = (~arr.isnull()).values

        var_gaps: List[Tuple[str, str]] = []
        in_gap = False
        gap_start = None

        for i in range(len(has_data)):
            if not has_data[i] and not in_gap:
                in_gap = True
                gap_start = times[i]
            elif has_data[i] and in_gap:
                var_gaps.append((str(gap_start), str(times[i - 1])))
                in_gap = False
                gap_start = None
        if in_gap and gap_start is not None:
            var_gaps.append((str(gap_start), str(times[-1])))

        if var_gaps:
            gaps[str(var)] = var_gaps

    total = sum(len(v) for v in gaps.values())
    logger.info("Detected %d gap(s) across %d variable(s).", total, len(gaps))
    return gaps


def fill_gaps(
    dataset: xr.Dataset,
    method: str = "linear",
    time_dim: str = "time",
    max_gap: Optional[int] = None,
) -> xr.Dataset:
    """Fill temporal gaps using interpolation.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset with NaN gaps along the time dimension.
    method : str
        Interpolation method passed to :meth:`xarray.Dataset.interpolate_na`.
        Common choices: ``"linear"``, ``"nearest"``, ``"cubic"``,
        ``"polynomial"``.
    time_dim : str
        Name of the time dimension.
    max_gap : int, optional
        Maximum number of consecutive NaNs to fill.  Larger gaps are left
        unchanged.

    Returns
    -------
    xarray.Dataset
        Dataset with gaps filled.
    """
    kwargs: Dict[str, object] = {"dim": time_dim, "method": method}
    if max_gap is not None:
        kwargs["max_gap"] = max_gap

    result = dataset.interpolate_na(**kwargs)  # type: ignore[arg-type]

    n_before = int(dataset.isnull().sum())
    n_after = int(result.isnull().sum())
    logger.info(
        "Gap filling (%s): NaN count %d -> %d (filled %d).",
        method,
        n_before,
        n_after,
        n_before - n_after,
    )
    return result
