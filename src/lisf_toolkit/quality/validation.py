"""
Quality assurance and quality control (QA/QC) for land-surface datasets.

Provides configurable checks for:
    - Physical range violations
    - Spatial completeness (percentage of valid pixels per time step)
    - Temporal gap detection
    - Summary statistics

Results are collected into a :class:`QAReport` dataclass which can be
printed, serialised to JSON, or inspected programmatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QA report
# ---------------------------------------------------------------------------

@dataclass
class QACheckResult:
    """Result of a single QA check."""

    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}"


@dataclass
class QAReport:
    """Aggregated QA/QC report for a dataset.

    Attributes
    ----------
    dataset_name : str
        Human-readable identifier for the dataset.
    created_at : str
        ISO-format timestamp of report creation.
    checks : list of QACheckResult
        Individual check results.
    """

    dataset_name: str = "unnamed"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checks: List[QACheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return all(c.passed for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"QA Report: {self.dataset_name}",
            f"Created:   {self.created_at}",
            f"Result:    {'ALL PASSED' if self.passed else 'FAILURES DETECTED'}",
            f"Checks:    {self.n_passed} passed, {self.n_failed} failed",
            "-" * 50,
        ]
        for check in self.checks:
            lines.append(str(check))
            for key, val in check.details.items():
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in self.checks
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_range(
    dataset: xr.Dataset,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> QACheckResult:
    """Check that variable values fall within physical bounds.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    thresholds : dict, optional
        Mapping of variable name to ``(min, max)`` tuple.  Variables not
        listed are skipped.  If *None*, a set of common defaults is used.

    Returns
    -------
    QACheckResult
    """
    defaults: Dict[str, Tuple[float, float]] = {
        "ndvi": (-1.0, 1.0),
        "evi": (-1.0, 1.0),
        "lai": (0.0, 15.0),
        "lst": (180.0, 350.0),
        "temperature": (180.0, 350.0),
        "2m_temperature": (180.0, 350.0),
        "precipitation": (0.0, 1000.0),
        "total_precipitation": (0.0, 1.0),
        "soil_moisture": (0.0, 1.0),
        "elevation": (-500.0, 9000.0),
    }

    if thresholds is None:
        thresholds = defaults

    violations: Dict[str, Dict[str, Any]] = {}
    all_ok = True

    for var in dataset.data_vars:
        var_str = str(var)
        if var_str not in thresholds:
            continue
        lo, hi = thresholds[var_str]
        arr = dataset[var].values
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        below = int(np.sum(valid < lo))
        above = int(np.sum(valid > hi))
        if below > 0 or above > 0:
            all_ok = False
            violations[var_str] = {
                "expected_range": (lo, hi),
                "below_min": below,
                "above_max": above,
                "actual_min": float(np.nanmin(valid)),
                "actual_max": float(np.nanmax(valid)),
            }

    details: Dict[str, Any] = {
        "variables_checked": [v for v in dataset.data_vars if str(v) in thresholds],
        "violations": violations if violations else "none",
    }

    result = QACheckResult(name="range_check", passed=all_ok, details=details)
    logger.info("Range check: %s", result)
    return result


def check_spatial_completeness(
    dataset: xr.Dataset,
    threshold: float = 0.8,
    time_dim: str = "time",
) -> QACheckResult:
    """Check per-timestep spatial completeness.

    A time step *fails* when the fraction of non-NaN pixels is below
    *threshold*.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    threshold : float
        Minimum acceptable fraction of valid pixels (0--1).
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    QACheckResult
    """
    failing_steps: Dict[str, List[str]] = {}
    all_ok = True

    for var in dataset.data_vars:
        arr = dataset[var]
        if time_dim not in arr.dims:
            continue
        spatial_dims = [d for d in arr.dims if d != time_dim]
        n_spatial = int(np.prod([arr.sizes[d] for d in spatial_dims])) if spatial_dims else 1
        frac_valid = (~arr.isnull()).sum(dim=spatial_dims) / n_spatial
        bad_times = frac_valid[frac_valid < threshold]
        if bad_times.size > 0:
            all_ok = False
            failing_steps[str(var)] = [
                str(t) for t in bad_times[time_dim].values
            ]

    details: Dict[str, Any] = {
        "threshold": threshold,
        "failing_timesteps": failing_steps if failing_steps else "none",
    }

    result = QACheckResult(name="spatial_completeness", passed=all_ok, details=details)
    logger.info("Spatial completeness check: %s", result)
    return result


def check_temporal_gaps(
    dataset: xr.Dataset,
    time_dim: str = "time",
) -> QACheckResult:
    """Detect temporal gaps (entirely-NaN time steps).

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    QACheckResult
    """
    gap_info: Dict[str, int] = {}
    all_ok = True

    for var in dataset.data_vars:
        arr = dataset[var]
        if time_dim not in arr.dims:
            continue
        other_dims = [d for d in arr.dims if d != time_dim]
        if other_dims:
            all_nan = arr.isnull().all(dim=other_dims)
        else:
            all_nan = arr.isnull()
        n_gaps = int(all_nan.sum())
        if n_gaps > 0:
            all_ok = False
            gap_info[str(var)] = n_gaps

    details: Dict[str, Any] = {
        "gaps_per_variable": gap_info if gap_info else "none",
    }

    result = QACheckResult(name="temporal_gaps", passed=all_ok, details=details)
    logger.info("Temporal gap check: %s", result)
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def summary_statistics(
    dataset: xr.Dataset,
) -> Dict[str, Dict[str, float]]:
    """Compute per-variable summary statistics.

    Returns a dictionary mapping each variable name to:
    ``mean``, ``std``, ``min``, ``max``, ``median``, ``count``,
    ``nan_count``, and ``nan_fraction``.

    Parameters
    ----------
    dataset : xarray.Dataset

    Returns
    -------
    dict
    """
    stats: Dict[str, Dict[str, float]] = {}
    for var in dataset.data_vars:
        arr = dataset[var].values.ravel()
        valid = arr[~np.isnan(arr)]
        n_total = len(arr)
        n_nan = int(np.sum(np.isnan(arr)))
        stats[str(var)] = {
            "mean": float(np.nanmean(arr)) if len(valid) else float("nan"),
            "std": float(np.nanstd(arr)) if len(valid) else float("nan"),
            "min": float(np.nanmin(arr)) if len(valid) else float("nan"),
            "max": float(np.nanmax(arr)) if len(valid) else float("nan"),
            "median": float(np.nanmedian(arr)) if len(valid) else float("nan"),
            "count": float(len(valid)),
            "nan_count": float(n_nan),
            "nan_fraction": n_nan / n_total if n_total > 0 else 0.0,
        }
    return stats


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

_CHECK_REGISTRY: Dict[str, Any] = {
    "range": check_range,
    "spatial_completeness": check_spatial_completeness,
    "temporal_gaps": check_temporal_gaps,
}


def run_qa(
    dataset: xr.Dataset,
    checks: Optional[Sequence[str]] = None,
    *,
    dataset_name: str = "unnamed",
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    completeness_threshold: float = 0.8,
    time_dim: str = "time",
) -> QAReport:
    """Run a suite of QA checks and return a report.

    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset.
    checks : sequence of str, optional
        Names of checks to run.  If *None*, all registered checks are run.
        Valid names: ``"range"``, ``"spatial_completeness"``,
        ``"temporal_gaps"``.
    dataset_name : str
        Label for the report.
    thresholds : dict, optional
        Passed to :func:`check_range`.
    completeness_threshold : float
        Passed to :func:`check_spatial_completeness`.
    time_dim : str
        Time dimension name.

    Returns
    -------
    QAReport
    """
    if checks is None:
        checks = list(_CHECK_REGISTRY.keys())

    report = QAReport(dataset_name=dataset_name)

    for name in checks:
        if name not in _CHECK_REGISTRY:
            logger.warning("Unknown QA check '%s' -- skipping.", name)
            continue

        if name == "range":
            result = check_range(dataset, thresholds=thresholds)
        elif name == "spatial_completeness":
            result = check_spatial_completeness(
                dataset, threshold=completeness_threshold, time_dim=time_dim
            )
        elif name == "temporal_gaps":
            result = check_temporal_gaps(dataset, time_dim=time_dim)
        else:
            result = _CHECK_REGISTRY[name](dataset)

        report.checks.append(result)

    logger.info(
        "QA report for '%s': %d passed, %d failed.",
        dataset_name,
        report.n_passed,
        report.n_failed,
    )
    return report
