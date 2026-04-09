"""
LISF Toolkit -- Quick Start Example
====================================

Demonstrates core capabilities of the toolkit using synthetic data.
No network access or external credentials are required.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from lisf_toolkit.parameters import vegetation, terrain
from lisf_toolkit.processing import temporal
from lisf_toolkit.quality import validation


def make_synthetic_reflectance(
    ntime: int = 24,
    nlat: int = 50,
    nlon: int = 50,
) -> xr.Dataset:
    """Generate a synthetic surface-reflectance dataset."""
    rng = np.random.default_rng(42)
    times = xr.cftime_range("2023-01-01", periods=ntime, freq="ME")
    lats = np.linspace(37.0, 39.0, nlat)
    lons = np.linspace(-80.0, -78.0, nlon)

    nir = rng.uniform(0.2, 0.8, size=(ntime, nlat, nlon))
    red = rng.uniform(0.05, 0.3, size=(ntime, nlat, nlon))
    blue = rng.uniform(0.02, 0.15, size=(ntime, nlat, nlon))

    return xr.Dataset(
        {
            "nir": (["time", "latitude", "longitude"], nir),
            "red": (["time", "latitude", "longitude"], red),
            "blue": (["time", "latitude", "longitude"], blue),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def main() -> None:
    print("LISF Toolkit -- Quick Start\n")

    # ---------------------------------------------------------------
    # 1. Compute vegetation indices
    # ---------------------------------------------------------------
    print("[1/4] Computing vegetation indices ...")
    ds = make_synthetic_reflectance()
    ndvi = vegetation.calculate_ndvi(ds["nir"], ds["red"])
    evi = vegetation.calculate_evi(ds["nir"], ds["red"], ds["blue"])
    lai = vegetation.estimate_lai(ndvi)
    print(f"  NDVI range: [{float(ndvi.min()):.3f}, {float(ndvi.max()):.3f}]")
    print(f"  EVI  range: [{float(evi.min()):.3f}, {float(evi.max()):.3f}]")
    print(f"  LAI  range: [{float(lai.min()):.3f}, {float(lai.max()):.3f}]")

    # ---------------------------------------------------------------
    # 2. Temporal aggregation
    # ---------------------------------------------------------------
    print("\n[2/4] Temporal aggregation ...")
    ndvi_ds = xr.Dataset({"ndvi": ndvi})
    quarterly = temporal.aggregate(ndvi_ds, freq="QE", method="mean")
    clim = temporal.monthly_climatology(ndvi_ds)
    print(f"  Quarterly time steps: {quarterly.sizes['time']}")
    print(f"  Climatology months:   {clim.sizes['month']}")

    # ---------------------------------------------------------------
    # 3. Terrain analysis
    # ---------------------------------------------------------------
    print("\n[3/4] Terrain analysis ...")
    rng = np.random.default_rng(0)
    y = np.arange(100, dtype=np.float64)
    x = np.arange(100, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dem = 800.0 + 3.0 * yy + 2.0 * xx + rng.normal(0, 1, (100, 100))

    slope = terrain.calculate_slope(dem, cell_size=30.0, units="degrees")
    aspect = terrain.calculate_aspect(dem, cell_size=30.0)
    twi = terrain.topographic_wetness_index(dem, cell_size=30.0)
    print(f"  Slope range:  [{slope.min():.2f}, {slope.max():.2f}] deg")
    print(f"  Aspect range: [{aspect.min():.1f}, {aspect.max():.1f}] deg")
    print(f"  TWI range:    [{twi.min():.2f}, {twi.max():.2f}]")

    # ---------------------------------------------------------------
    # 4. Quality control
    # ---------------------------------------------------------------
    print("\n[4/4] Running QA checks ...")
    report = validation.run_qa(ndvi_ds, dataset_name="Synthetic NDVI")
    print(report.summary())

    print("\nDone.")


if __name__ == "__main__":
    main()
