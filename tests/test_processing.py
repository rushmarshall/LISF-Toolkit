"""Tests for the processing and parameters modules."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from lisf_toolkit.parameters import vegetation, terrain
from lisf_toolkit.processing import temporal
from lisf_toolkit.quality import validation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(
    ntime: int = 12,
    nlat: int = 10,
    nlon: int = 10,
    var_name: str = "ndvi",
    seed: int = 42,
) -> xr.Dataset:
    """Create a synthetic xarray Dataset for testing."""
    rng = np.random.default_rng(seed)
    times = xr.cftime_range("2024-01-01", periods=ntime, freq="ME")
    lats = np.linspace(37.0, 39.0, nlat)
    lons = np.linspace(-80.0, -78.0, nlon)
    data = rng.uniform(0.1, 0.9, size=(ntime, nlat, nlon))
    return xr.Dataset(
        {var_name: (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def _make_dem(nrows: int = 50, ncols: int = 50) -> np.ndarray:
    """Create a synthetic DEM (tilted plane + noise)."""
    rng = np.random.default_rng(0)
    y = np.arange(nrows, dtype=np.float64)
    x = np.arange(ncols, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return 500.0 + 2.0 * yy + 1.5 * xx + rng.normal(0, 0.5, (nrows, ncols))


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------

class TestVegetation:
    """Test vegetation index computations."""

    def test_ndvi_basic(self):
        nir = np.array([0.5, 0.8, 0.3])
        red = np.array([0.1, 0.2, 0.3])
        ndvi = vegetation.calculate_ndvi(nir, red)
        expected = (nir - red) / (nir + red)
        np.testing.assert_allclose(ndvi, expected, atol=1e-10)

    def test_ndvi_range(self):
        nir = np.random.default_rng(0).uniform(0, 1, 100)
        red = np.random.default_rng(1).uniform(0, 1, 100)
        ndvi = vegetation.calculate_ndvi(nir, red)
        assert np.all(ndvi[~np.isnan(ndvi)] >= -1.0)
        assert np.all(ndvi[~np.isnan(ndvi)] <= 1.0)

    def test_ndvi_zero_denominator(self):
        nir = np.array([0.0])
        red = np.array([0.0])
        ndvi = vegetation.calculate_ndvi(nir, red)
        assert np.isnan(ndvi[0])

    def test_ndvi_nodata(self):
        nir = np.array([0.5, -9999.0])
        red = np.array([0.1, 0.2])
        ndvi = vegetation.calculate_ndvi(nir, red, nodata=-9999.0)
        assert not np.isnan(ndvi[0])
        assert np.isnan(ndvi[1])

    def test_ndvi_xarray(self):
        da = xr.DataArray(np.array([0.5, 0.8]), dims=["pixel"])
        red = xr.DataArray(np.array([0.1, 0.2]), dims=["pixel"])
        result = vegetation.calculate_ndvi(da, red)
        assert isinstance(result, xr.DataArray)

    def test_evi_basic(self):
        nir = np.array([0.5])
        red = np.array([0.1])
        blue = np.array([0.05])
        evi = vegetation.calculate_evi(nir, red, blue)
        assert evi.shape == (1,)
        assert -1.0 <= evi[0] <= 1.0

    def test_lai_from_ndvi(self):
        ndvi = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        lai = vegetation.estimate_lai(ndvi)
        assert lai.shape == ndvi.shape
        # Higher NDVI should give higher LAI
        assert lai[-1] > lai[0]


# ---------------------------------------------------------------------------
# Terrain parameters
# ---------------------------------------------------------------------------

class TestTerrain:
    """Test terrain parameter derivation."""

    def test_slope_flat(self):
        flat = np.ones((10, 10)) * 100.0
        slope = terrain.calculate_slope(flat, cell_size=30.0)
        np.testing.assert_allclose(slope, 0.0, atol=1e-10)

    def test_slope_positive(self):
        dem = _make_dem()
        slope = terrain.calculate_slope(dem, cell_size=30.0, units="degrees")
        assert np.all(slope >= 0)

    def test_aspect_shape(self):
        dem = _make_dem()
        aspect = terrain.calculate_aspect(dem, cell_size=30.0)
        assert aspect.shape == dem.shape

    def test_curvature_shape(self):
        dem = _make_dem()
        curv = terrain.calculate_curvature(dem, cell_size=30.0)
        assert curv.shape == dem.shape

    def test_twi_shape(self):
        dem = _make_dem()
        twi = terrain.topographic_wetness_index(dem, cell_size=30.0)
        assert twi.shape == dem.shape
        assert np.all(np.isfinite(twi))

    def test_slope_xarray(self):
        dem_np = _make_dem(20, 20)
        da = xr.DataArray(dem_np, dims=["y", "x"])
        slope = terrain.calculate_slope(da, cell_size=30.0)
        assert isinstance(slope, xr.DataArray)

    def test_flow_accumulation(self):
        dem = _make_dem(20, 20)
        acc = terrain.flow_accumulation_d8(dem)
        assert acc.shape == dem.shape
        assert acc.min() >= 1  # every cell counts itself


# ---------------------------------------------------------------------------
# Temporal processing
# ---------------------------------------------------------------------------

class TestTemporal:
    """Test temporal aggregation and gap filling."""

    def test_aggregate_mean(self):
        ds = _make_dataset(ntime=12)
        result = temporal.aggregate(ds, freq="QE", method="mean")
        assert result.sizes["time"] <= 12

    def test_monthly_climatology(self):
        ds = _make_dataset(ntime=24)
        clim = temporal.monthly_climatology(ds)
        assert "month" in clim.dims

    def test_anomalies(self):
        ds = _make_dataset(ntime=24)
        anom = temporal.compute_anomalies(ds)
        assert anom["ndvi"].shape == ds["ndvi"].shape

    def test_rolling_stat(self):
        ds = _make_dataset(ntime=12)
        result = temporal.rolling_stat(ds, window=3, method="mean")
        assert result.sizes["time"] == 12

    def test_detect_gaps_none(self):
        ds = _make_dataset(ntime=12)
        gaps = temporal.detect_gaps(ds)
        assert len(gaps) == 0

    def test_detect_gaps_present(self):
        ds = _make_dataset(ntime=12)
        # Introduce a gap
        ds["ndvi"].values[5, :, :] = np.nan
        gaps = temporal.detect_gaps(ds)
        assert "ndvi" in gaps

    def test_fill_gaps(self):
        ds = _make_dataset(ntime=12)
        ds["ndvi"].values[5, :, :] = np.nan
        filled = temporal.fill_gaps(ds, method="linear")
        assert int(filled["ndvi"].isnull().sum()) < int(ds["ndvi"].isnull().sum())


# ---------------------------------------------------------------------------
# Quality validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Test QA/QC checks."""

    def test_run_qa_all_pass(self):
        ds = _make_dataset(ntime=12, var_name="ndvi")
        report = validation.run_qa(ds, dataset_name="test")
        assert report.n_failed == 0

    def test_range_check_fail(self):
        ds = _make_dataset(ntime=12, var_name="ndvi")
        ds["ndvi"].values[0, 0, 0] = 5.0  # out of [-1, 1]
        result = validation.check_range(ds)
        assert not result.passed

    def test_summary_statistics(self):
        ds = _make_dataset(ntime=12)
        stats = validation.summary_statistics(ds)
        assert "ndvi" in stats
        assert "mean" in stats["ndvi"]
        assert "nan_fraction" in stats["ndvi"]

    def test_report_json(self):
        ds = _make_dataset(ntime=12)
        report = validation.run_qa(ds, dataset_name="json_test")
        json_str = report.to_json()
        assert "json_test" in json_str

    def test_report_summary(self):
        ds = _make_dataset(ntime=12)
        report = validation.run_qa(ds, dataset_name="summary_test")
        summary = report.summary()
        assert "summary_test" in summary
