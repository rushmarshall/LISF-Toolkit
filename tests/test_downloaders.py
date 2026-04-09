"""Tests for the downloaders module."""

from __future__ import annotations

from pathlib import Path

import pytest

from lisf_toolkit.downloaders.base import (
    BaseDownloader,
    DownloadConfig,
    DownloadError,
    ValidationError,
)
from lisf_toolkit.downloaders.modis import MODIS_PRODUCTS, MODISDownloader
from lisf_toolkit.downloaders.era5 import ERA5_VARIABLES, ERA5Downloader


# ---------------------------------------------------------------------------
# DownloadConfig
# ---------------------------------------------------------------------------

class TestDownloadConfig:
    """Test the DownloadConfig dataclass."""

    def test_defaults(self):
        cfg = DownloadConfig()
        assert cfg.output_dir == "./data"
        assert cfg.max_retries == 3
        assert cfg.retry_delay == 2.0
        assert cfg.skip_existing is True
        assert cfg.chunk_size == 8192
        assert cfg.verify_checksum is True
        assert cfg.timeout == 120.0

    def test_custom(self):
        cfg = DownloadConfig(output_dir="/custom", max_retries=5, skip_existing=False)
        assert cfg.output_dir == "/custom"
        assert cfg.max_retries == 5
        assert cfg.skip_existing is False


# ---------------------------------------------------------------------------
# Bounding-box validation
# ---------------------------------------------------------------------------

class TestBboxValidation:
    """Test BaseDownloader.validate_bbox."""

    def test_valid_bbox(self):
        BaseDownloader.validate_bbox((-120.0, 35.0, -115.0, 40.0))

    def test_invalid_west_longitude(self):
        with pytest.raises(ValidationError, match="Western longitude"):
            BaseDownloader.validate_bbox((-200.0, 35.0, -115.0, 40.0))

    def test_invalid_east_longitude(self):
        with pytest.raises(ValidationError, match="Eastern longitude"):
            BaseDownloader.validate_bbox((-120.0, 35.0, 200.0, 40.0))

    def test_invalid_south_latitude(self):
        with pytest.raises(ValidationError, match="Southern latitude"):
            BaseDownloader.validate_bbox((-120.0, -100.0, -115.0, 40.0))

    def test_invalid_north_latitude(self):
        with pytest.raises(ValidationError, match="Northern latitude"):
            BaseDownloader.validate_bbox((-120.0, 35.0, -115.0, 100.0))

    def test_west_greater_than_east(self):
        with pytest.raises(ValidationError, match="less than"):
            BaseDownloader.validate_bbox((-100.0, 35.0, -115.0, 40.0))

    def test_south_greater_than_north(self):
        with pytest.raises(ValidationError, match="less than"):
            BaseDownloader.validate_bbox((-120.0, 42.0, -115.0, 40.0))


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

class TestDateValidation:
    """Test BaseDownloader.validate_dates."""

    def test_valid_dates(self):
        s, e = BaseDownloader.validate_dates("2024-01-01", "2024-12-31")
        assert s == "2024-01-01"
        assert e == "2024-12-31"

    def test_same_date(self):
        s, e = BaseDownloader.validate_dates("2024-06-15", "2024-06-15")
        assert s == e

    def test_start_after_end(self):
        with pytest.raises(ValidationError, match="on or before"):
            BaseDownloader.validate_dates("2024-12-31", "2024-01-01")

    def test_bad_format(self):
        with pytest.raises(ValidationError, match="Date format error"):
            BaseDownloader.validate_dates("01-01-2024", "12-31-2024")


# ---------------------------------------------------------------------------
# MODIS product catalogue
# ---------------------------------------------------------------------------

class TestMODISProducts:
    """Test the MODIS product catalogue and metadata."""

    def test_products_not_empty(self):
        assert len(MODIS_PRODUCTS) > 0

    def test_list_products(self):
        products = MODISDownloader.list_products()
        assert "MOD13A2" in products
        assert isinstance(products["MOD13A2"], str)

    def test_product_info_valid(self):
        info = MODISDownloader.product_info("MOD13A2")
        assert info["product"] == "MOD13A2"
        assert "temporal" in info
        assert "spatial" in info
        assert "platform" in info

    def test_product_info_invalid(self):
        with pytest.raises(ValidationError, match="Unknown MODIS product"):
            MODISDownloader.product_info("INVALID_PRODUCT")


# ---------------------------------------------------------------------------
# ERA5 variable catalogue
# ---------------------------------------------------------------------------

class TestERA5Variables:
    """Test the ERA5 variable catalogue."""

    def test_variables_not_empty(self):
        assert len(ERA5_VARIABLES) > 0

    def test_list_variables(self):
        variables = ERA5Downloader.list_variables()
        assert "2m_temperature" in variables

    def test_variable_info_valid(self):
        info = ERA5Downloader.variable_info("2m_temperature")
        assert info["variable"] == "2m_temperature"
        assert "units" in info

    def test_variable_info_invalid(self):
        with pytest.raises(ValidationError, match="Unknown ERA5 variable"):
            ERA5Downloader.variable_info("nonexistent_var")
