"""
ERA5 reanalysis downloader via the Copernicus Climate Data Store API.

Retrieves ERA5 hourly and monthly-mean single-level and pressure-level
variables using the ``cdsapi`` client.  Supports sub-setting by area,
time range, and variable selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lisf_toolkit.downloaders.base import (
    BaseDownloader,
    DownloadConfig,
    DownloadError,
    ValidationError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable catalogue
# ---------------------------------------------------------------------------

ERA5_VARIABLES: Dict[str, Dict[str, str]] = {
    "2m_temperature": {
        "long_name": "2-metre temperature",
        "units": "K",
        "dataset": "reanalysis-era5-single-levels",
    },
    "total_precipitation": {
        "long_name": "Total precipitation",
        "units": "m",
        "dataset": "reanalysis-era5-single-levels",
    },
    "10m_u_component_of_wind": {
        "long_name": "10-metre U wind component",
        "units": "m s-1",
        "dataset": "reanalysis-era5-single-levels",
    },
    "10m_v_component_of_wind": {
        "long_name": "10-metre V wind component",
        "units": "m s-1",
        "dataset": "reanalysis-era5-single-levels",
    },
    "surface_pressure": {
        "long_name": "Surface pressure",
        "units": "Pa",
        "dataset": "reanalysis-era5-single-levels",
    },
    "2m_dewpoint_temperature": {
        "long_name": "2-metre dewpoint temperature",
        "units": "K",
        "dataset": "reanalysis-era5-single-levels",
    },
    "surface_solar_radiation_downwards": {
        "long_name": "Surface solar radiation downwards",
        "units": "J m-2",
        "dataset": "reanalysis-era5-single-levels",
    },
    "surface_thermal_radiation_downwards": {
        "long_name": "Surface thermal radiation downwards",
        "units": "J m-2",
        "dataset": "reanalysis-era5-single-levels",
    },
    "soil_temperature_level_1": {
        "long_name": "Soil temperature level 1",
        "units": "K",
        "dataset": "reanalysis-era5-single-levels",
    },
    "volumetric_soil_water_layer_1": {
        "long_name": "Volumetric soil water layer 1",
        "units": "m3 m-3",
        "dataset": "reanalysis-era5-single-levels",
    },
}


def _months_in_range(start: str, end: str) -> List[Tuple[int, int]]:
    """Return a list of (year, month) tuples spanning *start* to *end*."""
    from datetime import datetime

    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    months: List[Tuple[int, int]] = []
    year, month = s.year, s.month
    while (year, month) <= (e.year, e.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


class ERA5Downloader(BaseDownloader):
    """Download ERA5 reanalysis data from the Copernicus Climate Data Store.

    Requires the ``cdsapi`` package and a valid CDS API key stored at
    ``~/.cdsapirc`` (see https://cds.climate.copernicus.eu/how-to-api).

    Parameters
    ----------
    config : DownloadConfig, optional
        Download configuration (output directory, retries, etc.).

    Examples
    --------
    >>> from lisf_toolkit.downloaders import ERA5Downloader, DownloadConfig
    >>> dl = ERA5Downloader(DownloadConfig(output_dir="./data/era5"))
    >>> files = dl.download(
    ...     product="2m_temperature",
    ...     bbox=(-80, 37, -78, 39),
    ...     start_date="2024-01-01",
    ...     end_date="2024-01-31",
    ... )
    """

    def __init__(self, config: Optional[DownloadConfig] = None) -> None:
        super().__init__(config)
        self._client: Any = None

    # ------------------------------------------------------------------
    # CDS client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily create a ``cdsapi.Client`` instance."""
        if self._client is not None:
            return self._client
        try:
            import cdsapi  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'cdsapi' package is required for ERA5 downloads. "
                "Install it with:  pip install lisf-toolkit[cds]"
            ) from exc
        self._client = cdsapi.Client(quiet=True)
        self._logger.info("CDS API client initialised.")
        return self._client

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_variables() -> Dict[str, str]:
        """Return a mapping of variable names to long descriptions."""
        return {k: v["long_name"] for k, v in ERA5_VARIABLES.items()}

    @staticmethod
    def variable_info(variable: str) -> Dict[str, str]:
        """Return metadata for a single ERA5 variable.

        Raises
        ------
        ValidationError
            If the variable name is not recognised.
        """
        if variable not in ERA5_VARIABLES:
            raise ValidationError(
                f"Unknown ERA5 variable '{variable}'. "
                f"Supported: {sorted(ERA5_VARIABLES)}"
            )
        return {**ERA5_VARIABLES[variable], "variable": variable}

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        product: str,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        *,
        hours: Optional[Sequence[str]] = None,
        output_format: str = "netcdf",
        output_subdir: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Path]:
        """Download ERA5 data for a variable, region, and time range.

        Data is retrieved in monthly chunks to keep individual requests
        within the CDS size limits.

        Parameters
        ----------
        product : str
            ERA5 variable name (e.g. ``"2m_temperature"``).
        bbox : tuple of float
            *(west, south, east, north)* in decimal degrees.
        start_date, end_date : str
            ISO-format date strings (``YYYY-MM-DD``).
        hours : sequence of str, optional
            List of hours to retrieve (``["00:00", "06:00", ...]``).
            Defaults to all 24 hours.
        output_format : str
            ``"netcdf"`` (default) or ``"grib"``.
        output_subdir : str, optional
            Override the default per-variable sub-directory name.

        Returns
        -------
        list of Path
            Paths to the downloaded files on disk.

        Raises
        ------
        ValidationError
            If inputs are invalid.
        DownloadError
            If the CDS request fails.
        """
        if product not in ERA5_VARIABLES:
            raise ValidationError(
                f"Unsupported ERA5 variable '{product}'. "
                f"Supported: {sorted(ERA5_VARIABLES)}"
            )
        self.validate_bbox(bbox)
        self.validate_dates(start_date, end_date)

        client = self._get_client()

        dataset = ERA5_VARIABLES[product]["dataset"]
        west, south, east, north = bbox
        # CDS area format: [north, west, south, east]
        area = [north, west, south, east]

        if hours is None:
            hours = [f"{h:02d}:00" for h in range(24)]

        ext = "nc" if output_format == "netcdf" else "grib"
        save_dir = self.output_dir / (output_subdir or product)
        save_dir.mkdir(parents=True, exist_ok=True)

        months = _months_in_range(start_date, end_date)
        self._logger.info(
            "Requesting %s for %d month(s) | area=%s", product, len(months), area
        )

        downloaded: List[Path] = []
        for year, month in months:
            filename = f"{product}_{year}{month:02d}.{ext}"
            dest = save_dir / filename

            if self.config.skip_existing and dest.exists():
                self._logger.info("Skipping existing: %s", dest.name)
                downloaded.append(dest)
                continue

            days = [f"{d:02d}" for d in range(1, 32)]

            request: Dict[str, Any] = {
                "product_type": "reanalysis",
                "variable": product,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": list(hours),
                "area": area,
                "format": output_format,
            }

            self._logger.info("Submitting CDS request for %s ...", filename)
            try:
                client.retrieve(dataset, request, str(dest))
                self._logger.info("Saved %s", dest)
                downloaded.append(dest)
            except Exception as exc:
                self._logger.error("CDS request failed for %s: %s", filename, exc)
                dest.unlink(missing_ok=True)
                continue

        self._logger.info(
            "ERA5 download complete: %d file(s) in %s", len(downloaded), save_dir
        )
        return downloaded
