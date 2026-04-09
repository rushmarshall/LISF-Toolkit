"""
MODIS satellite data downloader using NASA Earthdata and ``earthaccess``.

Supports a wide range of MODIS/Terra and MODIS/Aqua land products including
vegetation indices, LAI/FPAR, land-surface temperature, land-cover, and
surface reflectance.

Authentication is handled transparently through ``earthaccess.login()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import earthaccess

from lisf_toolkit.downloaders.base import (
    AuthenticationError,
    BaseDownloader,
    DownloadConfig,
    DownloadError,
    ValidationError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------

MODIS_PRODUCTS: Dict[str, Dict[str, str]] = {
    "MOD13A2": {
        "description": "MODIS/Terra Vegetation Indices 16-Day L3 Global 1 km",
        "temporal": "16-day",
        "spatial": "1 km",
        "platform": "Terra",
    },
    "MOD13Q1": {
        "description": "MODIS/Terra Vegetation Indices 16-Day L3 Global 250 m",
        "temporal": "16-day",
        "spatial": "250 m",
        "platform": "Terra",
    },
    "MYD13A2": {
        "description": "MODIS/Aqua Vegetation Indices 16-Day L3 Global 1 km",
        "temporal": "16-day",
        "spatial": "1 km",
        "platform": "Aqua",
    },
    "MYD13Q1": {
        "description": "MODIS/Aqua Vegetation Indices 16-Day L3 Global 250 m",
        "temporal": "16-day",
        "spatial": "250 m",
        "platform": "Aqua",
    },
    "MOD15A2H": {
        "description": "MODIS/Terra LAI/FPAR 8-Day L4 Global 500 m",
        "temporal": "8-day",
        "spatial": "500 m",
        "platform": "Terra",
    },
    "MYD15A2H": {
        "description": "MODIS/Aqua LAI/FPAR 8-Day L4 Global 500 m",
        "temporal": "8-day",
        "spatial": "500 m",
        "platform": "Aqua",
    },
    "MOD11A2": {
        "description": "MODIS/Terra LST/Emissivity 8-Day L3 Global 1 km",
        "temporal": "8-day",
        "spatial": "1 km",
        "platform": "Terra",
    },
    "MYD11A2": {
        "description": "MODIS/Aqua LST/Emissivity 8-Day L3 Global 1 km",
        "temporal": "8-day",
        "spatial": "1 km",
        "platform": "Aqua",
    },
    "MCD12Q1": {
        "description": "MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500 m",
        "temporal": "yearly",
        "spatial": "500 m",
        "platform": "Terra+Aqua",
    },
    "MOD09A1": {
        "description": "MODIS/Terra Surface Reflectance 8-Day L3 Global 500 m",
        "temporal": "8-day",
        "spatial": "500 m",
        "platform": "Terra",
    },
    "MYD09A1": {
        "description": "MODIS/Aqua Surface Reflectance 8-Day L3 Global 500 m",
        "temporal": "8-day",
        "spatial": "500 m",
        "platform": "Aqua",
    },
    "MOD16A2": {
        "description": "MODIS/Terra Evapotranspiration 8-Day L4 Global 500 m",
        "temporal": "8-day",
        "spatial": "500 m",
        "platform": "Terra",
    },
    "MOD44B": {
        "description": "MODIS/Terra Vegetation Continuous Fields Yearly L3 Global 250 m",
        "temporal": "yearly",
        "spatial": "250 m",
        "platform": "Terra",
    },
}


class MODISDownloader(BaseDownloader):
    """Download MODIS land products from NASA Earthdata.

    Uses the ``earthaccess`` library for authentication, granule search, and
    file retrieval.  Credentials are resolved automatically by earthaccess
    (environment variables, ``.netrc``, or interactive prompt).

    Parameters
    ----------
    config : DownloadConfig, optional
        Download configuration (output directory, retries, etc.).

    Examples
    --------
    >>> from lisf_toolkit.downloaders import MODISDownloader, DownloadConfig
    >>> dl = MODISDownloader(DownloadConfig(output_dir="./data/modis"))
    >>> files = dl.download("MOD13A2", (-80, 37, -78, 39), "2024-01-01", "2024-03-31")
    """

    def __init__(self, config: Optional[DownloadConfig] = None) -> None:
        super().__init__(config)
        self._authenticated = False

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        """Authenticate with NASA Earthdata if not already done.

        ``earthaccess.login()`` checks, in order, environment variables
        (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), a ``.netrc``
        file, and finally an interactive prompt.
        """
        if self._authenticated:
            return
        try:
            auth = earthaccess.login(strategy="environment")
            if auth is None:
                auth = earthaccess.login(strategy="netrc")
            if auth is None:
                auth = earthaccess.login(strategy="interactive")
            self._authenticated = True
            self._logger.info("Authenticated with NASA Earthdata.")
        except Exception as exc:
            raise AuthenticationError(
                f"NASA Earthdata authentication failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Product metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_products() -> Dict[str, str]:
        """Return a mapping of product short-names to descriptions."""
        return {code: meta["description"] for code, meta in MODIS_PRODUCTS.items()}

    @staticmethod
    def product_info(product: str) -> Dict[str, str]:
        """Return metadata for a single MODIS product.

        Raises
        ------
        ValidationError
            If the product code is not recognised.
        """
        if product not in MODIS_PRODUCTS:
            raise ValidationError(
                f"Unknown MODIS product '{product}'. "
                f"Supported: {sorted(MODIS_PRODUCTS)}"
            )
        return {**MODIS_PRODUCTS[product], "product": product}

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
        max_granules: Optional[int] = None,
        output_subdir: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Path]:
        """Download MODIS granules for a product, region, and time range.

        Parameters
        ----------
        product : str
            MODIS short name (e.g. ``"MOD13A2"``).
        bbox : tuple of float
            *(west, south, east, north)* in decimal degrees.
        start_date, end_date : str
            ISO-format date strings (``YYYY-MM-DD``).
        max_granules : int, optional
            Cap the number of granules downloaded (useful for testing).
        output_subdir : str, optional
            Override the default per-product sub-directory name.

        Returns
        -------
        list of Path
            Paths to the downloaded files on disk.

        Raises
        ------
        ValidationError
            If inputs are invalid.
        AuthenticationError
            If Earthdata login fails.
        DownloadError
            If the search or download operation fails.
        """
        # Validate inputs
        if product not in MODIS_PRODUCTS:
            raise ValidationError(
                f"Unsupported product '{product}'. "
                f"Supported: {sorted(MODIS_PRODUCTS)}"
            )
        self.validate_bbox(bbox)
        self.validate_dates(start_date, end_date)

        self._ensure_authenticated()

        save_dir = self.output_dir / (output_subdir or product.lower())
        save_dir.mkdir(parents=True, exist_ok=True)

        self._logger.info(
            "Searching %s | bbox=%s | %s to %s", product, bbox, start_date, end_date
        )

        try:
            results = earthaccess.search_data(
                short_name=product,
                temporal=(start_date, end_date),
                bounding_box=bbox,
            )
        except Exception as exc:
            raise DownloadError(f"Granule search failed for {product}: {exc}") from exc

        if not results:
            self._logger.warning("No granules matched the query.")
            return []

        self._logger.info("Found %d granule(s).", len(results))

        if max_granules is not None and len(results) > max_granules:
            self._logger.info("Limiting to %d granule(s).", max_granules)
            results = results[:max_granules]

        downloaded: List[Path] = []
        for idx, granule in enumerate(results, 1):
            self._logger.info("Downloading granule %d / %d ...", idx, len(results))
            try:
                local_files = earthaccess.download(granule, str(save_dir))
                if local_files:
                    downloaded.extend(Path(f) for f in local_files)
            except Exception as exc:
                self._logger.error("Failed to download granule %d: %s", idx, exc)
                # Continue with remaining granules instead of aborting
                continue

        self._logger.info(
            "Download complete: %d file(s) saved to %s", len(downloaded), save_dir
        )
        return downloaded
