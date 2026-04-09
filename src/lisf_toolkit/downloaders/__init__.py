"""
Data downloaders for satellite observations and reanalysis products.

Supported sources:
    - NASA Earthdata: MODIS land products via ``earthaccess``
    - Copernicus CDS: ERA5 reanalysis via ``cdsapi``

All downloaders inherit from :class:`BaseDownloader` which provides retry
logic, session management, and progress tracking.
"""

from lisf_toolkit.downloaders.base import BaseDownloader, DownloadConfig
from lisf_toolkit.downloaders.modis import MODISDownloader
from lisf_toolkit.downloaders.era5 import ERA5Downloader

__all__ = [
    "BaseDownloader",
    "DownloadConfig",
    "MODISDownloader",
    "ERA5Downloader",
]
