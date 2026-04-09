"""
Base downloader with retry logic, authentication, and session management.

All concrete downloaders inherit from :class:`BaseDownloader` and implement
the :meth:`download` method for their respective data sources.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DownloadError(Exception):
    """Raised when a file download fails after all retry attempts."""


class AuthenticationError(Exception):
    """Raised when authentication with a remote data provider fails."""


class ValidationError(Exception):
    """Raised when input parameters fail validation."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DownloadConfig:
    """Configuration for data download operations.

    Parameters
    ----------
    output_dir : str
        Root directory for downloaded files.  Sub-directories are created
        automatically per product.
    max_retries : int
        Maximum number of retry attempts for transient failures.
    retry_delay : float
        Initial delay in seconds between retries.  Each subsequent retry
        doubles the delay (exponential back-off).
    skip_existing : bool
        When *True*, files that already exist on disk are not re-downloaded.
    chunk_size : int
        Byte size of streaming download chunks.
    verify_checksum : bool
        When *True*, downloaded files are verified against server-reported
        checksums when available.
    timeout : float
        HTTP request timeout in seconds.
    max_concurrent : int
        Maximum number of concurrent downloads (reserved for future use).
    """

    output_dir: str = "./data"
    max_retries: int = 3
    retry_delay: float = 2.0
    skip_existing: bool = True
    chunk_size: int = 8192
    verify_checksum: bool = True
    timeout: float = 120.0
    max_concurrent: int = 4


# ---------------------------------------------------------------------------
# Base downloader
# ---------------------------------------------------------------------------

class BaseDownloader(ABC):
    """Abstract base class for all data downloaders.

    Provides common functionality shared across concrete implementations:

    * HTTP session management with optional authentication
    * Bounding-box and date-range validation
    * File download with streaming, progress bars, and retry logic
    * Checksum verification
    * Logging throughout

    Subclasses must implement :meth:`download`.

    Parameters
    ----------
    config : DownloadConfig, optional
        Download configuration.  Defaults are used when *None*.
    """

    def __init__(self, config: Optional[DownloadConfig] = None) -> None:
        self.config = config or DownloadConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._session: Optional[requests.Session] = None
        self._logger = logging.getLogger(self.__class__.__qualname__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    @property
    def session(self) -> requests.Session:
        """Lazily initialised HTTP session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _set_auth(self, username: str, password: str) -> None:
        """Assign Basic-Auth credentials to the HTTP session."""
        self.session.auth = (username, password)
        self._logger.info("HTTP session credentials configured.")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_bbox(bbox: Tuple[float, float, float, float]) -> None:
        """Validate a bounding box expressed as *(west, south, east, north)*.

        Raises
        ------
        ValidationError
            If any coordinate is out of range or the box is degenerate.
        """
        west, south, east, north = bbox
        if not (-180.0 <= west <= 180.0):
            raise ValidationError(f"Western longitude {west} out of range [-180, 180].")
        if not (-180.0 <= east <= 180.0):
            raise ValidationError(f"Eastern longitude {east} out of range [-180, 180].")
        if not (-90.0 <= south <= 90.0):
            raise ValidationError(f"Southern latitude {south} out of range [-90, 90].")
        if not (-90.0 <= north <= 90.0):
            raise ValidationError(f"Northern latitude {north} out of range [-90, 90].")
        if west >= east:
            raise ValidationError("Western longitude must be less than eastern longitude.")
        if south >= north:
            raise ValidationError("Southern latitude must be less than northern latitude.")

    @staticmethod
    def validate_dates(start_date: str, end_date: str, fmt: str = "%Y-%m-%d") -> Tuple[str, str]:
        """Parse and validate a date range.

        Returns the validated strings unchanged so callers can pass them
        directly to upstream APIs.

        Raises
        ------
        ValidationError
            If the format is wrong or start > end.
        """
        from datetime import datetime

        try:
            start_dt = datetime.strptime(start_date, fmt)
            end_dt = datetime.strptime(end_date, fmt)
        except ValueError as exc:
            raise ValidationError(f"Date format error (expected {fmt}): {exc}") from exc
        if start_dt > end_dt:
            raise ValidationError("start_date must be on or before end_date.")
        return start_date, end_date

    # ------------------------------------------------------------------
    # Download with retry
    # ------------------------------------------------------------------

    def download_file(
        self,
        url: str,
        dest: Path,
        *,
        show_progress: bool = True,
        expected_md5: Optional[str] = None,
    ) -> Path:
        """Download a single file with exponential-backoff retry.

        Parameters
        ----------
        url : str
            Remote URL to fetch.
        dest : Path
            Local destination path.
        show_progress : bool
            Show a ``tqdm`` progress bar.
        expected_md5 : str, optional
            If provided, the downloaded file is verified against this MD5 hash.

        Returns
        -------
        Path
            The *dest* path after successful download.

        Raises
        ------
        DownloadError
            After all retry attempts are exhausted.
        """
        if self.config.skip_existing and dest.exists():
            self._logger.info("Skipping existing file: %s", dest.name)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._logger.info(
                    "Downloading %s (attempt %d/%d)", dest.name, attempt, self.config.max_retries
                )
                response = self.session.get(
                    url, stream=True, timeout=self.config.timeout
                )
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                hasher = hashlib.md5() if self.config.verify_checksum else None

                with open(dest, "wb") as fh:
                    iterator = response.iter_content(chunk_size=self.config.chunk_size)
                    if show_progress and total > 0:
                        iterator = tqdm(
                            iterator,
                            total=total,
                            unit="B",
                            unit_scale=True,
                            desc=dest.name,
                            leave=False,
                        )
                    for chunk in iterator:
                        if chunk:
                            fh.write(chunk)
                            if hasher is not None:
                                hasher.update(chunk)

                # Checksum verification
                if expected_md5 and hasher is not None:
                    computed = hasher.hexdigest()
                    if computed != expected_md5:
                        dest.unlink(missing_ok=True)
                        raise DownloadError(
                            f"Checksum mismatch for {dest.name}: "
                            f"expected {expected_md5}, got {computed}"
                        )

                self._logger.info("Saved %s (%s bytes)", dest.name, dest.stat().st_size)
                return dest

            except (requests.RequestException, DownloadError) as exc:
                last_exc = exc
                delay = self.config.retry_delay * (2 ** (attempt - 1))
                self._logger.warning(
                    "Attempt %d failed (%s). Retrying in %.1fs ...", attempt, exc, delay
                )
                time.sleep(delay)

        raise DownloadError(
            f"Failed to download {url} after {self.config.max_retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def download(
        self,
        product: str,
        bbox: Tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> List[Path]:
        """Download data for a given product, region, and time range.

        Concrete subclasses must implement this method.
        """
        ...
