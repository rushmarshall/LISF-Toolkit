"""
Map generation utilities for raster land-surface data.

Provides:
    - :func:`plot_raster`  -- Publication-quality static maps with cartopy.
    - :func:`interactive_map` -- Folium-based interactive HTML maps.
    - :func:`comparison_panel` -- Side-by-side or difference maps.

All functions accept :class:`xarray.DataArray` or :class:`numpy.ndarray`
inputs and return the relevant figure or map object.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, xr.DataArray]


# ---------------------------------------------------------------------------
# Static raster maps (matplotlib + cartopy)
# ---------------------------------------------------------------------------

def plot_raster(
    data: ArrayLike,
    *,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    add_colorbar: bool = True,
    colorbar_label: str = "",
    extent: Optional[Tuple[float, float, float, float]] = None,
    figsize: Tuple[float, float] = (10, 8),
    projection: Optional[str] = None,
    coastlines: bool = True,
    gridlines: bool = True,
    dpi: int = 150,
) -> Any:
    """Create a static raster map using matplotlib and cartopy.

    Parameters
    ----------
    data : array-like
        2-D raster data (or an xarray.DataArray with lat/lon coords).
    title : str
        Figure title.
    cmap : str
        Matplotlib colourmap name.
    vmin, vmax : float, optional
        Colourmap limits.  Derived from data when *None*.
    add_colorbar : bool
        Add a colourbar.
    colorbar_label : str
        Label for the colourbar.
    extent : tuple, optional
        Geographic extent ``(west, east, south, north)``.  Inferred from
        xarray coordinates when available.
    figsize : tuple of float
        Figure size in inches ``(width, height)``.
    projection : str, optional
        Cartopy projection name (e.g. ``"PlateCarree"``, ``"Mercator"``).
        Defaults to ``"PlateCarree"``.
    coastlines : bool
        Draw coastlines.
    gridlines : bool
        Draw latitude/longitude grid lines.
    dpi : int
        Figure resolution.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError(
            "matplotlib and cartopy are required for static maps. "
            "Install with:  pip install lisf-toolkit[viz]"
        ) from exc

    # Resolve projection
    proj_map = {
        "PlateCarree": ccrs.PlateCarree,
        "Mercator": ccrs.Mercator,
        "LambertConformal": ccrs.LambertConformal,
        "Robinson": ccrs.Robinson,
    }
    proj_name = projection or "PlateCarree"
    proj_cls = proj_map.get(proj_name)
    if proj_cls is None:
        raise ValueError(
            f"Unknown projection '{proj_name}'. Choose from {sorted(proj_map)}."
        )
    crs = proj_cls()

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": crs},
        dpi=dpi,
    )

    # Data and extent
    if isinstance(data, xr.DataArray):
        arr = data.values
        if extent is None:
            lat_name = _find_coord(data, ("latitude", "lat", "y"))
            lon_name = _find_coord(data, ("longitude", "lon", "x"))
            lats = data[lat_name].values
            lons = data[lon_name].values
            extent = (float(lons.min()), float(lons.max()),
                      float(lats.min()), float(lats.max()))
    else:
        arr = np.asarray(data, dtype=np.float64)

    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    im_kwargs: Dict[str, Any] = {
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "transform": ccrs.PlateCarree(),
    }

    if extent is not None:
        im = ax.imshow(
            arr,
            origin="upper",
            extent=extent,
            **im_kwargs,
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        im = ax.imshow(arr, origin="upper", **im_kwargs)

    if coastlines:
        try:
            ax.coastlines(resolution="50m", linewidth=0.6, color="#444444")
        except Exception:
            ax.coastlines(linewidth=0.6, color="#444444")

    if gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=11)

    fig.tight_layout()
    logger.info("Created static raster map: '%s'.", title or "(untitled)")
    return fig


# ---------------------------------------------------------------------------
# Interactive maps (folium)
# ---------------------------------------------------------------------------

def interactive_map(
    data: ArrayLike,
    *,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "viridis",
    opacity: float = 0.7,
    name: str = "Data",
    zoom_start: int = 7,
) -> Any:
    """Create an interactive HTML map using folium.

    Parameters
    ----------
    data : array-like
        2-D raster data or xarray.DataArray with lat/lon coordinates.
    bounds : tuple, optional
        ``(south, west, north, east)`` for the image overlay.  Inferred
        from xarray coordinates when available.
    cmap : str
        Matplotlib colourmap name used to render the raster.
    opacity : float
        Overlay opacity (0--1).
    name : str
        Layer name shown in the layer control.
    zoom_start : int
        Initial zoom level.

    Returns
    -------
    folium.Map
        Interactive map object.  Call ``.save("map.html")`` to export.
    """
    try:
        import folium
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError as exc:
        raise ImportError(
            "folium and matplotlib are required for interactive maps. "
            "Install with:  pip install lisf-toolkit[viz]"
        ) from exc

    if isinstance(data, xr.DataArray):
        arr = data.values
        if bounds is None:
            lat_name = _find_coord(data, ("latitude", "lat", "y"))
            lon_name = _find_coord(data, ("longitude", "lon", "x"))
            lats = data[lat_name].values
            lons = data[lon_name].values
            bounds = (float(lats.min()), float(lons.min()),
                      float(lats.max()), float(lons.max()))
    else:
        arr = np.asarray(data, dtype=np.float64)

    if bounds is None:
        raise ValueError("bounds must be provided for numpy array inputs.")

    south, west, north, east = bounds
    center = ((south + north) / 2, (west + east) / 2)

    # Normalise and colourise
    norm = mcolors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
    colormap = plt.get_cmap(cmap)
    rgba = colormap(norm(arr))
    rgba[np.isnan(arr)] = [0, 0, 0, 0]  # transparent for NaN
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Encode as PNG via PIL
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for interactive maps: pip install Pillow")

    img = Image.fromarray(rgb_uint8)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    import base64
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[south, west], [north, east]],
        opacity=opacity,
        name=name,
    ).add_to(m)
    folium.LayerControl().add_to(m)

    logger.info("Created interactive map (centre=%.2f, %.2f).", center[0], center[1])
    return m


# ---------------------------------------------------------------------------
# Comparison panels
# ---------------------------------------------------------------------------

def comparison_panel(
    datasets: Sequence[ArrayLike],
    titles: Optional[Sequence[str]] = None,
    *,
    cmap: str = "viridis",
    shared_scale: bool = True,
    suptitle: str = "",
    figsize: Optional[Tuple[float, float]] = None,
) -> Any:
    """Plot multiple rasters side by side for visual comparison.

    Parameters
    ----------
    datasets : sequence of array-like
        Two or more 2-D rasters to compare.
    titles : sequence of str, optional
        Per-panel titles.
    cmap : str
        Colourmap applied to all panels.
    shared_scale : bool
        If *True*, all panels share the same colour scale.
    suptitle : str
        Super-title above all panels.
    figsize : tuple of float, optional
        Figure size.  Defaults to ``(5*n, 5)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for comparison panels. "
            "Install with:  pip install lisf-toolkit[viz]"
        ) from exc

    n = len(datasets)
    if n < 2:
        raise ValueError("comparison_panel requires at least two datasets.")
    if titles and len(titles) != n:
        raise ValueError("Length of titles must match the number of datasets.")

    arrays = [
        d.values if isinstance(d, xr.DataArray) else np.asarray(d, dtype=np.float64)
        for d in datasets
    ]

    if figsize is None:
        figsize = (5 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=120)
    if n == 2:
        axes = list(axes)

    if shared_scale:
        global_min = min(float(np.nanmin(a)) for a in arrays)
        global_max = max(float(np.nanmax(a)) for a in arrays)
    else:
        global_min, global_max = None, None

    for i, (ax, arr) in enumerate(zip(axes, arrays)):
        vmin = global_min if shared_scale else float(np.nanmin(arr))
        vmax = global_max if shared_scale else float(np.nanmax(arr))
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        if titles:
            ax.set_title(titles[i], fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    logger.info("Created comparison panel with %d datasets.", n)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_coord(
    da: xr.DataArray,
    candidates: Tuple[str, ...],
) -> str:
    """Find the first matching coordinate name."""
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    raise ValueError(
        f"Could not find any of {candidates} in DataArray coordinates: "
        f"{list(da.coords)}"
    )
