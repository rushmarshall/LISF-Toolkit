"""
Geospatial data processing: spatial and temporal transformations.

Modules:
    spatial  -- Regridding, resampling, clipping, and zonal statistics.
    temporal -- Temporal aggregation, rolling windows, and gap filling.
"""

from lisf_toolkit.processing import spatial, temporal

__all__ = ["spatial", "temporal"]
