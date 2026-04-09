"""
Land surface parameter derivation from raster datasets.

Modules:
    vegetation -- NDVI, EVI, and LAI from surface reflectance bands.
    terrain    -- Slope, aspect, curvature, and TWI from digital elevation models.
"""

from lisf_toolkit.parameters import vegetation, terrain

__all__ = ["vegetation", "terrain"]
