"""
LISF Toolkit
=============

NASA Land Information System Framework data toolkit for satellite data
acquisition, processing, and analysis.

Provides modules for:
    - downloaders: Authenticated retrieval of satellite and reanalysis data
    - processing: Spatial and temporal transformations of geospatial rasters
    - parameters: Derivation of land surface parameters (vegetation, terrain)
    - quality: Automated QA/QC checks and reporting
    - visualization: Static and interactive map generation

Developed at Hydrosense Lab, University of Virginia.
"""

__version__ = "0.1.0"
__author__ = "Sebastian R.O. Marshall"

from lisf_toolkit import downloaders, processing, parameters, quality, visualization

__all__ = [
    "downloaders",
    "processing",
    "parameters",
    "quality",
    "visualization",
]
