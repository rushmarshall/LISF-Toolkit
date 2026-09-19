<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:111111,30:333333,60:666666,100:999999&height=220&section=header&text=LISF%20Toolkit&fontSize=56&fontColor=ffffff&fontAlignY=38&desc=NASA%20Land%20Information%20System%20Framework%20Data%20Toolkit&descSize=16&descAlignY=58&descColor=cccccc&animation=fadeIn" width="100%" alt="LISF Toolkit" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-333333?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-333333?style=flat-square" alt="License" /></a>
  <a href="https://github.com/rushmarshall/LISF-Toolkit/actions"><img src="https://img.shields.io/badge/CI-passing-333333?style=flat-square&logo=githubactions&logoColor=white" alt="CI" /></a>
  <a href="https://www.nasa.gov/"><img src="https://img.shields.io/badge/NASA-LISF-333333?style=flat-square&logo=nasa&logoColor=white" alt="NASA" /></a>
  <a href="https://github.com/rushmarshall/LISF-Toolkit"><img src="https://img.shields.io/badge/Status-Active-333333?style=flat-square" alt="Status" /></a>
</p>

<br/>

## Overview

LISF Toolkit is a Python library for acquiring, processing, and analyzing land surface data from satellite observations and reanalysis products. It is designed to streamline the data preparation pipeline for the [NASA Land Information System Framework (LISF)](https://github.com/NASA-LIS/LISF), enabling researchers to focus on science rather than data wrangling.

The toolkit provides end-to-end functionality: authenticated satellite data downloads with retry logic, spatial and temporal processing of geospatial rasters, land surface parameter derivation, automated quality control, and publication-ready visualization.

---

## Installation

### From source

```bash
git clone https://github.com/rushmarshall/LISF-Toolkit.git
cd LISF-Toolkit
pip install -e ".[dev]"
```

### Dependencies only

```bash
pip install -e .
```

### Optional dependency groups

```bash
pip install -e ".[viz]"       # Visualization extras (cartopy, folium, plotly)
pip install -e ".[ml]"        # Machine learning (scikit-learn, xgboost)
pip install -e ".[dev]"       # Development tools (pytest, black, mypy)
pip install -e ".[all]"       # Everything
```

---

## Quick Start

### Download MODIS vegetation data

```python
from lisf_toolkit.downloaders import MODISDownloader, DownloadConfig

config = DownloadConfig(
    output_dir="./data/modis",
    max_retries=3,
    skip_existing=True,
)

downloader = MODISDownloader(config=config)
files = downloader.download(
    product="MOD13A2",
    bbox=(-80.0, 37.0, -78.0, 39.0),
    start_date="2024-01-01",
    end_date="2024-06-30",
)
```

### Process spatial data

```python
from lisf_toolkit.processing import spatial

# Regrid a dataset to a target resolution
ds_regridded = spatial.regrid(
    dataset=ds,
    target_resolution=0.05,
    method="bilinear",
)

# Clip to a watershed boundary
ds_clipped = spatial.clip_to_shapefile(
    dataset=ds_regridded,
    shapefile="watersheds/james_river.shp",
)
```

### Derive terrain parameters from a DEM

```python
from lisf_toolkit.parameters import terrain

slope = terrain.calculate_slope(dem, units="degrees")
aspect = terrain.calculate_aspect(dem)
twi = terrain.topographic_wetness_index(dem)
```

### Run quality control checks

```python
from lisf_toolkit.quality import validation

report = validation.run_qa(
    dataset=ds,
    checks=["range", "spatial_completeness", "temporal_gaps"],
)
report.summary()
```

### Generate a map

```python
from lisf_toolkit.visualization import maps

fig = maps.plot_raster(
    data=ndvi,
    title="NDVI Composite -- James River Basin",
    cmap="YlGn",
    add_colorbar=True,
)
fig.savefig("ndvi_map.png", dpi=300, bbox_inches="tight")
```

---

## Features

### Data Acquisition
- MODIS land products via NASA Earthdata (earthaccess)
- ERA5 reanalysis from the Copernicus Climate Data Store
- Authenticated downloads with exponential backoff retry
- Concurrent file transfers with progress tracking
- Automatic deduplication and checksum verification

### Geospatial Processing
- Regridding and resampling with configurable interpolation methods
- Shapefile-based spatial masking and clipping
- Coordinate reference system transformations
- Zonal statistics aggregation

### Parameter Derivation
- Vegetation indices: NDVI, EVI, LAI from surface reflectance
- Terrain analysis: slope, aspect, curvature, Topographic Wetness Index
- All computations backed by NumPy with proper nodata handling

### Quality Control
- Configurable range checks with per-variable thresholds
- Spatial completeness assessment
- Temporal gap detection and interpolation-based filling
- Summary statistics and structured QA reports

### Visualization
- Static maps with cartopy and matplotlib
- Interactive HTML maps with folium
- Side-by-side comparison panels
- Consistent academic styling throughout

---

## Architecture

```
lisf_toolkit/
|
|-- downloaders/        Authenticated satellite and reanalysis data retrieval
|   |-- base.py         Abstract base with retry logic and session management
|   |-- modis.py        MODIS products via earthaccess
|   |-- era5.py         ERA5 reanalysis via CDS API
|
|-- processing/         Geospatial data transformations
|   |-- spatial.py      Regridding, clipping, masking, zonal statistics
|   |-- temporal.py     Aggregation, rolling windows, gap filling
|
|-- parameters/         Land surface parameter derivation
|   |-- vegetation.py   NDVI, EVI, LAI from reflectance bands
|   |-- terrain.py      Slope, aspect, curvature, TWI from DEMs
|
|-- quality/            Data validation and quality assurance
|   |-- validation.py   Range checks, completeness, gap detection, QA reports
|
|-- visualization/      Publication-quality figures and interactive maps
    |-- maps.py         Static raster maps, interactive maps, comparisons
```

---

## Requirements

| Dependency | Purpose |
|---|---|
| numpy, scipy | Numerical computation |
| xarray, netCDF4 | Multidimensional labeled arrays |
| rasterio, rioxarray | Geospatial raster I/O |
| geopandas, shapely | Vector geometry operations |
| earthaccess | NASA Earthdata authentication and search |
| cdsapi | Copernicus Climate Data Store access |
| matplotlib, cartopy | Static map generation |
| folium | Interactive web maps |

Python 3.10 or higher is required.

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository and create a feature branch.
2. Write tests for any new functionality.
3. Ensure all tests pass: `pytest tests/`
4. Format code with `black` and check types with `mypy`.
5. Open a pull request with a clear description of the changes.

---

## Citation

If you use LISF Toolkit in published research, please cite:

```bibtex
@software{lisf_toolkit_2025,
  author       = {Marshall, Sebastian R.O.},
  title        = {{LISF Toolkit: NASA Land Information System Framework Data Toolkit}},
  year         = {2025},
  url          = {https://github.com/rushmarshall/LISF-Toolkit},
  note         = {Python library for satellite data acquisition, processing, and analysis}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Developed at <strong>Hydrosense Lab</strong>, University of Virginia
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:111111,30:333333,60:666666,100:999999&height=120&section=footer" width="100%" alt="" />
</p>
