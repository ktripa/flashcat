# FlashCAT 🌩️⚡

[![PyPI version](https://badge.fury.io/py/flashcat.svg)](https://badge.fury.io/py/flashcat)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/username/flashcat/workflows/CI/badge.svg)](https://github.com/username/flashcat/actions)
[![Documentation Status](https://readthedocs.org/projects/flashcat/badge/?version=latest)](https://flashcat.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Flash-drought Computation & Analytics Toolkit**

A comprehensive Python package for detecting, analyzing, and visualizing flash drought events using state-of-the-art meteorological and hydrological indicators.

> 🔥 **Flash droughts** develop rapidly (2-7 weeks) and can devastate agriculture, ecosystems, and water resources. FlashCAT provides the tools scientists and practitioners need to monitor these extreme events.

![FlashCAT Demo](https://raw.githubusercontent.com/username/flashcat/main/docs/assets/flashcat_demo.gif)

## ✨ Key Features

**🔍 Multi-Indicator Detection**
- 10+ flash drought indicators (EDDI, ESI, SMVI, RSMI, SESR, dSPEI/dt, FDII, FDSI, VPD surge)
- Traditional drought indices for comparison (SPI, SPEI, PDSI)
- Robust rate-of-change calculations with gap handling

**📊 Advanced Analytics**
- Event onset, termination, and tracking algorithms
- Spatial-temporal characteristics (duration, severity, spread)
- Skill assessment and indicator intercomparison
- CF-compliant metadata and units handling via `pint-xarray`

**🌍 Multi-Dataset Support**
- ERA5, GLEAM, SMAP, USDM data loaders
- Flexible resampling (daily → weekly/monthly/pentad)
- Land/vegetation masks and growing season filters

**📈 Rich Visualization**
- Cartographic maps with `cartopy` and `geopandas` integration
- Event timeline ribbons and onset histograms

**⚡ Performance & Reliability**
- Vectorized operations with `xarray` and `dask`
- Comprehensive test suite with property-based testing
- Benchmark suite for reproducible performance tracking

## 🚀 Quick Start

### Installation

```bash
pip install flashcat
```

For development installation:
```bash
git clone https://github.com/username/flashcat.git
cd flashcat
pip install -e ".[dev,docs]"
```

### Basic Usage

```python
import flashcat as fc
import xarray as xr

# Load your meteorological data
ds = xr.open_dataset("your_climate_data.nc")

# Calculate flash drought indicators
eddi = fc.indicators.eddi(ds.precip, ds.pet, window=4)
esi = fc.indicators.esi(ds.sm, window=4)

# Detect flash drought events
events = fc.detect.onset(
    indicators=[eddi, esi],
    thresholds=[-1.3, -1.3],
    min_duration=2  # weeks
)

# Analyze event characteristics
characteristics = fc.detect.characteristics(events, ds)

# Create visualization
fig = fc.viz.maps.flash_drought_map(
    events.sel(time="2012-07-15"), 
    title="Flash Drought Events - July 2012"
)
```

### Command Line Interface

FlashCAT also provides a convenient CLI:

```bash
# Calculate EDDI for a NetCDF file
flashcat eddi input.nc --output eddi_results.nc --window 4

# Detect flash drought events
flashcat detect eddi_results.nc esi_results.nc --thresholds -1.3 -1.3

# Generate summary plots
flashcat viz events.nc --type timeline --region CONUS
```

## 📖 Documentation & Examples

- **[Documentation](https://flashcat.readthedocs.io)**: Complete API reference and user guide
- **[Example Notebooks](https://github.com/ktripa/flashcat/tree/main/examples)**:
  - [Flash Drought Onset Detection (CONUS)](https://mybinder.org/v2/gh/username/flashcat/main?filepath=examples%2Fflash_onset_CONUS.ipynb)
  - [Flash Drought Climatology Analysis](https://mybinder.org/v2/gh/username/flashcat/main?filepath=examples%2Ffd_climatology_CONUS.ipynb)
- **[Interactive Binder](https://mybinder.org/v2/gh/ktripa/flashcat/main)**: Try FlashCAT in your browser

## 🔬 Supported Indicators

| Indicator | Description | Reference |
|-----------|-------------|-----------|
| **EDDI** | Evaporative Demand Drought Index | Hobbins et al. (2016) |
| **ESI** | Evaporative Stress Index | Anderson et al. (2011) |
| **SMVI** | Soil Moisture Volatility Index | Bergman et al. (2017) |
| **RSMI** | Rapid Soil Moisture Intensification | Liu et al. (2020) |
| **SMPD** | Soil Moisture Percentile Drop | Xu et al. (2019) |
| **SESR** | Standardized Evapotranspiration Stress Ratio | Kim et al. (2021) |
| **dSPEI/dt** | SPEI Rate of Change | Vicente-Serrano et al. (2010) |
| **FDII** | Flash Drought Intensity Index (Composite) | Zhang et al. (2022) |
| **FDSI** | Flash Drought Stress Index | Christian et al. (2019) |
| **VPD Surge** | Vapor Pressure Deficit Surge | Yuan et al. (2019) |

## 🌡️ Data Compatibility

FlashCAT works seamlessly with popular climate datasets:

- **ERA5** (Copernicus Climate Data Store)
- **GLEAM** (Global Land Evaporation Amsterdam Model)
- **SMAP** (Soil Moisture Active Passive)
- **USDM** (US Drought Monitor)
- **NLDAS-2** (North American Land Data Assimilation System)
- Custom NetCDF files following CF conventions

## 📊 Benchmarks & Validation

FlashCAT includes comprehensive benchmarks comparing indicators against:
- Historical flash drought events (2012 US drought, 2018 European heatwave)
- USDM rapid intensification categories
- Agricultural impact records
- Peer-reviewed detection algorithms

Run benchmarks locally:
```bash
python benchmarks/indicator_intercomparison/run_benchmarks.py
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Development environment setup
- Code style guidelines  
- Testing requirements
- Pull request process

### Development Setup

```bash
git clone https://github.com/username/flashcat.git
cd flashcat
conda env create -f environment.yml
conda activate flashcat-dev
pip install -e ".[dev]"
pytest tests/  # Run test suite
```

## 📄 Citation

If you use FlashCAT in your research, please cite:

```bibtex
@software{flashcat2024,
  title = {FlashCAT: Flash-drought Computation \& Analytics Toolkit},
  author = {Kumar Puran Tripathy and Ashok Kumar Mishra},
  year = {2025},
  url = {https://github.com/ktripa/flashcat},
  doi = {will be given later}
}
```

Also consider citing the original papers for specific indicators you use (see documentation for full references).

## 📜 License

FlashCAT is released under the [BSD 3-Clause License](LICENSE).

## 🙏 Acknowledgments

- NOAA/NIDIS for flash drought research coordination
- ECMWF for ERA5 reanalysis data
- NASA for SMAP soil moisture data
- The broader climate science community for foundational algorithms

## 📞 Support & Community

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/username/flashcat/issues)
- **Discussions**: Join our [GitHub Discussions](https://github.com/username/flashcat/discussions)
- **Email**: tripathypuranbdk@gmail.com
---

<div align="center">

**⚡ Fast drought detection for a changing climate ⚡**

[Install FlashCAT](https://pypi.org/project/flashcat/) • [Documentation](https://flashcat.readthedocs.io) • [Examples](https://github.com/username/flashcat/tree/main/examples)

</div>









