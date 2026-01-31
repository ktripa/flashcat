# ⚡ FlashCAT: Flash Drought Characterization and Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-development-orange)]()

**FlashCAT** is a comprehensive Python package designed for the identification, characterization, and analysis of flash droughts. It bridges the gap between raw hydroclimatic data and actionable drought insights by providing a unified framework for calculating state-of-the-art flash drought indices, classical drought metrics, and potential evapotranspiration (PET).

---

## 📖 Software Design and Architecture

FlashCAT is built on widely used scientific computing libraries, **NumPy**, **Pandas**, and **SciPy**. The software architecture is organized into three core modules:

1.  **⚡ Flash Drought Estimation Module**
2.  **📉 Classical Drought Indices Module**
3.  **☀️ Potential Evapotranspiration (PET) Estimation Module**

Users supply input variables—such as temperature, precipitation, and soil moisture—in tabular or vectorized formats (e.g., NumPy arrays or Pandas DataFrames), which are subsequently processed through the selected computational modules to generate standardized drought metrics.

FlashCAT further integrates **built-in identification criteria** to objectively detect flash drought onset, peak intensity, and recovery phases. These algorithms automate the transformation from continuous index time series to discrete, event-based flash drought characterization, enabling consistent and reproducible analyses across diverse hydroclimatic settings.

---

## 🚀 Features

* **Comprehensive Index Suite:** Includes over 10 drought indices, ranging from atmospheric demand (EDDI) to soil moisture response (SMPD, RZSM).
* **Automated Detection:** Built-in algorithms to identify flash drought **Onset**, **Termination**, and **Duration** based on peer-reviewed definitions (e.g., Pendergrass et al., Christian et al., Yuan et al.).
* **Flexible PET Calculation:** Switch between 4 different PET methods depending on your available data.
* **Standardized Workflow:** Consistent API for all indices—input time series, get standardized Z-scores or event lists back.
* **High Performance:** Vectorized calculations using NumPy/Pandas for efficient processing of large datasets (grids/satellite data).

---

## 📊 Supported Indicators

### 1. Flash Drought Indices
| Index | Full Name | Reference | Driver |
| :--- | :--- | :--- | :--- |
| **EDDI** | Evaporative Demand Drought Index | Hobbins et al. (2016) | Atmospheric Demand |
| **ESI** | Evaporative Stress Index | Anderson et al. (2011) | ET / PET Ratio |
| **SESR** | Standardized Evaporative Stress Ratio | Christian et al. (2019) | Rapid Change in Stress |
| **SMVI** | Soil Moisture Volatility Index | Osman et al. (2021) | Volatility & Deficit |
| **SMPD** | Soil Moisture Percentile Drop | Ford & Labosier (2017) | Rapid Soil Depletion |
| **FDRZSM** | Flash Drought based on RZSM | Yuan et al. (2019) | Root Zone Soil Moisture |

### 2. Classical Drought Indices
| Index | Full Name | Application |
| :--- | :--- | :--- |
| **SPI** | Standardized Precipitation Index | Meteorological Drought |
| **SPEI** | Standardized Precipitation Evapotranspiration Index | Ag/Met Drought |
| **scPDSI** | Self-Calibrated Palmer Drought Severity Index | Long-term Soil Moisture |

### 3. Multivariate Indices
* **MFDI:** Multivariate Flash Drought Index
* **FDII:** Flash Drought Intensity Index

---

## ☀️ PET Estimation Methods
FlashCAT allows users to calculate Potential Evapotranspiration (PET) using four standard methods, depending on data availability:

1.  **Thornthwaite (1948):** Requires Mean Temperature (Monthly).
2.  **Hargreaves (1985):** Requires Tmin, Tmax, Tmean (Daily). *Recommended for Flash Drought when radiation is missing.*
3.  **Priestley-Taylor (1972):** Requires Net Radiation & Temperature.
4.  **Penman-Monteith (FAO-56):** Requires Temperature, Humidity, Wind, Radiation. *The Gold Standard.*

---

## 📂 Project Structure

```text
FlashCAT/
│
├── src/
│   └── flashcat/
│       ├── __init__.py        # Package entry
│       ├── pet.py             # PET Estimation Module
│       ├── utils.py           # Shared utilities (Probability, Pentads)
│       └── indices/
│           ├── __init__.py
│           ├── eddi.py        # Evaporative Demand Drought Index
│           ├── esi.py         # Evaporative Stress Index
│           ├── sesr.py        # Standardized Evaporative Stress Ratio
│           ├── smvi.py        # Soil Moisture Volatility Index
│           ├── smpd.py        # Soil Moisture Percentile Drop
│           ├── rzsm.py        # Root Zone Soil Moisture Index
│           ├── spi.py         # Standard Precipitation Index
│           └── ...
│
├── examples/                  # Jupyter Notebook Tutorials
│   ├── 01_calculate_pet.ipynb
│   ├── 02_eddi_tutorial.ipynb
│   └── ...
│
├── data/                      # Sample datasets
├── tests/                     # Unit tests
├── pyproject.toml             # Configuration & Dependencies
└── README.md                  # Documentation