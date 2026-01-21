FlashCAT/
│
├── .gitignore
├── README.md                  <-- We will add images here later
├── pyproject.toml             <-- Config file
│
├── assets/                    <-- NEW: Folder for README images/logos
│   └── logo.png               <-- (Optional) We can put a cool logo here later
│
├── data/                      <-- NEW: Sample data for tutorials (NetCDF/CSV)
│   └── sample_precip.nc
│
├── examples/                  <-- Jupyter Notebooks for tutorials
│   ├── 01_calculate_pet.ipynb
│   ├── 02_eddi_tutorial.ipynb
│   ├── 03_sesr_tutorial.ipynb
│   └── 04_full_flash_drought_workflow.ipynb
│
├── src/
│   └── flashcat/
│       ├── __init__.py        <-- Main package entry
│       ├── pet.py             <-- 4 PET methods (Thornthwaite, Penman, etc.)
│       ├── utils.py           <-- Shared tools (date handling, checking NetCDF)
│       │
│       └── indices/           <-- FLATTENED: One file per index
│           ├── __init__.py    <-- Exposes all indices
│           ├── eddi.py        <-- Contains: calc_eddi() + identify_deficits()
│           ├── esi.py
│           ├── sesr.py
│           ├── smvi.py
│           ├── rzsm.py
│           ├── spi.py
│           ├── spei.py
│           ├── scpdsi.py      <-- The hardest one (Palmers), we will tackle this
│           ├── fdii.py        <-- Flash Drought Intensity Index
│           └── mfdi.py        <-- Multivariate Flash Drought Index
│
└── tests/                     <-- Unit tests
    ├── test_pet.py
    └── test_eddi.py