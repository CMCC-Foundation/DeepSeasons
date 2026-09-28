# DeepSeasons

Code and example notebooks for

> A. Navarra and G. G. Navarra, *DeepSeasons: a case study of a Deep Learning scale-selecting approach to Seasonal Forecasts*, submitted to *Artificial Intelligence for the Earth Systems* (AIES-D-25-0077).

DeepSeasons forecasts monthly and seasonal climate anomalies with a Transformer (Informer) encoder–decoder. The input fields are projected onto EOFs, and the model is trained on ERA5 monthly means (1940–2005), validated on 2006–2015 and tested on 2016–2022. Forecasts are verified against persistence, linear baselines and the CMCC operational seasonal system (C3S).

The repository contains everything needed to rerun the paper's experiments **except the input data**. ERA5 is freely available from the Copernicus Climate Data Store; see [Input data](#input-data) for how to download it.

## Repository layout

```
DeepSeasons/
├── notebooks/                 example experiments (with the stored outputs used for the paper)
│   ├── SST-Y620d.ipynb        Global SST, monthly forecasts        (inputs SST, U850)
│   ├── SST-Y621d-MONT.ipynb   Europe T2M, monthly forecasts        (inputs T850, T2M)
│   ├── SST-Y622d-MONT.ipynb   North America T2M, monthly forecasts (inputs tropical SST, T2M)
│   └── SST-Y621s-SEAS.ipynb   Europe T2M, seasonal-mean forecasts  (inputs T850, T2M)
├── lib/                       local libraries imported by the notebooks
│   ├── Zapata/                zapata (data I/O, EOFs, mapping) + klus (kernel/Koopman tools)
│   ├── AIModels/              data pipeline, Informer model, training, validation, plotting
│   ├── ig_attribution.py      Integrated Gradients attribution by lead time
│   └── dspaths.py             single place where data and working paths are configured
├── scripts/
│   ├── download_era5.py       fetch the ERA5 input fields from the CDS
│   ├── linear_baselines.py    ridge / AR / persistence / climatology baselines
│   ├── make_baselines_table.py, boxplot_baselines.py, plot_ig_combined.py   paper tables and figures
├── models/                    trained weights and verification data of the four cases of the paper
│   ├── global_sst/            Y620d (with the IG attributions)
│   ├── europe_t2m/            Y621d (with the IG attributions)
│   ├── north_america_t2m/     Y622d
│   └── seasonal_europe/       Y621s
├── data/                      (empty, git-ignored) default location for the ERA5 archive
├── environment.yml / requirements.txt
└── LICENSE, NOTICE, CITATION.cff
```

| Notebook | Case | Target (score field) | Predictors | Domain | Forecast |
|---|---|---|---|---|---|
| `SST-Y620d` | Y620d | SST | SST, U850 | Global | monthly, leads 1–12 |
| `SST-Y621d-MONT` | Y621d | T2M | T850, T2M | Europe | monthly |
| `SST-Y622d-MONT` | Y622d | T2M | SST (tropics), T2M | North America | monthly |
| `SST-Y621s-SEAS` | Y621s | T2M | T850, T2M | Europe | seasonal means |

All cases use 8 attention heads, a Student-t likelihood and MaxMin scaling of the EOF coefficients. The other settings differ by case (Table S1 of the Supplementary Material):

| Case | EOFs | Input window | Lags | Layers | `d_model` | Discount γ |
|---|---|---|---|---|---|---|
| Y620d Global SST | SST 15, U850 25 | 48 months | 1–4 | 2 | 256 | 0.9 |
| Y621d Europe T2M | T850 5, T2M 5 (75 %) | 18 | 1–4 | 1 | 256 | 0.9 |
| Y622d North America T2M | SST 29 (75 %), T2M 5 | 18 | 1–2 | 2 | 512 | 0.9 |
| Y621s Europe seasonal | T850 14, T2M 26 (95 %) | 18 | 1–4 | 1 | 256 | 1.0 |

## Trained models and verification data

The trained weights and the verification results of the four cases of the paper are included in `models/` (about 80 MB in total), one folder per case with a README describing the files and how to load them:

| Folder | Case | Files |
|---|---|---|
| `models/global_sst/` | Y620d Global SST | weights, verification (DeepSeasons, persistence, GCM), AR(4) and ridge benchmarks, mean scores, IG attributions |
| `models/europe_t2m/` | Y621d Europe T2M | weights, verification, AR(4) and ridge benchmarks, mean scores, IG attributions |
| `models/north_america_t2m/` | Y622d North America T2M | weights, verification, AR(2) and ridge benchmarks, mean scores |
| `models/seasonal_europe/` | Y621s Europe seasonal T2M | weights, verification, AR(4) and ridge benchmarks, mean scores |

They allow the box plots (Figs 2, 6, 9 and 14), Table 2 and the IG figures (Figs 12–13) to be reproduced without retraining, and the models to be applied to new ERA5 data. The verification pickles are those used for the paper. The weights have been checked by rerunning the four notebooks in restart mode: the scores are reproduced to within 10⁻⁶ for the T2M cases (fixed random seed) and to within the ensemble sampling noise (about 0.01 in the mean ACC) for the global SST case.

## Installation

Python ≥ 3.10 is required (the code uses `match` statements).

```bash
git clone https://github.com/CMCC-Foundation/DeepSeasons.git
cd DeepSeasons
conda env create -f environment.yml
conda activate deepseasons
```

Or with pip: `pip install -r requirements.txt` (on some platforms cartopy is easier to install through conda).

You don't need to install the bundled libraries. Each notebook adds `lib/` and `lib/Zapata/` to `sys.path` through `lib/dspaths.py`.

## Input data

The ERA5 data are **not** included in this repository. The notebooks read monthly-mean ERA5 fields on the native 0.25° global grid, January 1940 – December 2022 (996 months, about 2 GB per field):

| Field | CDS dataset | CDS variable | Level | Used by |
|---|---|---|---|---|
| SST | `reanalysis-era5-single-levels-monthly-means` | `sea_surface_temperature` | surface | Y620d, Y622d |
| T2M | `reanalysis-era5-single-levels-monthly-means` | `2m_temperature` | surface | Y621d, Y621s, Y622d |
| U850 | `reanalysis-era5-pressure-levels-monthly-means` | `u_component_of_wind` | 850 hPa | Y620d |
| T850 | `reanalysis-era5-pressure-levels-monthly-means` | `temperature` | 850 hPa | Y621d, Y621s |

Product type: `monthly_averaged_reanalysis`.

### 1. Get CDS access

1. Register (free) at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu).
2. Accept the licence (*Licence to use Copernicus Products*) on the download pages of [ERA5 monthly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) and [ERA5 monthly pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means).
3. Install the API client (`pip install "cdsapi>=0.7.2"`). Then create `~/.cdsapirc` with your personal access token (shown on your CDS profile page):
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-personal-access-token>
   ```

### 2. Download

```bash
export DEEPSEASONS_DATA=/path/to/ERA5_DATA      # or leave unset to use ./data/ERA5_DATA
python scripts/download_era5.py                 # SST, T2M, U850, T850
python scripts/download_era5.py --fields SST U850   # only what Y620d needs
```

The script writes the files in the layout expected by `zapata.data.read_era5`:

```
$DEEPSEASONS_DATA/
├── SST/SST_SST_V5.nc     (variable: sst)
├── T2M/t2m_t2m_V5.nc     (variable: t2m)
├── U/U_850_V5.nc         (variable: u)
└── T/T_850_V5.nc         (variable: t)
```

Each file has dimensions `time × latitude × longitude` (996 × 721 × 1440), with latitude running from 90 to −90 and longitude from 0 to 359.75. If you already have ERA5 monthly means from another source, rename them to this layout. On first use, Zapata writes a cache of processed fields to `$DEEPSEASONS_DATA/DATA_CACHE/`, so the directory must be writable.

### 3. (Optional) Seasonal forecasts for the dynamical-model comparison

With `verify_dyn = True`, the notebooks compare DeepSeasons with the CMCC seasonal prediction system (C3S systems 3 and 35). These hindcasts/forecasts come from the CDS dataset [`seasonal-postprocessed-single-levels`](https://cds.climate.copernicus.eu/datasets/seasonal-postprocessed-single-levels) (originating centre `cmcc`, systems `3` and `35`, product type `monthly_mean`, lead-time months 1–6). The variables are `sea_surface_temperature_anomaly` (`ssta`) and `2m_temperature_anomaly` (`t2a`). Save one file per system and lead:

```
$DEEPSEASONS_SEASONAL/SEASONAL_SST/SST_cmcc_<system>_<lead>.nc
$DEEPSEASONS_SEASONAL/SEASONAL_T2M/2t_cmcc_<system>_<lead>.nc
```

On the first run the notebook builds a verification cache (`DYN_*.nc`) in the working directory. To skip this comparison, set `verify_dyn = False` in the configuration cell.

## Running the notebooks

```bash
export DEEPSEASONS_DATA=/path/to/ERA5_DATA
export DEEPSEASONS_SEASONAL=/path/to/SEASONAL     # optional
export DEEPSEASONS_WORKDIR=/path/to/work          # optional, default ./work
cd notebooks
jupyter lab
```

The first code cell calls `dspaths.setup()`, which prints the paths in use and changes into the working directory. All trained models (`MOD_*.pt`), result pickles (`*_corr_data`), CSV scores and figures are written there. You can also edit the defaults in `lib/dspaths.py` instead of setting environment variables.

The main switches are in the "Configure case" cell:

- `choose_device`: `'MPS'` (Apple silicon, as used for the paper), `'CPU'`; other values fall back to CPU.
- `verify_dyn`: compare against the CMCC seasonal system (needs the optional data above).
- `integrated_gradients` (Y620d, Y621d): compute or reload the IG attribution.
- `params['RESTART']`: reload a previously trained model instead of training.

The random seed is fixed (`torch.manual_seed(17553984)`). Results can still differ slightly between hardware back-ends.

### Baselines and paper figures

After running the notebooks (so the `*_corr_data` pickles exist in the working directory):

```bash
python scripts/linear_baselines.py --case all
python scripts/make_baselines_table.py
python scripts/boxplot_baselines.py --case all
python scripts/plot_ig_combined.py
```

## Licences

This repository bundles two CMCC libraries with different licences:

- `lib/AIModels` — GNU GPL v3 (<https://github.com/CMCC-Foundation/AIModels>)
- `lib/Zapata` — Apache 2.0 (<https://github.com/CMCC-Foundation/Zapata>)
- `lib/AIModels/Informer` — adapted from Hugging Face `transformers`, Apache 2.0

As a combined work the repository is distributed under **GPL v3** (see `LICENSE` and `NOTICE`). Each bundled component keeps its own licence file.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Contact

Antonio Navarra — antonio.navarra@cmcc.it — Fondazione CMCC, Bologna, Italy
