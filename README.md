# Evapotranspiration drivers at US-ARM (2017): masked autoencoder + SHAP

`autoencoder.ipynb` predicts evapotranspiration (latent heat flux LE) at the AmeriFlux site
US-ARM (Lamont, Oklahoma) in 2017 and explains the predictions. It runs these steps:

0. **Preprocessing**: gap-filling of the raw half-hourly data with MDS (Reichstein et al., 2005)
1. **Pretraining**: masked autoencoder on the drivers (soil moisture, radiation, temperature, …)
2. **Transfer learning**: frozen encoder + small MLP that predicts LE
3. **Latent counterfactuals** (daily only): which drivers the model must change to raise or lower ET
4. **SHAP values**: contribution of each driver to the prediction

Steps 1, 2 and 4 run on both daily data and daytime half-hourly data; step 3 runs on daily data only.

All the code is in `autoencoder.py`. The notebook calls its functions.

## The data file is not in this repository

You need the raw half-hourly FLUXNET file of US-ARM for 2017, named exactly
**`ARM_HH_2017.csv`**. It is the only input.

It is a comma-separated file with one row per half-hour (17,520 rows) and FLUXNET column names
(`TIMESTAMP_START`, `TIMESTAMP_END`, `LE`, `H`, `NETRAD`, `SW_IN_F`, `TA_F`, `VPD_F`, `SWC_F_MDS_*`,
`TS_F_MDS_*`, `RECO`, `NIGHT`, …). Missing values are coded as `-9999`.

## Where to put the CSV

The first cell of the notebook downloads this repository and unpacks it into a folder called
**`evotranspiration-main/`**. The CSV must be in that folder, next to `autoencoder.py`:

```
evotranspiration-main/
├── autoencoder.py
├── autoencoder.ipynb
└── ARM_HH_2017.csv      <- put the file here
```

### On Google Colab

1. Open `autoencoder.ipynb` in Colab:
   https://colab.research.google.com/github/mich3leRusso/evotranspiration/blob/main/autoencoder.ipynb
2. Run **only the first cell**. It creates `/content/evotranspiration-main/`.
3. Open the **Files** panel (folder icon on the left), go into `evotranspiration-main`
   and upload `ARM_HH_2017.csv` there (right-click on the folder → Upload).
4. Run all the other cells (Runtime → Run after).

Colab deletes uploaded files when the runtime restarts. After a restart, upload the CSV again.

### On your own computer

1. Download `autoencoder.ipynb` and open it in Jupyter or VS Code.
2. Run the first cell. It creates `evotranspiration-main/` next to the notebook.
3. Copy `ARM_HH_2017.csv` into `evotranspiration-main/`.
4. Run all the other cells.

If you run the first cell again later, the CSV stays where it is. The download only
overwrites the files that are in this repository.

If the CSV is missing, step 0 stops with
`FileNotFoundError: ... evotranspiration-main/ARM_HH_2017.csv`.

## Requirements

Python ≥ 3.10. The first cell installs `torch`, `shap`, `xgboost` and `scikit-learn` if they are
missing. `numpy`, `pandas` and `matplotlib` must already be installed (they are on Colab).

A full run takes about 3 minutes on a CPU.

## Outputs

These are written inside `evotranspiration-main/`:

- `ARM_HH_2017_filled.csv`: gap-filled data
- `figures/autoencoder/`: all figures
- `results/autoencoder/`: tables and trained model weights
