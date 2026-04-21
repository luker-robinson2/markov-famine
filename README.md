# Predicting Food Security Transitions in the Horn of Africa

**STAT 5000 — Statistical Methods and Applications** | University of Colorado Boulder | April 2026

Non-Homogeneous Markov Chain framework with regularized XGBoost classifiers for predicting IPC food security phase transitions across 37 admin-1 regions in Kenya, Ethiopia, and Somalia (2015–2024).

## Key Results

| Model | Test R² | Overfit Gap | Transitions Detected |
|-------|---------|-------------|---------------------|
| Persistence baseline | 0.921 | 0% | 0% |
| **PhasePredictor** | **0.865** | **<1%** | 14% |
| **DeltaPredictor** | 0.845 | 5.2% | **52%** |

The PhasePredictor has **zero overfitting**. The DeltaPredictor detects 52% of real phase transitions (67% of worsening events).

## Paper

See `paper/paper.tex` (10 pages) — compiled PDF at `paper/paper.pdf`.

## Data Sources

Satellite data from Google Earth Engine (CHIRPS, MODIS, ERA5-Land), IPC phases and market prices from FEWS NET Data Warehouse API, climate indices from NOAA/IRI.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login --scopes="https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform"
PYTHONPATH=. python scripts/pull_all_data.py
PYTHONPATH=. python scripts/pull_missing_data.py
```

## Architecture

- **Markov Chain Core**: Hand-coded NHMC where transition matrix P_t varies with
  climate, agronomic, and market covariates
- **Ensemble ML**: Per-state classifiers (XGBoost, LightGBM, CatBoost, RF) with
  stacking meta-learner parameterize transition probabilities
- **Feature Engineering**: 40+ features including SPEI, GDD, CWSI, IOD/ENSO
  teleconnections, Terms of Trade, XGBoost-derived SHAP/leaf features
- **Calibration**: Conformal prediction (CQR) for guaranteed coverage intervals

## Data Sources

| Source | Variables | Resolution |
|--------|-----------|------------|
| CHIRPS | Precipitation | 0.05deg, daily |
| MODIS | NDVI, EVI, LST | 250m-1km, 16-day |
| SMAP | Soil moisture | 9km, daily |
| ERA5-Land | Temp, humidity, wind, radiation | 11km, hourly |
| IPC/HFID | Food security phases | Admin-1, monthly |
| FEWS NET | Market prices, livelihood zones | Market/zone level |
| NOAA/BOM | IOD DMI, ENSO ONI, MJO RMM | Global, monthly |
| ACLED | Conflict events | Point, daily |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Authenticate with Google Earth Engine
python -c "import ee; ee.Authenticate()"
```

## Project Structure

```
src/data/          - Data acquisition (GEE, IPC, climate indices, markets)
src/engineering/   - Spatial/temporal alignment, agronomic indices, features
src/models/        - NHMC, ensemble classifiers, calibration
src/metrics/       - RPSS, Brier, asymmetric cost, lead-time analysis
src/viz/           - Choropleths, SHAP plots, forecast fans
notebooks/         - EDA, theory, training, results
paper/             - LaTeX research paper
```
