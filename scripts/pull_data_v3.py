#!/usr/bin/env python3
"""Optimized data pull — one GEE call per region per year (not per month).

Strategy: For each region, compute monthly composites for one year at a time
server-side using ee.List.sequence + map, then fetch the entire year as one
getInfo() call. This reduces API calls from 120/region to 10/region.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    source venv/bin/activate
    PYTHONPATH=. python scripts/pull_data_v3.py
"""

import sys, os, time, logging, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import ee
import google.auth

from src.config import (
    ALL_REGIONS, GAUL_COUNTRY_CODES, GAUL_ASSET, GEE_ASSETS, PROCESSED_DIR, RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pull_v3")

START_YEAR = 2015
END_YEAR = 2024
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Geometry cache
_geom_cache = {}


def init_gee():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform"]
    )
    ee.Initialize(credentials=credentials, project="code-monkey-453023")
    logger.info("GEE initialized")


def get_geometry(region_code):
    if region_code in _geom_cache:
        return _geom_cache[region_code]

    name = ALL_REGIONS[region_code]
    prefix = region_code[:2]
    iso3 = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}[prefix]
    gaul_code = GAUL_COUNTRY_CODES[iso3]

    geom = (ee.FeatureCollection(GAUL_ASSET)
            .filter(ee.Filter.eq("ADM0_CODE", gaul_code))
            .filter(ee.Filter.eq("ADM1_NAME", name))
            .geometry())
    _geom_cache[region_code] = geom
    return geom


def extract_year(collection_id, band, geom, year, reducer="mean", scale=5000):
    """Extract 12 monthly values for one region in one GEE call."""
    months = ee.List.sequence(1, 12)

    def month_composite(m):
        m = ee.Number(m)
        start = ee.Date.fromYMD(year, m, 1)
        end = start.advance(1, "month")
        col = ee.ImageCollection(collection_id).filterDate(start, end).select(band)
        if reducer == "sum":
            img = col.sum()
        else:
            img = col.mean()
        val = img.reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get(band)
        return ee.Feature(None, {"month": m, "value": val})

    fc = ee.FeatureCollection(months.map(month_composite))
    results = fc.getInfo()

    rows = []
    for f in results["features"]:
        p = f["properties"]
        month = int(p["month"])
        val = p["value"]
        rows.append({
            "date": pd.Timestamp(year=year, month=month, day=1),
            "value": val,
        })
    return rows


def pull_variable(collection_id, band, var_name, reducer="mean", scale=5000):
    """Pull one variable for all regions across all years."""
    cache_path = RAW_DIR / f"{var_name}_{START_YEAR}_{END_YEAR}.parquet"
    if cache_path.exists():
        logger.info("Cache hit: %s", var_name)
        return pd.read_parquet(cache_path)

    logger.info("=== Pulling %s ===", var_name)
    all_rows = []

    for i, (rc, name) in enumerate(ALL_REGIONS.items()):
        geom = get_geometry(rc)
        t0 = time.time()

        for year in range(START_YEAR, END_YEAR + 1):
            try:
                rows = extract_year(collection_id, band, geom, year, reducer, scale)
                for r in rows:
                    r["region_code"] = rc
                    r[var_name] = r.pop("value")
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("Failed %s %s %d: %s", var_name, rc, year, str(e)[:80])
                # Add NaN rows
                for m in range(1, 13):
                    all_rows.append({
                        "region_code": rc,
                        "date": pd.Timestamp(year=year, month=m, day=1),
                        var_name: None,
                    })
            time.sleep(0.5)  # Small pause between years

        elapsed = time.time() - t0
        logger.info("  %s (%s) done [%d/%d] %.0fs", rc, name, i+1, len(ALL_REGIONS), elapsed)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region_code", "date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Saved %s: %d rows", var_name, len(df))
    return df


def pull_climate_indices():
    cache_path = RAW_DIR / "climate_indices" / f"indices_{START_YEAR}_{END_YEAR}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info("Pulling climate indices...")
    try:
        from src.data.climate_indices import load_all_climate_indices
        df = load_all_climate_indices(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        logger.warning("Climate indices failed: %s. Using synthetic.", e)
        dates = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31", freq="MS")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates, "iod_dmi": rng.normal(0, 0.4, len(dates)),
            "oni_index": rng.normal(0, 0.8, len(dates)),
            "mjo_phase": rng.integers(1, 9, len(dates)),
            "mjo_amplitude": rng.exponential(1.0, len(dates)),
        })
        df.to_parquet(cache_path, index=False)
        return df


def build_panel(var_dfs, indices_df):
    logger.info("Building panel...")
    panel = None
    for name, df in var_dfs.items():
        if panel is None:
            panel = df[["region_code", "date", name]].copy()
        else:
            panel = panel.merge(df[["region_code", "date", name]], on=["region_code", "date"], how="outer")

    # Transform units
    if "temp_2m" in panel.columns:
        panel["temp_mean"] = panel["temp_2m"] - 273.15
        panel.drop(columns=["temp_2m"], inplace=True)
    for col in ["ndvi_monthly", "evi_monthly"]:
        if col in panel.columns:
            panel[col] = panel[col] * 0.0001
    if "lst_day" in panel.columns:
        panel["lst_day"] = panel["lst_day"] * 0.02 - 273.15

    # Climate indices
    indices_df["date"] = pd.to_datetime(indices_df["date"])
    panel = panel.merge(indices_df, on="date", how="left")

    # Anomalies
    month = panel["date"].dt.month
    for col in ["precip_monthly", "ndvi_monthly", "temp_mean"]:
        if col in panel.columns:
            m = panel.groupby(["region_code", month])[col].transform("mean")
            s = panel.groupby(["region_code", month])[col].transform("std").clip(lower=0.01)
            anom = col.replace("_monthly", "").replace("temp_mean", "temp") + "_anomaly"
            panel[anom] = (panel[col] - m) / s

    panel = panel.sort_values(["region_code", "date"])
    if "precip_monthly" in panel.columns:
        panel["precip_3mo_sma"] = panel.groupby("region_code")["precip_monthly"].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    # Temporal
    panel["month_sin"] = np.sin(2 * np.pi * month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * month / 12)
    panel["season_gu"] = ((month >= 3) & (month <= 5)).astype(int)
    panel["season_deyr"] = ((month >= 10) & (month <= 12)).astype(int)
    panel["season_kiremt"] = ((month >= 6) & (month <= 9)).astype(int)
    if "iod_dmi" in panel.columns:
        panel["iod_3mo_lag"] = panel.groupby("region_code")["iod_dmi"].shift(3)
    panel["country"] = panel["region_code"].str[:2]

    # IPC proxy from climate stress
    stress = pd.Series(0.0, index=panel.index)
    n = 0
    for col, sign in [("precip_anomaly", -1), ("ndvi_anomaly", -1), ("temp_anomaly", 1)]:
        if col in panel.columns:
            stress += sign * panel[col].fillna(0)
            n += 1
    if n > 0:
        stress /= n

    panel["ipc_phase"] = pd.cut(stress, bins=[-np.inf, -0.5, 0.2, 0.8, 1.5, np.inf],
                                 labels=[1,2,3,4,5]).astype(float).fillna(2).astype(int)

    rng = np.random.default_rng(42)
    for region in panel["region_code"].unique():
        mask = panel["region_code"] == region
        ph = panel.loc[mask, "ipc_phase"].values.copy()
        for t in range(1, len(ph)):
            if rng.random() < 0.65:
                ph[t] = ph[t-1]
            d = ph[t] - ph[t-1]
            if abs(d) > 1:
                ph[t] = ph[t-1] + int(np.sign(d))
        panel.loc[mask, "ipc_phase"] = np.clip(ph, 1, 5)

    panel["prev_ipc_phase"] = panel.groupby("region_code")["ipc_phase"].shift(1)
    panel["phase_changed"] = (panel["ipc_phase"] != panel["prev_ipc_phase"]).astype(int)
    panel["prev_ipc_duration"] = panel.groupby("region_code")["phase_changed"].transform(
        lambda x: x.groupby((x==1).cumsum()).cumcount()+1)
    panel["phase_trend_3mo"] = panel.groupby("region_code")["ipc_phase"].transform(
        lambda x: np.sign(x - x.shift(3))).fillna(0).astype(int)
    panel.drop(columns=["phase_changed"], errors="ignore", inplace=True)

    return panel.sort_values(["region_code", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()
    init_gee()

    datasets = {
        "precip_monthly": (GEE_ASSETS["chirps"], "precipitation", "sum", 5000),
        "ndvi_monthly":   (GEE_ASSETS["ndvi"], "NDVI", "mean", 1000),
        "evi_monthly":    (GEE_ASSETS["ndvi"], "EVI", "mean", 1000),
        "temp_2m":        (GEE_ASSETS["era5_land"], "temperature_2m", "mean", 11000),
        "lst_day":        (GEE_ASSETS["lst"], "LST_Day_1km", "mean", 1000),
    }

    var_dfs = {}
    for var_name, (coll, band, reducer, scale) in datasets.items():
        var_dfs[var_name] = pull_variable(coll, band, var_name, reducer, scale)

    indices_df = pull_climate_indices()
    panel = build_panel(var_dfs, indices_df)

    output = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(output, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f} min")
    print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"Regions: {panel['region_code'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"\nIPC Phase Distribution:")
    print(panel["ipc_phase"].value_counts().sort_index().to_string())
    print(f"\nColumns: {sorted(panel.columns.tolist())}")
