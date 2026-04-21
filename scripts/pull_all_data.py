#!/usr/bin/env python3
"""Comprehensive data pull — consolidates existing caches, pulls all missing data.

Pulls:
  1. GEE satellite data (CHIRPS, MODIS NDVI/EVI, MODIS LST, ERA5-Land, SMAP)
  2. Climate indices (IOD DMI, ENSO ONI, MJO RMM)
  3. IPC food security phase data (API + HFID fallback)
  4. FEWS NET market prices + Terms of Trade

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    .venv/bin/python scripts/pull_all_data.py
"""

import sys, os, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import ee
import google.auth

from src.config import (
    ALL_REGIONS, GAUL_COUNTRY_CODES, GAUL_ASSET, GEE_ASSETS,
    PROCESSED_DIR, RAW_DIR,
    ANALYSIS_START, ANALYSIS_END,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pull_all")

# Date range driven by src/config.py (ANALYSIS_START / ANALYSIS_END).
# Extension to 2011 captures the 2011 Somalia famine (Phase 5) events.
START = ANALYSIS_START  # "2011-01-01" after revision
END = ANALYSIS_END      # "2024-12-31"
START_YEAR = int(START[:4])
END_YEAR = int(END[:4])

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_geom_cache = {}


# =====================================================================
#  GEE init + helpers
# =====================================================================

def init_gee():
    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/earthengine",
            "https://www.googleapis.com/auth/cloud-platform",
        ]
    )
    ee.Initialize(credentials=credentials, project="code-monkey-453023")
    logger.info("GEE initialized (project=code-monkey-453023)")


def get_geometry(region_code):
    if region_code in _geom_cache:
        return _geom_cache[region_code]
    name = ALL_REGIONS[region_code]
    prefix = region_code[:2]
    iso3 = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}[prefix]
    gaul_code = GAUL_COUNTRY_CODES[iso3]
    geom = (
        ee.FeatureCollection(GAUL_ASSET)
        .filter(ee.Filter.eq("ADM0_CODE", gaul_code))
        .filter(ee.Filter.eq("ADM1_NAME", name))
        .geometry()
    )
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
        img = col.sum() if reducer == "sum" else col.mean()
        val = img.reduceRegion(ee.Reducer.mean(), geom, scale, maxPixels=1e9).get(band)
        return ee.Feature(None, {"month": m, "value": val})

    fc = ee.FeatureCollection(months.map(month_composite))
    results = fc.getInfo()
    rows = []
    for f in results["features"]:
        p = f["properties"]
        rows.append({
            "date": pd.Timestamp(year=year, month=int(p["month"]), day=1),
            "value": p["value"],
        })
    return rows


# =====================================================================
#  Step 1: Consolidate existing per-region caches
# =====================================================================

def consolidate_per_region_caches():
    """Merge per-region parquet files into single consolidated files."""
    var_dirs = {
        "precip_monthly": ("precip_monthly", "precipitation"),
        "ndvi_monthly": ("ndvi_monthly", "NDVI"),
        "evi_monthly": ("evi_monthly", "EVI"),
        "lst_day": ("lst_day", "LST_Day_1km"),
        "temp_2m": ("temp_2m", "temperature_2m"),
    }

    for var_name, (subdir, _band) in var_dirs.items():
        cache_path = RAW_DIR / f"{var_name}_{START_YEAR}_{END_YEAR}.parquet"
        if cache_path.exists():
            continue

        region_dir = RAW_DIR / subdir
        if not region_dir.exists():
            continue

        parquet_files = sorted(region_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        logger.info("Consolidating %d per-region files for %s ...", len(parquet_files), var_name)
        frames = []
        for pf in parquet_files:
            rc = pf.stem  # e.g. "KE001"
            if rc not in ALL_REGIONS:
                continue
            try:
                df = pd.read_parquet(pf)
                # Normalize column names — the per-region caches may have
                # different column naming conventions
                df["region_code"] = rc
                if "date" not in df.columns:
                    # Look for a date-like column
                    for col in df.columns:
                        if "date" in col.lower() or "time" in col.lower():
                            df = df.rename(columns={col: "date"})
                            break
                # Find the value column
                val_col = None
                for col in df.columns:
                    if col not in ("region_code", "date") and df[col].dtype in ("float64", "float32", "int64"):
                        val_col = col
                        break
                if val_col and val_col != var_name:
                    df = df.rename(columns={val_col: var_name})
                if var_name in df.columns:
                    frames.append(df[["region_code", "date", var_name]])
            except Exception as e:
                logger.warning("Failed to read %s: %s", pf, e)

        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"])
            merged = merged.sort_values(["region_code", "date"]).reset_index(drop=True)
            merged.to_parquet(cache_path, index=False)
            logger.info("Consolidated %s: %d rows, %d regions",
                        var_name, len(merged), merged["region_code"].nunique())


# =====================================================================
#  Step 2: Pull GEE variables (skip cached)
# =====================================================================

def pull_gee_variable(collection_id, band, var_name, reducer="mean", scale=5000):
    """Pull one variable for all regions across all years."""
    cache_path = RAW_DIR / f"{var_name}_{START_YEAR}_{END_YEAR}.parquet"
    if cache_path.exists():
        logger.info("Cache hit: %s (%s)", var_name, cache_path.name)
        return pd.read_parquet(cache_path)

    logger.info("=== Pulling %s from GEE ===", var_name)
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
                logger.warning("Failed %s %s %d: %s", var_name, rc, year, str(e)[:120])
                for m in range(1, 13):
                    all_rows.append({
                        "region_code": rc,
                        "date": pd.Timestamp(year=year, month=m, day=1),
                        var_name: None,
                    })
            time.sleep(0.3)

        elapsed = time.time() - t0
        logger.info("  %s (%s) [%d/%d] %.0fs", rc, name, i + 1, len(ALL_REGIONS), elapsed)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region_code", "date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Saved %s: %d rows", var_name, len(df))
    return df


def pull_era5_soil_moisture():
    """Pull soil moisture from ERA5-Land MONTHLY_AGGR (pre-aggregated monthly).

    Uses ECMWF/ERA5_LAND/MONTHLY_AGGR which has only 120 images for 10 years
    instead of 87,600 hourly images. Much faster.
    """
    return pull_gee_variable(
        "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "volumetric_soil_water_layer_1",
        "soil_moisture_era5",
        reducer="mean",
        scale=11000,
    )


def pull_all_gee():
    """Pull all GEE satellite datasets."""
    datasets = {
        "precip_monthly": (GEE_ASSETS["chirps"], "precipitation", "sum", 5000),
        "ndvi_monthly":   (GEE_ASSETS["ndvi"], "NDVI", "mean", 1000),
        "evi_monthly":    (GEE_ASSETS["ndvi"], "EVI", "mean", 1000),
        "temp_2m":        (GEE_ASSETS["era5_land"], "temperature_2m", "mean", 11000),
        "lst_day":        (GEE_ASSETS["lst"], "LST_Day_1km", "mean", 1000),
    }

    dfs = {}
    for var_name, (coll, band, reducer, scale) in datasets.items():
        try:
            dfs[var_name] = pull_gee_variable(coll, band, var_name, reducer, scale)
        except Exception as e:
            logger.error("FAILED to pull %s: %s", var_name, e)

    # ERA5-Land soil moisture (same collection as temp, reliable)
    try:
        dfs["soil_moisture_era5"] = pull_era5_soil_moisture()
    except Exception as e:
        logger.error("FAILED to pull ERA5 soil moisture: %s", e)

    return dfs


# =====================================================================
#  Step 3: Climate indices
# =====================================================================

def pull_climate_indices():
    cache_path = RAW_DIR / "climate_indices" / f"indices_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: climate indices")
        return pd.read_parquet(cache_path)

    logger.info("Pulling climate indices (IOD, ENSO, MJO)...")
    try:
        from src.data.climate_indices import load_all_climate_indices
        df = load_all_climate_indices(START, END)
        df.to_parquet(cache_path, index=False)
        logger.info("Climate indices: %d rows, columns=%s", len(df), list(df.columns))
        return df
    except Exception as e:
        logger.error("Climate indices failed: %s", e)
        raise


# =====================================================================
#  Step 4: IPC data
# =====================================================================

def pull_ipc_data():
    cache_path = RAW_DIR / "ipc" / f"ipc_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: IPC data")
        return pd.read_parquet(cache_path)

    logger.info("Pulling IPC food security phase data...")
    try:
        from src.data.ipc_loader import load_ipc_data
        df = load_ipc_data(start=START, end=END)
        if not df.empty:
            df.to_parquet(cache_path, index=False)
            logger.info("IPC data: %d rows, %d regions", len(df), df["region_code"].nunique())
        else:
            logger.warning("IPC data empty — API may be unavailable")
        return df
    except Exception as e:
        logger.error("IPC data pull failed: %s", e)
        return pd.DataFrame(columns=["region_code", "date", "ipc_phase", "population_in_phase"])


# =====================================================================
#  Step 5: Market data
# =====================================================================

def pull_market_data():
    cache_path = RAW_DIR / "markets" / f"markets_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: market data")
        return pd.read_parquet(cache_path)

    logger.info("Pulling FEWS NET market price data...")
    try:
        from src.data.market_loader import load_market_prices, compute_terms_of_trade
        prices = load_market_prices(start=START, end=END)
        if not prices.empty:
            prices.to_parquet(cache_path, index=False)
            logger.info("Market prices: %d rows", len(prices))

            tot = compute_terms_of_trade(prices)
            tot_path = RAW_DIR / "markets" / f"tot_{START}_{END}.parquet"
            tot.to_parquet(tot_path, index=False)
            logger.info("Terms of Trade: %d rows", len(tot))
            return prices
        else:
            logger.warning("Market data empty — FEWS NET may be unavailable")
            return prices
    except Exception as e:
        logger.error("Market data pull failed: %s", e)
        return pd.DataFrame(columns=["region_code", "date", "commodity", "market", "price", "currency", "unit"])


# =====================================================================
#  Step 6: Build final panel
# =====================================================================

def build_panel(gee_dfs, indices_df, ipc_df, market_df):
    logger.info("Building comprehensive panel...")

    # Merge GEE data
    panel = None
    for name, df in gee_dfs.items():
        if panel is None:
            panel = df[["region_code", "date", name]].copy()
        else:
            panel = panel.merge(df[["region_code", "date", name]],
                                on=["region_code", "date"], how="outer")

    if panel is None:
        raise RuntimeError("No GEE data available!")

    panel["date"] = pd.to_datetime(panel["date"])

    # Unit conversions
    if "temp_2m" in panel.columns:
        panel["temp_mean"] = panel["temp_2m"] - 273.15
        panel.drop(columns=["temp_2m"], inplace=True)
    for col in ["ndvi_monthly", "evi_monthly"]:
        if col in panel.columns:
            panel[col] = panel[col] * 0.0001
    if "lst_day" in panel.columns:
        panel["lst_day"] = panel["lst_day"] * 0.02 - 273.15

    # Rename soil moisture for clarity
    if "soil_moisture_era5" in panel.columns:
        panel.rename(columns={"soil_moisture_era5": "soil_moisture"}, inplace=True)
    elif "sm_surface" in panel.columns:
        panel.rename(columns={"sm_surface": "soil_moisture"}, inplace=True)

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

    if "soil_moisture" in panel.columns:
        sm_mean = panel.groupby(["region_code", month])["soil_moisture"].transform("mean")
        sm_std = panel.groupby(["region_code", month])["soil_moisture"].transform("std").clip(lower=0.001)
        panel["soil_moisture_anomaly"] = (panel["soil_moisture"] - sm_mean) / sm_std

    # Moving averages
    panel = panel.sort_values(["region_code", "date"])
    if "precip_monthly" in panel.columns:
        panel["precip_3mo_sma"] = panel.groupby("region_code")["precip_monthly"].transform(
            lambda x: x.rolling(3, min_periods=1).mean())

    # Temporal features
    panel["month_sin"] = np.sin(2 * np.pi * month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * month / 12)
    panel["season_gu"] = ((month >= 3) & (month <= 5)).astype(int)
    panel["season_deyr"] = ((month >= 10) & (month <= 12)).astype(int)
    panel["season_kiremt"] = ((month >= 6) & (month <= 9)).astype(int)
    if "iod_dmi" in panel.columns:
        panel["iod_3mo_lag"] = panel.groupby("region_code")["iod_dmi"].shift(3)
    panel["country"] = panel["region_code"].str[:2]

    # IPC phases — use real data if available, else climate proxy
    if not ipc_df.empty and "ipc_phase" in ipc_df.columns:
        logger.info("Using real IPC phase data (%d rows)", len(ipc_df))
        ipc_df["date"] = pd.to_datetime(ipc_df["date"])
        ipc_merge = ipc_df[["region_code", "date", "ipc_phase"]].drop_duplicates(
            subset=["region_code", "date"], keep="last"
        )
        panel = panel.merge(ipc_merge, on=["region_code", "date"], how="left")
        # Forward-fill gaps in IPC (analyses happen 2-3x/year)
        panel["ipc_phase"] = panel.groupby("region_code")["ipc_phase"].ffill()
        # Backfill the initial gap
        panel["ipc_phase"] = panel.groupby("region_code")["ipc_phase"].bfill()
        # Any remaining NaN -> default to phase 2
        panel["ipc_phase"] = panel["ipc_phase"].fillna(2).astype(int)
    else:
        logger.warning("No real IPC data — generating proxy from climate stress")
        stress = pd.Series(0.0, index=panel.index)
        n = 0
        for col, sign in [("precip_anomaly", -1), ("ndvi_anomaly", -1), ("temp_anomaly", 1)]:
            if col in panel.columns:
                stress += sign * panel[col].fillna(0)
                n += 1
        if n > 0:
            stress /= n
        panel["ipc_phase"] = pd.cut(
            stress, bins=[-np.inf, -0.5, 0.2, 0.8, 1.5, np.inf], labels=[1, 2, 3, 4, 5]
        ).astype(float).fillna(2).astype(int)

        rng = np.random.default_rng(42)
        for region in panel["region_code"].unique():
            mask = panel["region_code"] == region
            ph = panel.loc[mask, "ipc_phase"].values.copy()
            for t in range(1, len(ph)):
                if rng.random() < 0.65:
                    ph[t] = ph[t - 1]
                d = ph[t] - ph[t - 1]
                if abs(d) > 1:
                    ph[t] = ph[t - 1] + int(np.sign(d))
            panel.loc[mask, "ipc_phase"] = np.clip(ph, 1, 5)

    # Market / Terms of Trade
    if not market_df.empty:
        try:
            from src.data.market_loader import compute_terms_of_trade, compute_price_anomalies
            tot = compute_terms_of_trade(market_df)
            if not tot.empty:
                tot["date"] = pd.to_datetime(tot["date"])
                panel = panel.merge(
                    tot[["region_code", "date", "tot_livestock_grain"]],
                    on=["region_code", "date"], how="left"
                )
                # Compute ToT anomaly
                if "tot_livestock_grain" in panel.columns:
                    tot_mean = panel.groupby("region_code")["tot_livestock_grain"].transform("mean")
                    tot_std = panel.groupby("region_code")["tot_livestock_grain"].transform("std").clip(lower=0.01)
                    panel["tot_anomaly"] = (panel["tot_livestock_grain"] - tot_mean) / tot_std
                logger.info("Merged Terms of Trade into panel")
        except Exception as e:
            logger.warning("Failed to merge market data: %s", e)

    # Lagged state features
    panel["prev_ipc_phase"] = panel.groupby("region_code")["ipc_phase"].shift(1)
    panel["phase_changed"] = (panel["ipc_phase"] != panel["prev_ipc_phase"]).astype(int)
    panel["prev_ipc_duration"] = panel.groupby("region_code")["phase_changed"].transform(
        lambda x: x.groupby((x == 1).cumsum()).cumcount() + 1)
    panel["phase_trend_3mo"] = panel.groupby("region_code")["ipc_phase"].transform(
        lambda x: np.sign(x - x.shift(3))).fillna(0).astype(int)
    panel.drop(columns=["phase_changed"], errors="ignore", inplace=True)

    panel = panel.sort_values(["region_code", "date"]).reset_index(drop=True)
    return panel


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    # Step 0: Init GEE
    init_gee()

    # Step 1: Consolidate existing per-region caches
    logger.info("=" * 60)
    logger.info("STEP 1: Consolidating existing cached data")
    logger.info("=" * 60)
    consolidate_per_region_caches()

    # Step 2: Pull GEE data (skips cached variables)
    logger.info("=" * 60)
    logger.info("STEP 2: Pulling GEE satellite data")
    logger.info("=" * 60)
    gee_dfs = pull_all_gee()

    # Step 3: Climate indices
    logger.info("=" * 60)
    logger.info("STEP 3: Pulling climate teleconnection indices")
    logger.info("=" * 60)
    indices_df = pull_climate_indices()

    # Step 4: IPC data
    logger.info("=" * 60)
    logger.info("STEP 4: Pulling IPC food security data")
    logger.info("=" * 60)
    ipc_df = pull_ipc_data()

    # Step 5: Market data
    logger.info("=" * 60)
    logger.info("STEP 5: Pulling FEWS NET market prices")
    logger.info("=" * 60)
    market_df = pull_market_data()

    # Step 6: Build panel
    logger.info("=" * 60)
    logger.info("STEP 6: Building final panel")
    logger.info("=" * 60)
    panel = build_panel(gee_dfs, indices_df, ipc_df, market_df)

    output = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(output, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f} min")
    print(f"{'=' * 60}")
    print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"Regions: {panel['region_code'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"\nIPC Phase Distribution:")
    print(panel["ipc_phase"].value_counts().sort_index().to_string())
    print(f"\nNon-null coverage:")
    for col in sorted(panel.columns):
        pct = panel[col].notna().mean() * 100
        print(f"  {col:30s} {pct:5.1f}%")
    print(f"\nSaved to: {output}")
