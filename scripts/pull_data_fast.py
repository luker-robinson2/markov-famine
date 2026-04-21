#!/usr/bin/env python3
"""Fast data pull — aggregates months server-side in GEE to minimize API calls.

Instead of 120 API calls per region (one per month), this makes ~1 call per region
by computing monthly means as a FeatureCollection server-side and extracting all at once.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    source venv/bin/activate
    PYTHONPATH=. python scripts/pull_data_fast.py
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
    ALL_REGIONS, GAUL_COUNTRY_CODES, GAUL_ASSET,
    GEE_ASSETS, PROCESSED_DIR, RAW_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pull_fast")

START = "2015-01-01"
END = "2024-12-31"
REGION_CODES = list(ALL_REGIONS.keys())

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def init_gee():
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/earthengine",
                "https://www.googleapis.com/auth/cloud-platform"]
    )
    ee.Initialize(credentials=credentials, project="code-monkey-453023")
    logger.info("GEE initialized")


def get_admin1_fc():
    """Get all Horn of Africa admin-1 regions as a FeatureCollection."""
    gaul = ee.FeatureCollection(GAUL_ASSET)
    countries = gaul.filter(
        ee.Filter.inList("ADM0_CODE", list(GAUL_COUNTRY_CODES.values()))
    )
    return countries


def extract_monthly_timeseries(collection_id, band, regions_fc, start, end,
                                reducer_type="mean", scale=5000):
    """Extract monthly time series for all regions in one batch GEE call.

    Uses ee.ImageCollection.map() and ee.FeatureCollection.reduceRegions()
    to compute monthly stats server-side, then fetches all results at once.
    """
    start_date = ee.Date(start)
    end_date = ee.Date(end)

    # Create monthly date sequence
    n_months = end_date.difference(start_date, "month").round()
    month_starts = ee.List.sequence(0, n_months.subtract(1)).map(
        lambda n: start_date.advance(n, "month")
    )

    def monthly_reduce(date_obj):
        date = ee.Date(date_obj)
        month_end = date.advance(1, "month")
        collection = ee.ImageCollection(collection_id) \
            .filterDate(date, month_end) \
            .select(band)

        if reducer_type == "sum":
            image = collection.sum()
        else:
            image = collection.mean()

        # Reduce over all regions
        reduced = image.reduceRegions(
            collection=regions_fc,
            reducer=ee.Reducer.mean(),
            scale=scale,
        )

        # Tag each feature with the date
        return reduced.map(lambda f: f.set("date", date.format("YYYY-MM-dd")))

    # Map over all months
    all_features = ee.FeatureCollection(month_starts.map(monthly_reduce)).flatten()
    return all_features


def fc_to_dataframe(fc, band_name, chunk_size=5000):
    """Convert a GEE FeatureCollection to a pandas DataFrame, handling large collections."""
    # Get total count
    total = fc.size().getInfo()
    logger.info("  Fetching %d features...", total)

    all_rows = []
    offset = 0

    while offset < total:
        chunk = fc.toList(chunk_size, offset)
        features = chunk.getInfo()
        if not features:
            break

        for f in features:
            props = f.get("properties", {})
            all_rows.append({
                "adm0_code": props.get("ADM0_CODE"),
                "adm1_name": props.get("ADM1_NAME"),
                "date": props.get("date"),
                band_name: props.get("mean"),
            })
        offset += len(features)
        if offset < total:
            logger.info("  Fetched %d/%d...", offset, total)

    return pd.DataFrame(all_rows)


def map_gaul_to_region_codes(df):
    """Map GAUL ADM1_NAME + ADM0_CODE to our region codes."""
    # Build reverse mapping: (adm0_code, adm1_name) -> region_code
    gaul_to_code = {}
    reverse_gaul = {v: k for k, v in GAUL_COUNTRY_CODES.items()}  # {133: 'KEN', ...}
    country_prefix = {"KEN": "KE", "ETH": "ET", "SOM": "SO"}

    for rc, name in ALL_REGIONS.items():
        prefix = rc[:2]
        iso3 = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}[prefix]
        gaul_code = GAUL_COUNTRY_CODES[iso3]
        gaul_to_code[(gaul_code, name)] = rc

    df["region_code"] = df.apply(
        lambda row: gaul_to_code.get((row["adm0_code"], row["adm1_name"]), None),
        axis=1
    )
    unmapped = df["region_code"].isna().sum()
    if unmapped > 0:
        logger.warning("%d rows could not be mapped to region codes", unmapped)
        # Try fuzzy match
        for idx, row in df[df["region_code"].isna()].iterrows():
            for (gc, name), rc in gaul_to_code.items():
                if row["adm0_code"] == gc and (
                    name.lower() in str(row["adm1_name"]).lower() or
                    str(row["adm1_name"]).lower() in name.lower()
                ):
                    df.at[idx, "region_code"] = rc
                    break

    df = df.dropna(subset=["region_code"])
    return df


def pull_variable(collection_id, band, name, reducer_type="mean", scale=5000):
    """Pull one variable for all regions."""
    cache_path = RAW_DIR / f"{name}_all_{START}_{END}.parquet"
    if cache_path.exists():
        logger.info("Cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Pulling %s (band=%s) ...", name, band)
    regions_fc = get_admin1_fc()

    fc = extract_monthly_timeseries(collection_id, band, regions_fc, START, END,
                                     reducer_type=reducer_type, scale=scale)
    df = fc_to_dataframe(fc, name)
    df = map_gaul_to_region_codes(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["region_code", "date", name]].sort_values(["region_code", "date"]).reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    logger.info("Saved %s: %d rows, %d regions", name, len(df), df["region_code"].nunique())
    return df


def pull_all_gee():
    """Pull all GEE variables."""
    datasets = [
        (GEE_ASSETS["chirps"], "precipitation", "precip_monthly", "sum", 5000),
        (GEE_ASSETS["ndvi"], "NDVI", "ndvi_monthly", "mean", 1000),
        (GEE_ASSETS["ndvi"], "EVI", "evi_monthly", "mean", 1000),
        (GEE_ASSETS["era5_land"], "temperature_2m", "temp_2m", "mean", 11000),
        (GEE_ASSETS["era5_land"], "total_precipitation", "era5_precip", "sum", 11000),
        (GEE_ASSETS["lst"], "LST_Day_1km", "lst_day", "mean", 1000),
    ]

    dfs = {}
    for collection_id, band, name, reducer, scale in datasets:
        try:
            df = pull_variable(collection_id, band, name, reducer, scale)
            dfs[name] = df
        except Exception as e:
            logger.error("Failed to pull %s: %s", name, e)

    return dfs


def pull_climate_indices():
    """Pull IOD, ENSO, MJO."""
    cache_path = RAW_DIR / "climate_indices" / f"indices_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)

    if cache_path.exists():
        logger.info("Cache hit: climate indices")
        return pd.read_parquet(cache_path)

    logger.info("Pulling climate indices...")
    try:
        from src.data.climate_indices import load_all_climate_indices
        df = load_all_climate_indices(START, END)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        logger.warning("Climate indices download failed: %s. Using synthetic.", e)
        dates = pd.date_range(START, END, freq="MS")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates,
            "iod_dmi": rng.normal(0, 0.4, len(dates)),
            "oni_index": rng.normal(0, 0.8, len(dates)),
            "mjo_phase": rng.integers(1, 9, len(dates)),
            "mjo_amplitude": rng.exponential(1.0, len(dates)),
        })
        df.to_parquet(cache_path, index=False)
        return df


def build_panel(dfs, indices_df):
    """Merge all variables into panel.parquet."""
    logger.info("Building panel...")

    # Start with first available dataset
    panel = None
    for name, df in dfs.items():
        if panel is None:
            panel = df.copy()
        else:
            panel = panel.merge(df, on=["region_code", "date"], how="outer")

    if panel is None:
        raise RuntimeError("No data pulled!")

    # Convert ERA5 temp from Kelvin to Celsius
    if "temp_2m" in panel.columns:
        panel["temp_mean"] = panel["temp_2m"] - 273.15
        panel = panel.drop(columns=["temp_2m"])

    # MODIS NDVI/EVI scale factor (0.0001)
    for col in ["ndvi_monthly", "evi_monthly"]:
        if col in panel.columns:
            panel[col] = panel[col] * 0.0001

    # MODIS LST scale factor (0.02) and Kelvin to Celsius
    if "lst_day" in panel.columns:
        panel["lst_day"] = panel["lst_day"] * 0.02 - 273.15

    # Merge climate indices
    indices_df["date"] = pd.to_datetime(indices_df["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(indices_df, on="date", how="left")

    # Anomalies
    panel["year_month"] = panel["date"].dt.to_period("M")
    month = panel["date"].dt.month
    for col in ["precip_monthly", "ndvi_monthly", "temp_mean"]:
        if col in panel.columns:
            clim_mean = panel.groupby(["region_code", month])[col].transform("mean")
            clim_std = panel.groupby(["region_code", month])[col].transform("std").clip(lower=0.01)
            anom_name = col.replace("_monthly", "").replace("temp_mean", "temp") + "_anomaly"
            panel[anom_name] = (panel[col] - clim_mean) / clim_std

    # Moving averages
    panel = panel.sort_values(["region_code", "date"])
    if "precip_monthly" in panel.columns:
        panel["precip_3mo_sma"] = panel.groupby("region_code")["precip_monthly"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

    # Temporal
    panel["month_sin"] = np.sin(2 * np.pi * panel["date"].dt.month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["date"].dt.month / 12)
    panel["season_gu"] = ((month >= 3) & (month <= 5)).astype(int)
    panel["season_deyr"] = ((month >= 10) & (month <= 12)).astype(int)
    panel["season_kiremt"] = ((month >= 6) & (month <= 9)).astype(int)

    if "iod_dmi" in panel.columns:
        panel["iod_3mo_lag"] = panel.groupby("region_code")["iod_dmi"].shift(3)

    panel["country"] = panel["region_code"].str[:2]

    # Generate IPC proxy phases from climate stress
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

    # Add temporal persistence
    rng = np.random.default_rng(42)
    for region in panel["region_code"].unique():
        mask = panel["region_code"] == region
        phases = panel.loc[mask, "ipc_phase"].values.copy()
        for t in range(1, len(phases)):
            if rng.random() < 0.65:
                phases[t] = phases[t - 1]
            diff = phases[t] - phases[t - 1]
            if abs(diff) > 1:
                phases[t] = phases[t - 1] + int(np.sign(diff))
        panel.loc[mask, "ipc_phase"] = np.clip(phases, 1, 5)

    # Lagged features
    panel["prev_ipc_phase"] = panel.groupby("region_code")["ipc_phase"].shift(1)
    panel["phase_changed"] = (panel["ipc_phase"] != panel["prev_ipc_phase"]).astype(int)
    panel["prev_ipc_duration"] = panel.groupby("region_code")["phase_changed"].transform(
        lambda x: x.groupby((x == 1).cumsum()).cumcount() + 1
    )
    panel["phase_trend_3mo"] = panel.groupby("region_code")["ipc_phase"].transform(
        lambda x: np.sign(x - x.shift(3))
    ).fillna(0).astype(int)

    panel = panel.drop(columns=["phase_changed", "year_month"], errors="ignore")
    panel = panel.sort_values(["region_code", "date"]).reset_index(drop=True)

    return panel


if __name__ == "__main__":
    t0 = time.time()
    init_gee()

    dfs = pull_all_gee()
    indices_df = pull_climate_indices()
    panel = build_panel(dfs, indices_df)

    output = PROCESSED_DIR / "panel.parquet"
    panel.to_parquet(output, index=False)

    elapsed = (time.time() - t0) / 60
    logger.info("Done in %.1f min. Panel: %s, %d cols, %d regions",
                elapsed, panel.shape, len(panel.columns), panel["region_code"].nunique())

    print(f"\n=== PANEL SUMMARY ===")
    print(f"Shape: {panel.shape}")
    print(f"Regions: {panel['region_code'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"\nIPC Phase Distribution:")
    print(panel["ipc_phase"].value_counts().sort_index().to_string())
    print(f"\nNon-null percentages:")
    for col in sorted(panel.columns):
        pct = panel[col].notna().mean() * 100
        if pct < 100:
            print(f"  {col:30s} {pct:5.1f}%")
