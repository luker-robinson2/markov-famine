#!/usr/bin/env python3
"""Pull missing data sources (IPC, market prices, MJO) and rebuild panel.

Sources:
  - IPC phases:   FEWS NET Data Warehouse (fdw.fews.net/api/ipcphase/)
  - Market prices: FEWS NET Data Warehouse (fdw.fews.net/api/marketpricefacts/)
  - MJO RMM:      IRI Data Library (iridl.ldeo.columbia.edu)

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    .venv/bin/python scripts/pull_missing_data.py
"""

import sys, os, time, logging, io
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import requests

from src.config import ALL_REGIONS, PROCESSED_DIR, RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pull_missing")

START = "2015-01-01"
END = "2024-12-31"
TIMEOUT = 120


# =====================================================================
#  1. IPC Phase Data from FEWS NET Data Warehouse
# =====================================================================

def pull_ipc_fewsnet():
    """Pull IPC phase classifications from FEWS NET Data Warehouse."""
    cache_path = RAW_DIR / "ipc" / f"ipc_fewsnet_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: IPC FEWS NET")
        return pd.read_parquet(cache_path)

    logger.info("Pulling IPC data from FEWS NET Data Warehouse...")
    frames = []

    for country_code in ["KE", "ET", "SO"]:
        url = (
            f"https://fdw.fews.net/api/ipcphase/"
            f"?country={country_code}&format=csv&fields=simple"
        )
        logger.info("  Fetching IPC for %s ...", country_code)
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df["country_code_2"] = country_code
            frames.append(df)
            logger.info("    %s: %d rows", country_code, len(df))
        except Exception as e:
            logger.error("    Failed for %s: %s", country_code, e)

    if not frames:
        logger.error("No IPC data retrieved!")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Parse and normalize
    raw["reporting_date"] = pd.to_datetime(raw["reporting_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

    # Filter to Current Situation (CS)
    cs = raw[raw["scenario"] == "CS"].copy()
    cs = cs.dropna(subset=["value", "reporting_date"])

    # Build admin-1 name map
    region_name_to_code = {}
    for code, name in ALL_REGIONS.items():
        region_name_to_code[name.lower()] = code

    # Extract admin-1 from geographic_unit_full_name
    # Format: "Ward, District, Province, Country" or "Province, Country"
    def extract_admin1(full_name, country):
        prefix = str(country)[:2]
        parts = [p.strip() for p in str(full_name).split(",")]

        # Try each part from right to left (admin-1 is usually 2nd from right)
        for part in reversed(parts):
            p_lower = part.lower().strip()
            if p_lower in region_name_to_code:
                rc = region_name_to_code[p_lower]
                if rc.startswith(prefix):
                    return rc
            # Partial match
            for name_lower, rc in region_name_to_code.items():
                if not rc.startswith(prefix):
                    continue
                if name_lower in p_lower or p_lower in name_lower:
                    return rc
        return None

    cs["region_code"] = [
        extract_admin1(fn, cc)
        for fn, cc in zip(cs["geographic_unit_full_name"], cs["country_code_2"])
    ]
    cs = cs.dropna(subset=["region_code", "reporting_date", "value"])

    # Aggregate: take max phase per region per month
    cs["date"] = cs["reporting_date"].dt.to_period("M").dt.to_timestamp()
    cs = cs[(cs["date"] >= START) & (cs["date"] <= END)]

    result = (
        cs.groupby(["region_code", "date"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "ipc_phase"})
    )
    result["ipc_phase"] = result["ipc_phase"].astype(int).clip(1, 5)

    # Forward-fill to monthly panel
    date_range = pd.date_range(START, END, freq="MS")
    filled_parts = []
    for rc in result["region_code"].unique():
        rc_df = result[result["region_code"] == rc].set_index("date")
        rc_df = rc_df.reindex(date_range)
        rc_df["region_code"] = rc
        rc_df["ipc_phase"] = rc_df["ipc_phase"].ffill().bfill()
        rc_df = rc_df.reset_index().rename(columns={"index": "date"})
        filled_parts.append(rc_df)

    if filled_parts:
        result = pd.concat(filled_parts, ignore_index=True)
    result = result.dropna(subset=["ipc_phase"])
    result["ipc_phase"] = result["ipc_phase"].astype(int)

    result.to_parquet(cache_path, index=False)
    logger.info("IPC data: %d rows, %d regions", len(result), result["region_code"].nunique())
    return result


# =====================================================================
#  2. Market Prices from FEWS NET Data Warehouse
# =====================================================================

def pull_market_fewsnet():
    """Pull commodity prices from FEWS NET Data Warehouse."""
    cache_path = RAW_DIR / "markets" / f"markets_fewsnet_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: markets FEWS NET")
        return pd.read_parquet(cache_path)

    logger.info("Pulling market prices from FEWS NET Data Warehouse...")
    frames = []

    for country_code in ["KE", "ET", "SO"]:
        url = (
            f"https://fdw.fews.net/api/marketpricefacts.csv"
            f"?country_code={country_code}"
            f"&start_date={START}&end_date={END}"
            f"&fields=simple"
        )
        logger.info("  Fetching markets for %s ...", country_code)
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8-sig")
            df["country_code_2"] = country_code
            frames.append(df)
            logger.info("    %s: %d rows", country_code, len(df))
        except Exception as e:
            logger.error("    Failed for %s: %s", country_code, e)

    if not frames:
        logger.error("No market data retrieved!")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # Map admin_1 to region codes
    def match_admin1(admin1, country):
        admin1 = str(admin1).strip().lower()
        prefix = str(country)[:2]

        for code, name in ALL_REGIONS.items():
            if not code.startswith(prefix):
                continue
            if name.lower() == admin1 or name.lower() in admin1 or admin1 in name.lower():
                return code
        return None

    raw["region_code"] = [
        match_admin1(a, c)
        for a, c in zip(raw["admin_1"], raw["country_code_2"])
    ]
    raw["date"] = pd.to_datetime(raw["period_date"], errors="coerce")
    raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()
    raw["price"] = pd.to_numeric(raw.get("value", raw.get("common_currency_price")), errors="coerce")
    raw["commodity"] = raw["product"].str.strip().str.lower()

    raw = raw.dropna(subset=["region_code", "date", "price"])

    # Classify commodities
    livestock_kw = {"goat", "cattle", "camel", "sheep", "oxen", "ox", "cow", "bull", "heifer"}
    grain_kw = {"maize", "sorghum", "wheat", "rice", "teff", "millet", "bean", "cowpea"}

    def classify(commodity):
        c = commodity.lower()
        if any(kw in c for kw in livestock_kw):
            return "livestock"
        elif any(kw in c for kw in grain_kw):
            return "grain"
        return "other"

    raw["commodity_group"] = raw["commodity"].apply(classify)

    # Save full raw prices
    keep_cols = ["region_code", "date", "commodity", "commodity_group", "price",
                 "currency", "unit", "market", "admin_1"]
    for c in keep_cols:
        if c not in raw.columns:
            raw[c] = None
    raw = raw[keep_cols].copy()
    raw.to_parquet(cache_path, index=False)
    logger.info("Market data: %d rows, %d regions", len(raw), raw["region_code"].nunique())
    return raw


def compute_tot_and_anomalies(market_df):
    """Compute Terms of Trade and price anomalies from market data."""
    if market_df.empty:
        return pd.DataFrame(columns=["region_code", "date", "tot_livestock_grain", "tot_anomaly",
                                      "maize_price_anomaly"])

    # Terms of Trade: median livestock price / median grain price per region-month
    livestock = market_df[market_df["commodity_group"] == "livestock"]
    grain = market_df[market_df["commodity_group"] == "grain"]

    ls_med = livestock.groupby(["region_code", "date"])["price"].median().reset_index()
    ls_med.columns = ["region_code", "date", "livestock_price"]

    gr_med = grain.groupby(["region_code", "date"])["price"].median().reset_index()
    gr_med.columns = ["region_code", "date", "grain_price"]

    tot = ls_med.merge(gr_med, on=["region_code", "date"], how="inner")
    tot["tot_livestock_grain"] = tot["livestock_price"] / tot["grain_price"].replace(0, np.nan)

    # ToT anomaly (z-score relative to region mean)
    tot_mean = tot.groupby("region_code")["tot_livestock_grain"].transform("mean")
    tot_std = tot.groupby("region_code")["tot_livestock_grain"].transform("std").clip(lower=0.01)
    tot["tot_anomaly"] = (tot["tot_livestock_grain"] - tot_mean) / tot_std

    # Maize price anomaly
    maize = market_df[market_df["commodity"].str.contains("maize", case=False, na=False)]
    if not maize.empty:
        maize_monthly = maize.groupby(["region_code", "date"])["price"].median().reset_index()
        maize_monthly.columns = ["region_code", "date", "maize_price"]
        maize_monthly["month"] = maize_monthly["date"].dt.month
        m_mean = maize_monthly.groupby(["region_code", "month"])["maize_price"].transform("mean")
        m_std = maize_monthly.groupby(["region_code", "month"])["maize_price"].transform("std").clip(lower=0.01)
        maize_monthly["maize_price_anomaly"] = (maize_monthly["maize_price"] - m_mean) / m_std
        tot = tot.merge(
            maize_monthly[["region_code", "date", "maize_price_anomaly"]],
            on=["region_code", "date"], how="left"
        )

    result = tot[["region_code", "date", "tot_livestock_grain", "tot_anomaly"]].copy()
    if "maize_price_anomaly" in tot.columns:
        result["maize_price_anomaly"] = tot["maize_price_anomaly"]

    logger.info("ToT computed: %d rows, %d regions", len(result), result["region_code"].nunique())
    return result


# =====================================================================
#  3. MJO RMM from IRI Data Library
# =====================================================================

def pull_mjo_iri():
    """Pull MJO RMM index from IRI Data Library (Columbia)."""
    cache_path = RAW_DIR / "climate_indices" / f"mjo_iri_{START}_{END}.parquet"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        logger.info("Cache hit: MJO IRI")
        return pd.read_parquet(cache_path)

    logger.info("Pulling MJO RMM from IRI Data Library...")
    base = "https://iridl.ldeo.columbia.edu/SOURCES/.BoM/.MJO/.RMM"

    components = {}
    for var in ["RMM1", "RMM2", "phase", "amplitude"]:
        url = f"{base}/.{var}/data.tsv"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            vals = resp.text.strip().split("\t")
            components[var] = [float(v) if v.strip() else np.nan for v in vals]
            logger.info("  %s: %d daily values", var, len(vals))
        except Exception as e:
            logger.error("  Failed to fetch %s: %s", var, e)
            return pd.DataFrame()

    # All series start from 1974-06-01, one value per day
    n_vals = min(len(v) for v in components.values())
    start_date = datetime(1974, 6, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_vals)]

    daily = pd.DataFrame({
        "date": dates,
        "mjo_rmm1": components["RMM1"][:n_vals],
        "mjo_rmm2": components["RMM2"][:n_vals],
        "mjo_phase": components["phase"][:n_vals],
        "mjo_amplitude": components["amplitude"][:n_vals],
    })

    # Filter to our date range and aggregate to monthly
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[(daily["date"] >= START) & (daily["date"] <= END)]
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    monthly = daily.groupby("month_start", as_index=False).agg(
        mjo_rmm1=("mjo_rmm1", "mean"),
        mjo_rmm2=("mjo_rmm2", "mean"),
        mjo_phase=("mjo_phase", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
        mjo_amplitude=("mjo_amplitude", "mean"),
    ).rename(columns={"month_start": "date"})

    monthly["mjo_phase"] = monthly["mjo_phase"].astype(int)

    monthly.to_parquet(cache_path, index=False)
    logger.info("MJO data: %d monthly values", len(monthly))
    return monthly


# =====================================================================
#  4. Rebuild panel
# =====================================================================

def rebuild_panel(ipc_df, market_features, mjo_df):
    """Merge new data sources into the existing panel."""
    panel_path = PROCESSED_DIR / "panel.parquet"
    if not panel_path.exists():
        logger.error("No existing panel.parquet found!")
        return

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    logger.info("Loaded existing panel: %d rows x %d cols", panel.shape[0], panel.shape[1])

    # Drop old IPC proxy columns to replace with real data
    old_ipc_cols = ["ipc_phase", "prev_ipc_phase", "prev_ipc_duration", "phase_trend_3mo"]
    panel.drop(columns=[c for c in old_ipc_cols if c in panel.columns], inplace=True)

    # Drop old market columns if any
    old_market_cols = ["tot_livestock_grain", "tot_anomaly", "maize_price_anomaly"]
    panel.drop(columns=[c for c in old_market_cols if c in panel.columns], inplace=True)

    # Drop old MJO columns if any
    old_mjo_cols = ["mjo_phase", "mjo_amplitude", "mjo_rmm1", "mjo_rmm2"]
    panel.drop(columns=[c for c in old_mjo_cols if c in panel.columns], inplace=True)

    # Merge real IPC data
    if not ipc_df.empty:
        ipc_df["date"] = pd.to_datetime(ipc_df["date"])
        ipc_merge = ipc_df[["region_code", "date", "ipc_phase"]].drop_duplicates(
            subset=["region_code", "date"], keep="last"
        )
        panel = panel.merge(ipc_merge, on=["region_code", "date"], how="left")
        panel["ipc_phase"] = panel.groupby("region_code")["ipc_phase"].ffill().bfill()
        panel["ipc_phase"] = panel["ipc_phase"].fillna(2).astype(int)
        logger.info("Merged real IPC data: %d non-null", panel["ipc_phase"].notna().sum())
    else:
        panel["ipc_phase"] = 2

    # Merge market features (ToT, anomalies)
    if not market_features.empty:
        market_features["date"] = pd.to_datetime(market_features["date"])
        panel = panel.merge(market_features, on=["region_code", "date"], how="left")
        for col in ["tot_livestock_grain", "tot_anomaly", "maize_price_anomaly"]:
            if col in panel.columns:
                panel[col] = panel.groupby("region_code")[col].ffill()
        logger.info("Merged market features")

    # Merge MJO
    if not mjo_df.empty:
        mjo_df["date"] = pd.to_datetime(mjo_df["date"])
        panel = panel.merge(mjo_df, on="date", how="left")
        logger.info("Merged MJO data")

    # Recompute lagged IPC features
    panel = panel.sort_values(["region_code", "date"])
    panel["prev_ipc_phase"] = panel.groupby("region_code")["ipc_phase"].shift(1)
    panel["phase_changed"] = (panel["ipc_phase"] != panel["prev_ipc_phase"]).astype(int)
    panel["prev_ipc_duration"] = panel.groupby("region_code")["phase_changed"].transform(
        lambda x: x.groupby((x == 1).cumsum()).cumcount() + 1
    )
    panel["phase_trend_3mo"] = panel.groupby("region_code")["ipc_phase"].transform(
        lambda x: np.sign(x - x.shift(3))
    ).fillna(0).astype(int)
    panel.drop(columns=["phase_changed"], errors="ignore", inplace=True)

    panel = panel.sort_values(["region_code", "date"]).reset_index(drop=True)

    panel.to_parquet(panel_path, index=False)
    logger.info("Saved updated panel: %d rows x %d cols", panel.shape[0], panel.shape[1])
    return panel


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    # Step 1: IPC
    logger.info("=" * 60)
    logger.info("STEP 1: Pulling IPC data from FEWS NET Data Warehouse")
    logger.info("=" * 60)
    ipc_df = pull_ipc_fewsnet()

    # Step 2: Market prices
    logger.info("=" * 60)
    logger.info("STEP 2: Pulling market prices from FEWS NET Data Warehouse")
    logger.info("=" * 60)
    market_df = pull_market_fewsnet()
    market_features = compute_tot_and_anomalies(market_df)

    # Step 3: MJO
    logger.info("=" * 60)
    logger.info("STEP 3: Pulling MJO RMM from IRI Data Library")
    logger.info("=" * 60)
    mjo_df = pull_mjo_iri()

    # Step 4: Rebuild panel
    logger.info("=" * 60)
    logger.info("STEP 4: Rebuilding panel")
    logger.info("=" * 60)
    panel = rebuild_panel(ipc_df, market_features, mjo_df)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"COMPLETE in {elapsed:.1f} min")
    print(f"{'=' * 60}")

    if panel is not None:
        print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} cols")
        print(f"Regions: {panel['region_code'].nunique()}")
        print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
        print(f"\nIPC Phase Distribution:")
        print(panel["ipc_phase"].value_counts().sort_index().to_string())
        print(f"\nNon-null coverage:")
        for col in sorted(panel.columns):
            pct = panel[col].notna().mean() * 100
            print(f"  {col:30s} {pct:5.1f}%")
