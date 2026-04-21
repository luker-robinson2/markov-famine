"""FEWS NET market and price data loader.

Downloads or loads FEWS NET commodity price data for Horn-of-Africa
markets, then derives:

- **Terms of Trade (ToT)**: livestock price / grain price -- the core
  purchasing-power indicator for pastoral livelihoods.
- **Price anomalies**: monthly z-scores relative to long-term seasonal
  means -- flag abnormal market stress.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Sequence

import pandas as pd
import requests

from src.config import (
    ALL_REGIONS,
    ANALYSIS_END,
    ANALYSIS_START,
    COUNTRY_CODES,
    FEWSNET_PRICE_URL,
    RAW_DIR,
)
from src.data.cache import cache_exists, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_TIMEOUT = 90  # seconds

# Canonical commodity groupings for ToT computation
LIVESTOCK_COMMODITIES = {"cattle", "goat", "goats", "camel", "sheep"}
GRAIN_COMMODITIES = {"maize", "sorghum", "wheat", "rice"}


# =====================================================================
#  Raw data loading
# =====================================================================

def _fetch_fewsnet_prices(country_iso3: str) -> Optional[pd.DataFrame]:
    """Download price data from the FEWS NET data portal.

    Parameters
    ----------
    country_iso3 : str
        ISO-3 country code (``"KEN"``, ``"ETH"``, ``"SOM"``).

    Returns
    -------
    pd.DataFrame or None
        Raw price table or ``None`` on failure.
    """
    # FEWS NET exposes per-country CSV downloads
    url = f"{FEWSNET_PRICE_URL}/market-prices?country={country_iso3}&format=csv"
    logger.info("Fetching FEWS NET prices for %s from %s", country_iso3, url)

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        logger.info("Downloaded %d rows for %s.", len(df), country_iso3)
        return df
    except (requests.RequestException, pd.errors.ParserError) as exc:
        logger.warning("FEWS NET download failed for %s: %s", country_iso3, exc)
        return None


def _load_local_prices() -> Optional[pd.DataFrame]:
    """Load any CSV files already present in ``data/raw/markets/``.

    Returns
    -------
    pd.DataFrame or None
    """
    market_dir = RAW_DIR / "markets"
    csvs = sorted(market_dir.glob("*.csv"))
    if not csvs:
        return None
    frames = []
    for fpath in csvs:
        try:
            frames.append(pd.read_csv(fpath))
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename heterogeneous FEWS NET column names to canonical ones.

    Expected output columns: ``[region_code, date, commodity, market,
    price, currency, unit]``.
    """
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("admin1", "admin_1", "adm1_name", "region", "state", "county"):
            col_map[col] = "admin1_name"
        elif cl in ("date", "period_date", "month"):
            col_map[col] = "date"
        elif cl in ("commodity", "commodity_name", "product"):
            col_map[col] = "commodity"
        elif cl in ("market", "market_name", "mkt_name"):
            col_map[col] = "market"
        elif cl in ("price", "value", "price_value", "avg_price"):
            col_map[col] = "price"
        elif cl in ("currency", "cur"):
            col_map[col] = "currency"
        elif cl in ("unit", "unit_name"):
            col_map[col] = "unit"
        elif cl in ("country", "country_iso3", "iso3", "adm0_code"):
            col_map[col] = "country_iso3"
    return df.rename(columns=col_map)


def _match_region(admin1_name: str, country_iso3: str) -> Optional[str]:
    """Match an admin-1 name to a region code."""
    prefix_map = {"KEN": "KE", "ETH": "ET", "SOM": "SO"}
    prefix = prefix_map.get(country_iso3, "")
    name_lower = str(admin1_name).strip().lower()

    for code, name in ALL_REGIONS.items():
        if not code.startswith(prefix):
            continue
        if name.lower() == name_lower or name.lower() in name_lower or name_lower in name.lower():
            return code
    return None


# =====================================================================
#  Public API
# =====================================================================

def load_market_prices(
    countries: List[str] | None = None,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Load FEWS NET market price data.

    Tries the FEWS NET API first, then falls back to local CSVs in
    ``data/raw/markets/``.

    Parameters
    ----------
    countries : list of str, optional
        ISO-3 country codes.  Defaults to ``["KEN", "ETH", "SOM"]``.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, commodity, market, price,
        currency, unit]``.
    """
    if countries is None:
        countries = ["KEN", "ETH", "SOM"]

    cache_key = f"market_prices_{'_'.join(sorted(countries))}_{start}_{end}"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    # Try API downloads
    frames: list[pd.DataFrame] = []
    for iso3 in countries:
        result = _fetch_fewsnet_prices(iso3)
        if result is not None and not result.empty:
            result["country_iso3"] = iso3
            frames.append(result)

    if frames:
        raw = pd.concat(frames, ignore_index=True)
    else:
        logger.info("FEWS NET API unavailable; loading local CSVs.")
        raw = _load_local_prices()
        if raw is None:
            logger.error(
                "No market price data available. Place CSVs in data/raw/markets/."
            )
            return pd.DataFrame(
                columns=[
                    "region_code", "date", "commodity", "market",
                    "price", "currency", "unit",
                ]
            )

    raw = _normalise_columns(raw)

    # Parse dates
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])
        raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()

    # Parse price
    if "price" in raw.columns:
        raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
        raw = raw.dropna(subset=["price"])

    # Match regions
    if "country_iso3" not in raw.columns:
        raw["country_iso3"] = ""
    if "admin1_name" in raw.columns:
        raw["region_code"] = raw.apply(
            lambda r: _match_region(
                str(r.get("admin1_name", "")),
                str(r.get("country_iso3", "")),
            ),
            axis=1,
        )
    else:
        raw["region_code"] = None

    raw = raw.dropna(subset=["region_code"])

    # Normalise commodity names
    if "commodity" in raw.columns:
        raw["commodity"] = raw["commodity"].astype(str).str.strip().str.lower()

    # Date range filter
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

    # Select columns
    keep = ["region_code", "date", "commodity", "market", "price", "currency", "unit"]
    for c in keep:
        if c not in raw.columns:
            raw[c] = None
    df = raw[keep].copy()
    df = df.sort_values(["region_code", "commodity", "date"]).reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded market prices: %d rows.", len(df))
    return df


def compute_terms_of_trade(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute livestock-to-grain terms of trade (ToT).

    ToT = median livestock price / median grain price for each
    region-month.  A declining ToT signals reduced pastoral purchasing
    power -- a strong famine early-warning signal.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Output of :func:`load_market_prices`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, livestock_price, grain_price,
        tot_livestock_grain]``.
    """
    if prices_df.empty:
        return pd.DataFrame(
            columns=[
                "region_code", "date", "livestock_price",
                "grain_price", "tot_livestock_grain",
            ]
        )

    df = prices_df.copy()
    df["group"] = df["commodity"].apply(
        lambda c: "livestock"
        if c in LIVESTOCK_COMMODITIES
        else ("grain" if c in GRAIN_COMMODITIES else None)
    )
    df = df.dropna(subset=["group"])

    # Median price per region-month-group
    agg = (
        df.groupby(["region_code", "date", "group"], as_index=False)["price"]
        .median()
    )

    livestock = agg[agg["group"] == "livestock"].rename(
        columns={"price": "livestock_price"}
    ).drop(columns="group")
    grain = agg[agg["group"] == "grain"].rename(
        columns={"price": "grain_price"}
    ).drop(columns="group")

    merged = livestock.merge(grain, on=["region_code", "date"], how="inner")
    merged["tot_livestock_grain"] = merged["livestock_price"] / merged["grain_price"].replace(0, float("nan"))

    merged = merged.sort_values(["region_code", "date"]).reset_index(drop=True)
    logger.info("Computed ToT for %d region-months.", len(merged))
    return merged


def compute_price_anomalies(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly price anomalies (z-scores) per commodity per region.

    Anomaly = (price - long_term_monthly_mean) / long_term_monthly_std.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Output of :func:`load_market_prices`.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``price_anomaly``.
    """
    if prices_df.empty:
        return prices_df.assign(price_anomaly=pd.Series(dtype=float))

    df = prices_df.copy()
    df["month"] = df["date"].dt.month

    # Long-term monthly statistics per region-commodity
    stats = (
        df.groupby(["region_code", "commodity", "month"], as_index=False)["price"]
        .agg(mean_price="mean", std_price="std")
    )

    df = df.merge(stats, on=["region_code", "commodity", "month"], how="left")
    df["price_anomaly"] = (df["price"] - df["mean_price"]) / df["std_price"].replace(
        0, float("nan")
    )

    df = df.drop(columns=["month", "mean_price", "std_price"])
    df = df.sort_values(["region_code", "commodity", "date"]).reset_index(drop=True)
    logger.info("Computed price anomalies for %d rows.", len(df))
    return df
