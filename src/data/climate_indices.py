"""Climate teleconnection index loaders.

Downloads and parses three major indices that modulate Horn-of-Africa
rainfall:

- **IOD DMI** (Indian Ocean Dipole -- Dipole Mode Index)
- **ENSO ONI** (El Nino--Southern Oscillation -- Oceanic Nino Index)
- **MJO RMM** (Madden-Julian Oscillation -- Real-time Multivariate MJO)

Each loader returns a monthly-frequency DataFrame with a ``date`` column
(first of month) and index-specific value columns.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Optional

import pandas as pd
import requests

from src.config import ANALYSIS_END, ANALYSIS_START, CLIMATE_INDEX_URLS, RAW_DIR
from src.data.cache import cache_exists, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_TIMEOUT = 60  # seconds for HTTP requests


# =====================================================================
#  IOD -- Dipole Mode Index
# =====================================================================

def load_iod_dmi(
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Download and parse the Indian Ocean Dipole Mode Index (DMI).

    Source: NOAA PSL (HadISST-based DMI).

    The file uses a fixed-width format: the first row is the year range,
    then each subsequent row is ``year val1 val2 ... val12``.

    Parameters
    ----------
    start, end : str
        ISO date bounds for filtering.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, iod_dmi]``.
    """
    cache_key = f"climate_iod_dmi_{start}_{end}"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    url = CLIMATE_INDEX_URLS["iod_dmi"]
    logger.info("Downloading IOD DMI from %s", url)

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as exc:
        logger.error("Failed to download IOD DMI: %s", exc)
        return pd.DataFrame(columns=["date", "iod_dmi"])

    # Save raw file for provenance
    raw_path = RAW_DIR / "climate_indices" / "iod_dmi.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)

    # Parse fixed-width: first non-empty line has start/end years,
    # subsequent lines are year + 12 monthly values, terminated by a
    # line that starts with a year range or -999 sentinel.
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    rows: list[dict] = []

    for line in lines:
        tokens = line.split()
        if len(tokens) < 13:
            continue
        try:
            year = int(tokens[0])
        except ValueError:
            continue
        for month_idx, val_str in enumerate(tokens[1:13], start=1):
            try:
                val = float(val_str)
            except ValueError:
                val = None
            # NOAA uses -999 or -99.99 as missing sentinel
            if val is not None and val < -90:
                val = None
            rows.append(
                {
                    "date": pd.Timestamp(year=year, month=month_idx, day=1),
                    "iod_dmi": val,
                }
            )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["iod_dmi"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.sort_values("date").reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded IOD DMI: %d monthly values.", len(df))
    return df


# =====================================================================
#  ENSO -- Oceanic Nino Index
# =====================================================================

def load_enso_oni(
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Download and parse the ENSO Oceanic Nino Index (ONI).

    Source: NOAA CPC.

    The file is whitespace-delimited with columns:
    ``SEAS  YR  TOTAL  ANOM`` where SEAS is a 3-month season string
    (e.g. ``DJF``) and ANOM is the ONI anomaly.

    Parameters
    ----------
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, oni_index]``.
    """
    cache_key = f"climate_enso_oni_{start}_{end}"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    url = CLIMATE_INDEX_URLS["enso_oni"]
    logger.info("Downloading ENSO ONI from %s", url)

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as exc:
        logger.error("Failed to download ENSO ONI: %s", exc)
        return pd.DataFrame(columns=["date", "oni_index"])

    raw_path = RAW_DIR / "climate_indices" / "enso_oni.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)

    # Season-to-month mapping: use the *centre* month of the 3-month window
    _season_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }

    rows: list[dict] = []
    for line in text.strip().splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        seas = tokens[0].upper()
        if seas not in _season_month:
            continue
        try:
            year = int(tokens[1])
            anom = float(tokens[3])
        except (ValueError, IndexError):
            continue
        month = _season_month[seas]
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=1),
                "oni_index": anom,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.sort_values("date").reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded ENSO ONI: %d monthly values.", len(df))
    return df


# =====================================================================
#  MJO -- Real-time Multivariate MJO Index
# =====================================================================

def load_mjo_rmm(
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Download and parse the MJO RMM1/RMM2 index.

    Source: Australian Bureau of Meteorology (BOM).

    The file is whitespace-delimited with columns:
    ``year month day RMM1 RMM2 phase amplitude``.
    We aggregate to monthly means and modal phase.

    Parameters
    ----------
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, mjo_rmm1, mjo_rmm2, mjo_phase, mjo_amplitude]``.
    """
    cache_key = f"climate_mjo_rmm_{start}_{end}"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    url = CLIMATE_INDEX_URLS["mjo_rmm"]
    logger.info("Downloading MJO RMM from %s", url)

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        text = resp.text
    except requests.RequestException as exc:
        logger.error("Failed to download MJO RMM: %s", exc)
        return pd.DataFrame(
            columns=["date", "mjo_rmm1", "mjo_rmm2", "mjo_phase", "mjo_amplitude"]
        )

    raw_path = RAW_DIR / "climate_indices" / "mjo_rmm.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)

    # Parse daily data
    rows: list[dict] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", "year", "Year")):
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        try:
            year = int(tokens[0])
            month = int(tokens[1])
            day = int(tokens[2])
            rmm1 = float(tokens[3])
            rmm2 = float(tokens[4])
            phase = int(float(tokens[5]))
            amplitude = float(tokens[6])
        except (ValueError, IndexError):
            continue
        # BOM uses 1e36 or 999 as missing
        if abs(rmm1) > 100 or abs(rmm2) > 100:
            continue
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=day),
                "mjo_rmm1": rmm1,
                "mjo_rmm2": rmm2,
                "mjo_phase": phase,
                "mjo_amplitude": amplitude,
            }
        )

    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(
            columns=["date", "mjo_rmm1", "mjo_rmm2", "mjo_phase", "mjo_amplitude"]
        )

    # Monthly aggregation: mean for RMM1/RMM2/amplitude, mode for phase
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        daily.groupby("month_start", as_index=False)
        .agg(
            mjo_rmm1=("mjo_rmm1", "mean"),
            mjo_rmm2=("mjo_rmm2", "mean"),
            mjo_phase=("mjo_phase", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
            mjo_amplitude=("mjo_amplitude", "mean"),
        )
    )
    monthly = monthly.rename(columns={"month_start": "date"})
    monthly = monthly[(monthly["date"] >= start) & (monthly["date"] <= end)]
    monthly = monthly.sort_values("date").reset_index(drop=True)

    save_to_cache(cache_key, monthly)
    logger.info("Loaded MJO RMM: %d monthly values.", len(monthly))
    return monthly


# =====================================================================
#  Convenience: load all indices merged
# =====================================================================

def load_all_climate_indices(
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Load and merge all climate teleconnection indices.

    Merges IOD DMI, ENSO ONI, and MJO RMM on a monthly time axis.

    Parameters
    ----------
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, iod_dmi, oni_index, mjo_rmm1, mjo_rmm2,
        mjo_phase, mjo_amplitude]``.
    """
    iod = load_iod_dmi(start, end)
    oni = load_enso_oni(start, end)
    mjo = load_mjo_rmm(start, end)

    # Build a full monthly date spine
    date_range = pd.DataFrame(
        {"date": pd.date_range(start, end, freq="MS")}
    )

    merged = date_range
    for df in (iod, oni, mjo):
        if not df.empty:
            merged = merged.merge(df, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)
    logger.info(
        "Merged climate indices: %d rows, columns=%s",
        len(merged),
        list(merged.columns),
    )
    return merged
