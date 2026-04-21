"""IPC food security phase data loader.

Fetches Integrated Phase Classification (IPC) acute food insecurity data
from the IPC API or falls back to the Harvard HFID 2025 dataset.  Returns
a panel DataFrame with columns:

    [region_code, date, ipc_phase, population_in_phase]

where *date* is the first of each month and *ipc_phase* is an integer 1-5
conforming to :class:`src.config.IPCPhase`.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional

import pandas as pd
import requests

from src.config import (
    ALL_REGIONS,
    ANALYSIS_END,
    ANALYSIS_START,
    COUNTRY_CODES,
    IPC_API_BASE,
    IPCPhase,
    RAW_DIR,
)
from src.data.cache import cache_exists, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

# FEWS NET textual labels that appear in various source files
_FEWSNET_LABEL_MAP: dict[str, int] = {
    "none": IPCPhase.MINIMAL,
    "minimal": IPCPhase.MINIMAL,
    "none/minimal": IPCPhase.MINIMAL,
    "stressed": IPCPhase.STRESSED,
    "crisis": IPCPhase.CRISIS,
    "emergency": IPCPhase.EMERGENCY,
    "famine": IPCPhase.FAMINE,
    "catastrophe": IPCPhase.FAMINE,
    # Numeric strings
    "1": IPCPhase.MINIMAL,
    "2": IPCPhase.STRESSED,
    "3": IPCPhase.CRISIS,
    "4": IPCPhase.EMERGENCY,
    "5": IPCPhase.FAMINE,
}

# ── IPC API helpers ─────────────────────────────────────────────────────

_API_TIMEOUT = 60  # seconds


def _fetch_ipc_api(country_iso3: str) -> Optional[pd.DataFrame]:
    """Query the IPC API for a single country.

    Parameters
    ----------
    country_iso3 : str
        ISO 3166-1 alpha-3 country code (e.g. ``"KEN"``).

    Returns
    -------
    pd.DataFrame or None
        Parsed DataFrame or ``None`` when the request fails.
    """
    url = f"{IPC_API_BASE}"
    params = {
        "country": country_iso3,
        "format": "json",
        "type": "A",  # acute
    }
    try:
        resp = requests.get(url, params=params, timeout=_API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("IPC API request failed for %s: %s", country_iso3, exc)
        return None

    if not data:
        logger.warning("IPC API returned empty payload for %s", country_iso3)
        return None

    rows: list[dict] = []
    for record in data:
        # Each record covers an analysis period with area-level phase data
        analysis_date = record.get("analysis_date") or record.get("date")
        areas = record.get("areas") or record.get("groups", [])
        for area in areas:
            area_name = area.get("name", "")
            for phase_num in range(1, 6):
                pop_key = f"phase{phase_num}_population"
                pop = area.get(pop_key, area.get(f"p{phase_num}", 0))
                if pop:
                    rows.append(
                        {
                            "area_name": area_name,
                            "date": analysis_date,
                            "ipc_phase": phase_num,
                            "population_in_phase": int(pop),
                            "country_iso3": country_iso3,
                        }
                    )
    if not rows:
        return None
    return pd.DataFrame(rows)


# ── HFID fallback ──────────────────────────────────────────────────────

def _load_hfid_csv(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Attempt to load the Harvard HFID 2025 CSV from *data/raw/ipc/*.

    Parameters
    ----------
    path : str, optional
        Explicit path.  When ``None`` the function searches
        ``data/raw/ipc/`` for any CSV.

    Returns
    -------
    pd.DataFrame or None
    """
    ipc_dir = RAW_DIR / "ipc"
    if path:
        candidates = [ipc_dir / path]
    else:
        candidates = sorted(ipc_dir.glob("*.csv"))

    for fpath in candidates:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                logger.info("Loaded HFID data from %s (%d rows)", fpath, len(df))
                return df
            except Exception as exc:
                logger.warning("Failed to read %s: %s", fpath, exc)
    return None


# ── Region matching ────────────────────────────────────────────────────

def _match_region_code(area_name: str, country_iso3: str) -> Optional[str]:
    """Best-effort fuzzy match of an IPC area name to our region codes.

    Uses the region-name mapping in :data:`src.config.ALL_REGIONS`.

    Parameters
    ----------
    area_name : str
        Raw area/region name from the source data.
    country_iso3 : str
        Country ISO3 to narrow the search.

    Returns
    -------
    str or None
        Matching region code or ``None``.
    """
    prefix_map = {"KEN": "KE", "ETH": "ET", "SOM": "SO"}
    prefix = prefix_map.get(country_iso3, "")
    area_lower = area_name.strip().lower()

    for code, name in ALL_REGIONS.items():
        if not code.startswith(prefix):
            continue
        if name.lower() == area_lower:
            return code
        # Partial match (e.g. "Turkana County" -> "Turkana")
        if area_lower.startswith(name.lower()) or name.lower().startswith(area_lower):
            return code
    return None


# ── Forward-fill ───────────────────────────────────────────────────────

def _forward_fill_monthly(
    df: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    """Expand IPC observations to a continuous monthly panel.

    IPC analyses happen 2-3 times per year.  Between analyses the most
    recent classification is carried forward.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``[region_code, date, ipc_phase, population_in_phase]``.
    start, end : str
        ISO date strings bounding the desired range.

    Returns
    -------
    pd.DataFrame
        Monthly panel with forward-filled phases.
    """
    if df.empty:
        return df

    date_range = pd.date_range(start, end, freq="MS")
    regions = df["region_code"].unique()

    filled_parts: list[pd.DataFrame] = []
    for rc in regions:
        region_df = df[df["region_code"] == rc].copy()
        region_df = region_df.sort_values("date")
        region_df = region_df.set_index("date").reindex(date_range)
        region_df["region_code"] = rc
        region_df[["ipc_phase", "population_in_phase"]] = region_df[
            ["ipc_phase", "population_in_phase"]
        ].ffill()
        region_df = region_df.reset_index().rename(columns={"index": "date"})
        filled_parts.append(region_df)

    return pd.concat(filled_parts, ignore_index=True)


# ── Main entry point ──────────────────────────────────────────────────

def load_ipc_data(
    countries: List[str] | None = None,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Load IPC acute food insecurity data for the Horn of Africa.

    Tries the IPC API first; on failure loads the Harvard HFID 2025
    dataset from ``data/raw/ipc/``.  Results are cached as parquet.

    Parameters
    ----------
    countries : list of str, optional
        ISO3 country codes.  Defaults to ``["KEN", "ETH", "SOM"]``.
    start, end : str
        ISO date strings.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, ipc_phase, population_in_phase]``.
        ``date`` is ``pd.Timestamp`` at month start, ``ipc_phase`` is
        an integer 1-5.
    """
    if countries is None:
        countries = ["KEN", "ETH", "SOM"]

    cache_key = f"ipc_{'_'.join(sorted(countries))}_{start}_{end}"
    if cache_exists(cache_key):
        logger.info("IPC data found in cache.")
        return load_from_cache(cache_key)

    # ── Try IPC API first ──────────────────────────────────────────
    api_frames: list[pd.DataFrame] = []
    for iso3 in countries:
        logger.info("Querying IPC API for %s ...", iso3)
        result = _fetch_ipc_api(iso3)
        if result is not None and not result.empty:
            api_frames.append(result)

    if api_frames:
        raw = pd.concat(api_frames, ignore_index=True)
    else:
        # ── Fall back to HFID CSV ──────────────────────────────────
        logger.info("IPC API unavailable — falling back to HFID CSV.")
        raw = _load_hfid_csv()
        if raw is None:
            logger.error(
                "No IPC data source available. Place HFID CSV in data/raw/ipc/."
            )
            return pd.DataFrame(
                columns=["region_code", "date", "ipc_phase", "population_in_phase"]
            )

    # ── Normalise columns ──────────────────────────────────────────
    col_map: dict[str, str] = {}
    for col in raw.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("area", "area_name", "admin1", "region", "admin1_name"):
            col_map[col] = "area_name"
        elif cl in ("date", "analysis_date", "period_date", "reference_date"):
            col_map[col] = "date"
        elif cl in ("ipc_phase", "phase", "classification", "cs"):
            col_map[col] = "ipc_phase"
        elif cl in (
            "population_in_phase",
            "population",
            "pop",
            "affected_population",
        ):
            col_map[col] = "population_in_phase"
        elif cl in ("country", "country_iso3", "iso3", "adm0_code"):
            col_map[col] = "country_iso3"
    raw = raw.rename(columns=col_map)

    # Ensure required columns exist
    required = {"area_name", "date", "ipc_phase"}
    missing = required - set(raw.columns)
    if missing:
        logger.error("IPC raw data missing columns: %s", missing)
        return pd.DataFrame(
            columns=["region_code", "date", "ipc_phase", "population_in_phase"]
        )

    # Map FEWS NET textual labels to integers
    if raw["ipc_phase"].dtype == object:
        raw["ipc_phase"] = (
            raw["ipc_phase"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(_FEWSNET_LABEL_MAP)
        )
    raw["ipc_phase"] = pd.to_numeric(raw["ipc_phase"], errors="coerce")
    raw = raw.dropna(subset=["ipc_phase"])
    raw["ipc_phase"] = raw["ipc_phase"].astype(int)

    # Ensure population column
    if "population_in_phase" not in raw.columns:
        raw["population_in_phase"] = 0
    raw["population_in_phase"] = pd.to_numeric(
        raw["population_in_phase"], errors="coerce"
    ).fillna(0).astype(int)

    # Parse dates
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()

    # Map area names to canonical region codes
    if "country_iso3" not in raw.columns:
        raw["country_iso3"] = raw.get("country", "").astype(str).str.upper()

    raw["region_code"] = raw.apply(
        lambda r: _match_region_code(
            str(r.get("area_name", "")), str(r.get("country_iso3", ""))
        ),
        axis=1,
    )
    raw = raw.dropna(subset=["region_code"])

    # Filter to requested countries
    prefix_set = {
        COUNTRY_CODES.get(iso3[:2], iso3)[:2]
        for iso3 in countries
    }
    # Build prefix set from ISO3 -> 2-letter prefix
    iso3_to_prefix = {"KEN": "KE", "ETH": "ET", "SOM": "SO"}
    prefixes = {iso3_to_prefix.get(c, c[:2]) for c in countries}
    raw = raw[raw["region_code"].str[:2].isin(prefixes)]

    # Keep only needed columns and aggregate (take max phase per region-month)
    df = (
        raw.groupby(["region_code", "date"], as_index=False)
        .agg({"ipc_phase": "max", "population_in_phase": "sum"})
    )

    # Filter date range
    df = df[(df["date"] >= start) & (df["date"] <= end)]

    # Forward-fill to monthly panel
    df = _forward_fill_monthly(df, start, end)

    # Drop rows that are still NaN after forward-fill (before first observation)
    df = df.dropna(subset=["ipc_phase"])
    df["ipc_phase"] = df["ipc_phase"].astype(int)
    df["population_in_phase"] = df["population_in_phase"].fillna(0).astype(int)

    # Final column order
    df = df[["region_code", "date", "ipc_phase", "population_in_phase"]]
    df = df.sort_values(["region_code", "date"]).reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded IPC data: %d rows across %d regions.", len(df), df["region_code"].nunique())
    return df
