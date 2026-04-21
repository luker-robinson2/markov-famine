"""Loaders for static and slow-changing datasets.

These variables change rarely (or not at all) compared to monthly climate
and market data:

- **Conflict events** (ACLED) -- aggregated to monthly counts per region.
- **Population density** (WorldPop / FAO) -- annual or static estimates.
- **Livelihood zones** (FEWS NET) -- pastoral / agro-pastoral /
  agricultural classification per region (essentially static).
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
    RAW_DIR,
)
from src.data.cache import cache_exists, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_TIMEOUT = 90  # seconds

# ACLED API endpoint (public-access CSV export)
_ACLED_API_URL = "https://api.acleddata.com/acled/read.csv"

# Livelihood zone canonical types
LIVELIHOOD_TYPES = {"pastoral", "agro-pastoral", "agricultural", "urban", "riverine"}


# ── Helpers ─────────────────────────────────────────────────────────────

def _match_region(admin1_name: str, country_iso3: str) -> Optional[str]:
    """Best-effort match of an admin-1 name to our canonical codes."""
    prefix_map = {"KEN": "KE", "ETH": "ET", "SOM": "SO",
                  "Kenya": "KE", "Ethiopia": "ET", "Somalia": "SO"}
    prefix = prefix_map.get(country_iso3, "")
    name_lower = str(admin1_name).strip().lower()

    for code, name in ALL_REGIONS.items():
        if not code.startswith(prefix):
            continue
        if (
            name.lower() == name_lower
            or name.lower() in name_lower
            or name_lower in name.lower()
        ):
            return code
    return None


def _iso3_to_country_name(iso3: str) -> str:
    """Map ISO3 to the country name used by ACLED."""
    return {"KEN": "Kenya", "ETH": "Ethiopia", "SOM": "Somalia"}.get(iso3, iso3)


# =====================================================================
#  Conflict data -- ACLED
# =====================================================================

def _fetch_acled_api(country_iso3: str) -> Optional[pd.DataFrame]:
    """Query the ACLED API for conflict events in a single country.

    Requires ACLED API key and email to be set in environment variables
    ``ACLED_API_KEY`` and ``ACLED_EMAIL``, or falls back to local CSVs.

    Parameters
    ----------
    country_iso3 : str
        ISO3 country code.

    Returns
    -------
    pd.DataFrame or None
    """
    import os

    api_key = os.environ.get("ACLED_API_KEY", "")
    email = os.environ.get("ACLED_EMAIL", "")

    if not api_key or not email:
        logger.info("ACLED credentials not set; skipping API download.")
        return None

    country_name = _iso3_to_country_name(country_iso3)
    params = {
        "key": api_key,
        "email": email,
        "country": country_name,
        "limit": 0,  # no limit
    }

    try:
        resp = requests.get(_ACLED_API_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        logger.info("Downloaded %d ACLED events for %s.", len(df), country_iso3)
        return df
    except (requests.RequestException, pd.errors.ParserError) as exc:
        logger.warning("ACLED API request failed for %s: %s", country_iso3, exc)
        return None


def _load_local_acled() -> Optional[pd.DataFrame]:
    """Load ACLED CSV files from ``data/raw/static/``."""
    static_dir = RAW_DIR / "static"
    csvs = sorted(static_dir.glob("*acled*.csv")) + sorted(
        static_dir.glob("*conflict*.csv")
    )
    if not csvs:
        return None
    frames: list[pd.DataFrame] = []
    for fpath in csvs:
        try:
            frames.append(pd.read_csv(fpath))
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_conflict_data(
    countries: List[str] | None = None,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Load ACLED conflict events, aggregated to monthly counts per region.

    Parameters
    ----------
    countries : list of str, optional
        ISO-3 country codes.  Defaults to ``["KEN", "ETH", "SOM"]``.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, conflict_events, fatalities]``.
        ``date`` is month-start timestamp.
    """
    if countries is None:
        countries = ["KEN", "ETH", "SOM"]

    cache_key = f"conflict_{'_'.join(sorted(countries))}_{start}_{end}"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    # Try API then local files
    frames: list[pd.DataFrame] = []
    for iso3 in countries:
        result = _fetch_acled_api(iso3)
        if result is not None and not result.empty:
            result["country_iso3"] = iso3
            frames.append(result)

    if frames:
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = _load_local_acled()
        if raw is None:
            logger.error(
                "No ACLED data available. Place CSV(s) in data/raw/static/ "
                "or set ACLED_API_KEY and ACLED_EMAIL env vars."
            )
            return pd.DataFrame(
                columns=["region_code", "date", "conflict_events", "fatalities"]
            )

    # Normalise column names
    col_renames: dict[str, str] = {}
    for col in raw.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("admin1", "admin_1", "admin1_name", "region"):
            col_renames[col] = "admin1_name"
        elif cl in ("event_date", "date"):
            col_renames[col] = "event_date"
        elif cl in ("fatalities", "deaths"):
            col_renames[col] = "fatalities"
        elif cl in ("country", "country_iso3"):
            col_renames[col] = "country_iso3"
        elif cl in ("event_type",):
            col_renames[col] = "event_type"
    raw = raw.rename(columns=col_renames)

    # Parse date
    if "event_date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["event_date"], errors="coerce")
    elif "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    else:
        logger.error("ACLED data has no recognisable date column.")
        return pd.DataFrame(
            columns=["region_code", "date", "conflict_events", "fatalities"]
        )

    raw = raw.dropna(subset=["date"])
    raw["date"] = raw["date"].dt.to_period("M").dt.to_timestamp()

    # Ensure fatalities column
    if "fatalities" not in raw.columns:
        raw["fatalities"] = 0
    raw["fatalities"] = pd.to_numeric(raw["fatalities"], errors="coerce").fillna(0).astype(int)

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

    # Date range filter
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

    # Aggregate: monthly event counts and fatality sums per region
    agg = (
        raw.groupby(["region_code", "date"], as_index=False)
        .agg(
            conflict_events=("region_code", "size"),
            fatalities=("fatalities", "sum"),
        )
    )

    agg = agg.sort_values(["region_code", "date"]).reset_index(drop=True)

    save_to_cache(cache_key, agg)
    logger.info("Loaded conflict data: %d region-months.", len(agg))
    return agg


# =====================================================================
#  Population data
# =====================================================================

def load_population_data() -> pd.DataFrame:
    """Load population density estimates per region.

    Looks for a CSV in ``data/raw/static/`` with columns that map to
    ``[region_code, year, population, population_density]``.

    Sources: FAO or WorldPop.  If unavailable returns an empty frame.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, year, population, population_density]``.
    """
    cache_key = "population_static"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    static_dir = RAW_DIR / "static"
    candidates = sorted(static_dir.glob("*pop*.csv")) + sorted(
        static_dir.glob("*worldpop*.csv")
    )

    raw: Optional[pd.DataFrame] = None
    for fpath in candidates:
        try:
            raw = pd.read_csv(fpath)
            logger.info("Loaded population data from %s (%d rows).", fpath, len(raw))
            break
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)

    if raw is None or raw.empty:
        logger.warning(
            "No population data found. Place a CSV in data/raw/static/."
        )
        return pd.DataFrame(
            columns=["region_code", "year", "population", "population_density"]
        )

    # Normalise columns
    col_renames: dict[str, str] = {}
    for col in raw.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("admin1", "admin1_name", "region", "county"):
            col_renames[col] = "admin1_name"
        elif cl in ("year",):
            col_renames[col] = "year"
        elif cl in ("population", "pop", "total_population"):
            col_renames[col] = "population"
        elif cl in ("population_density", "pop_density", "density"):
            col_renames[col] = "population_density"
        elif cl in ("country", "country_iso3", "iso3"):
            col_renames[col] = "country_iso3"
    raw = raw.rename(columns=col_renames)

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

    # Ensure numeric columns
    for c in ("year", "population", "population_density"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    keep = ["region_code", "year", "population", "population_density"]
    for c in keep:
        if c not in raw.columns:
            raw[c] = None
    df = raw[keep].copy().dropna(subset=["region_code"])
    df = df.sort_values(["region_code", "year"]).reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded population data: %d rows.", len(df))
    return df


# =====================================================================
#  Livelihood zones
# =====================================================================

def load_livelihood_zones() -> pd.DataFrame:
    """Load FEWS NET livelihood zone classification per region.

    Expected categories: ``pastoral``, ``agro-pastoral``, ``agricultural``,
    ``urban``, ``riverine``.

    Looks for a CSV in ``data/raw/static/`` with livelihood zone info.
    When no file is found, returns a sensible default mapping based on
    known FEWS NET classifications for the Horn of Africa.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, region_name, livelihood_type]``.
    """
    cache_key = "livelihood_zones_static"
    if cache_exists(cache_key):
        return load_from_cache(cache_key)

    static_dir = RAW_DIR / "static"
    candidates = sorted(static_dir.glob("*livelihood*.csv")) + sorted(
        static_dir.glob("*lhz*.csv")
    )

    raw: Optional[pd.DataFrame] = None
    for fpath in candidates:
        try:
            raw = pd.read_csv(fpath)
            logger.info("Loaded livelihood zones from %s (%d rows).", fpath, len(raw))
            break
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)

    if raw is not None and not raw.empty:
        # Normalise columns
        col_renames: dict[str, str] = {}
        for col in raw.columns:
            cl = col.strip().lower().replace(" ", "_")
            if cl in ("admin1", "admin1_name", "region", "county"):
                col_renames[col] = "admin1_name"
            elif cl in ("livelihood_type", "livelihood_zone", "lhz_type", "type", "classification"):
                col_renames[col] = "livelihood_type"
            elif cl in ("country", "country_iso3", "iso3"):
                col_renames[col] = "country_iso3"
        raw = raw.rename(columns=col_renames)

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

        if "livelihood_type" in raw.columns:
            raw["livelihood_type"] = (
                raw["livelihood_type"].astype(str).str.strip().str.lower()
            )

        raw["region_name"] = raw["region_code"].map(ALL_REGIONS)
        df = raw[["region_code", "region_name", "livelihood_type"]].copy()
    else:
        # Fallback: hard-coded defaults from FEWS NET classification
        logger.info(
            "No livelihood zone CSV found; using hard-coded FEWS NET defaults."
        )
        df = _default_livelihood_zones()

    df = df.sort_values("region_code").reset_index(drop=True)

    save_to_cache(cache_key, df)
    logger.info("Loaded livelihood zones: %d regions.", len(df))
    return df


def _default_livelihood_zones() -> pd.DataFrame:
    """Return hard-coded FEWS NET livelihood zone classification.

    These represent the *dominant* livelihood type for each admin-1 area
    based on FEWS NET documentation.  In practice many regions span
    multiple zones; this provides the primary classification.
    """
    # Kenya -- pastoral counties vs. agricultural counties
    pastoral_ke = {
        "KE004", "KE005", "KE007", "KE008", "KE009", "KE010", "KE011",
        "KE023", "KE024", "KE025", "KE030", "KE031", "KE033", "KE034",
    }
    agropastoral_ke = {
        "KE002", "KE003", "KE006", "KE012", "KE013", "KE015", "KE017",
        "KE028", "KE036",
    }
    urban_ke = {"KE047"}  # Nairobi

    # Ethiopia -- pastoral regions
    pastoral_et = {"ET002", "ET005"}  # Afar, Somali
    agropastoral_et = {"ET006", "ET008"}  # Benishangul-Gumuz, Gambela
    urban_et = {"ET010", "ET011"}  # Addis Ababa, Dire Dawa

    # Somalia -- predominantly pastoral
    pastoral_so = {
        "SO001", "SO002", "SO003", "SO004", "SO005", "SO006", "SO007",
        "SO008", "SO009", "SO016", "SO017", "SO018",
    }
    agropastoral_so = {"SO010", "SO014", "SO015"}  # Hiraan, Bay, Bakool
    urban_so = {"SO012"}  # Banaadir (Mogadishu)
    riverine_so = {"SO011", "SO013"}  # Middle Shabelle, Lower Shabelle

    rows: list[dict] = []
    for code, name in ALL_REGIONS.items():
        if code in pastoral_ke | pastoral_et | pastoral_so:
            ltype = "pastoral"
        elif code in agropastoral_ke | agropastoral_et | agropastoral_so:
            ltype = "agro-pastoral"
        elif code in urban_ke | urban_et | urban_so:
            ltype = "urban"
        elif code in riverine_so:
            ltype = "riverine"
        else:
            ltype = "agricultural"
        rows.append(
            {"region_code": code, "region_name": name, "livelihood_type": ltype}
        )

    return pd.DataFrame(rows)
