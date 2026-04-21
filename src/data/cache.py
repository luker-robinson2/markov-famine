"""Simple parquet-based cache for data acquisition.

Stores DataFrames as parquet files under data/raw/ with deterministic keys
so repeated runs never re-download the same data.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


def _cache_path(key: str) -> Path:
    """Return the filesystem path for a given cache key.

    Parameters
    ----------
    key : str
        Deterministic cache key, typically
        ``f"{variable}_{region_code}_{start}_{end}"``.

    Returns
    -------
    Path
        Absolute path to the parquet file under ``data/raw/``.
    """
    # Sanitise the key so it is safe as a filename
    safe = key.replace("/", "_").replace(" ", "_").replace(":", "")
    return RAW_DIR / f"{safe}.parquet"


def cache_exists(key: str) -> bool:
    """Check whether a cached parquet file exists for *key*.

    Parameters
    ----------
    key : str
        Cache key (see :func:`_cache_path`).

    Returns
    -------
    bool
        ``True`` if the file exists and is non-empty.
    """
    p = _cache_path(key)
    return p.exists() and p.stat().st_size > 0


def save_to_cache(key: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to the parquet cache.

    Parameters
    ----------
    key : str
        Deterministic cache key.
    df : pd.DataFrame
        Data to cache.  Must be parquet-serialisable.
    """
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", index=False)
    logger.info("Cached %d rows -> %s", len(df), p)


def load_from_cache(key: str) -> pd.DataFrame:
    """Load a previously cached DataFrame.

    Parameters
    ----------
    key : str
        Cache key that was used in :func:`save_to_cache`.

    Returns
    -------
    pd.DataFrame
        The cached data.

    Raises
    ------
    FileNotFoundError
        If no cached file exists for *key*.
    """
    p = _cache_path(key)
    if not p.exists():
        raise FileNotFoundError(f"No cached data for key '{key}' at {p}")
    logger.info("Loading cache <- %s", p)
    return pd.read_parquet(p, engine="pyarrow")


def make_cache_key(variable: str, region_code: str, start: str, end: str) -> str:
    """Build a deterministic cache key.

    Parameters
    ----------
    variable : str
        Name of the variable (e.g. ``"chirps"``, ``"ndvi"``).
    region_code : str
        Region identifier (e.g. ``"KE007"``).
    start, end : str
        ISO-format date strings bounding the query.

    Returns
    -------
    str
        Key of the form ``"{variable}_{region_code}_{start}_{end}"``.
    """
    return f"{variable}_{region_code}_{start}_{end}"
