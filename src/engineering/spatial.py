"""Spatial alignment utilities for the food security prediction system.

Maps between different geographic naming conventions (IPC, GEE/GAUL, internal
region codes) and standardizes all DataFrames to the project's canonical
admin-1 region codes defined in ``src.config``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

from src.config import (
    ALL_REGIONS,
    COUNTRY_CODES,
    DATA_DIR,
    ETHIOPIA_REGIONS,
    GAUL_ASSET,
    GAUL_COUNTRY_CODES,
    KENYA_REGIONS,
    SOMALIA_REGIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known name mismatches — source name  ->  canonical config name
# ---------------------------------------------------------------------------
_IPC_NAME_OVERRIDES: dict[str, str] = {
    # Kenya
    "Nairobi County": "Nairobi",
    "Nairobi City": "Nairobi",
    "Tana River County": "Tana River",
    "Taita-Taveta": "Taita Taveta",
    "Elgeyo-Marakwet": "Elgeyo Marakwet",
    "Trans-Nzoia": "Trans Nzoia",
    "Uasin-Gishu": "Uasin Gishu",
    "Homa-Bay": "Homa Bay",
    "West-Pokot": "West Pokot",
    "Tharaka-Nithi": "Tharaka Nithi",
    "Murang'a": "Muranga",
    # Ethiopia
    "Southern Nations": "SNNPR",
    "Southern Nations, Nationalities, and Peoples' Region": "SNNPR",
    "Southern Nations, Nationalities and Peoples": "SNNPR",
    "Benishangul Gumuz": "Benishangul-Gumuz",
    "Benshangul-Gumuz": "Benishangul-Gumuz",
    "Addis Abeba": "Addis Ababa",
    "Dire Dawa City": "Dire Dawa",
    "Diredawa": "Dire Dawa",
    # Somalia
    "Woqooyi-Galbeed": "Woqooyi Galbeed",
    "Banaadir (Mogadishu)": "Banaadir",
    "Mogadishu": "Banaadir",
    "Lower Shabelle": "Lower Shabelle",
    "Middle Shabelle": "Middle Shabelle",
    "Middle Juba": "Middle Juba",
    "Lower Juba": "Lower Juba",
    "Galgaduud": "Galgaduud",
}

# Reverse lookup: canonical name -> region code
_NAME_TO_CODE: dict[str, str] = {v: k for k, v in ALL_REGIONS.items()}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_region_mapping() -> dict:
    """Build a comprehensive mapping between naming conventions.

    Returns
    -------
    dict
        Keys: ``"ipc_to_code"``, ``"gaul_to_code"``, ``"code_to_name"``,
        ``"name_to_code"``.  Each value is a ``dict[str, str]``.
    """
    ipc_to_code: dict[str, str] = {}
    gaul_to_code: dict[str, str] = {}
    code_to_name: dict[str, str] = dict(ALL_REGIONS)
    name_to_code: dict[str, str] = dict(_NAME_TO_CODE)

    # Populate IPC -> code mapping via overrides + canonical names
    for ipc_name, canonical in _IPC_NAME_OVERRIDES.items():
        code = _NAME_TO_CODE.get(canonical)
        if code is not None:
            ipc_to_code[ipc_name.lower()] = code

    for code, name in ALL_REGIONS.items():
        # Direct canonical name
        ipc_to_code[name.lower()] = code
        # GAUL names are uppercase by convention
        gaul_to_code[name.upper()] = code

    return {
        "ipc_to_code": ipc_to_code,
        "gaul_to_code": gaul_to_code,
        "code_to_name": code_to_name,
        "name_to_code": name_to_code,
    }


def _normalize_name(name: str) -> str:
    """Lower-case, strip, collapse whitespace, remove 'county'/'region'."""
    name = name.strip().lower()
    for suffix in (" county", " region", " city", " province"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return " ".join(name.split())


def _resolve_name_to_code(raw_name: str, mapping: dict[str, str]) -> Optional[str]:
    """Resolve a raw region name to an internal region code.

    Tries, in order:
    1. Direct lookup in overrides.
    2. Normalized direct lookup in canonical names.
    3. Fuzzy match (score >= 85) against canonical names.
    """
    # 1. Override table (case-insensitive)
    override_canonical = _IPC_NAME_OVERRIDES.get(raw_name)
    if override_canonical is None:
        override_canonical = _IPC_NAME_OVERRIDES.get(raw_name.strip())
    if override_canonical is not None:
        return _NAME_TO_CODE.get(override_canonical)

    # 2. Direct canonical match
    normed = _normalize_name(raw_name)
    for code, canonical in ALL_REGIONS.items():
        if _normalize_name(canonical) == normed:
            return code

    # 3. Fuzzy match
    candidates = list(ALL_REGIONS.values())
    match, score = process.extractOne(raw_name, candidates, scorer=fuzz.token_sort_ratio)
    if score >= 85:
        return _NAME_TO_CODE[match]

    return None


def align_to_admin1(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Standardize any DataFrame's region column to internal codes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column named ``region`` (or ``admin1``, ``ADM1_NAME``).
    source : str
        Identifier for the upstream source, used for logging only.
        Common values: ``"ipc"``, ``"gee"``, ``"fewsnet"``, ``"hfid"``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``region_code`` column and the original region
        names preserved in ``region_name_raw``.
    """
    df = df.copy()

    # Detect the region column
    region_col: Optional[str] = None
    for candidate in ("region", "admin1", "ADM1_NAME", "ADMIN1", "region_name"):
        if candidate in df.columns:
            region_col = candidate
            break
    if region_col is None:
        raise KeyError(
            f"Cannot find a region column in DataFrame from source={source!r}. "
            f"Columns: {list(df.columns)}"
        )

    mapping = build_region_mapping()

    df.rename(columns={region_col: "region_name_raw"}, inplace=True)

    codes: list[Optional[str]] = []
    unresolved: set[str] = set()

    for raw in df["region_name_raw"]:
        if pd.isna(raw):
            codes.append(None)
            continue
        raw_str = str(raw)

        # If already a code (e.g. "KE001"), pass through
        if raw_str in ALL_REGIONS:
            codes.append(raw_str)
            continue

        code = _resolve_name_to_code(raw_str, mapping)
        if code is None:
            unresolved.add(raw_str)
        codes.append(code)

    df["region_code"] = codes

    if unresolved:
        logger.warning(
            "[spatial.align_to_admin1] %d unresolved region names from source=%r: %s",
            len(unresolved),
            source,
            sorted(unresolved),
        )

    return df


def load_admin1_geometries(
    shapefile_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Load admin-1 boundary geometries for the three study countries.

    Parameters
    ----------
    shapefile_path : Path, optional
        Path to a local shapefile / GeoJSON with admin-1 boundaries.  If
        ``None`` the function looks for
        ``data/raw/boundaries/admin1_boundaries.geojson`` under the project
        root and, failing that, returns an empty ``GeoDataFrame`` with the
        correct schema.

    Returns
    -------
    gpd.GeoDataFrame
        With columns ``region_code``, ``region_name``, ``country``, and
        ``geometry``.
    """
    if shapefile_path is None:
        shapefile_path = DATA_DIR / "raw" / "boundaries" / "admin1_boundaries.geojson"

    if not shapefile_path.exists():
        logger.warning(
            "[spatial.load_admin1_geometries] Boundary file not found at %s. "
            "Returning empty GeoDataFrame. Download GAUL boundaries or provide "
            "a local path.",
            shapefile_path,
        )
        return gpd.GeoDataFrame(
            columns=["region_code", "region_name", "country", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.read_file(shapefile_path)

    # Attempt alignment
    gdf = align_to_admin1(gdf, source="boundaries")

    # Derive country from code prefix
    gdf["country"] = gdf["region_code"].str[:2].map(
        lambda c: COUNTRY_CODES.get(c, c) if pd.notna(c) else None
    )

    # Standardise column names
    keep_cols = ["region_code", "region_name_raw", "country", "geometry"]
    gdf = gdf[[c for c in keep_cols if c in gdf.columns]].rename(
        columns={"region_name_raw": "region_name"}
    )

    gdf = gdf.to_crs("EPSG:4326")

    return gdf
