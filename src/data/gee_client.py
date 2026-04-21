"""Google Earth Engine data acquisition client.

Provides functions to extract monthly climate and vegetation variables from
GEE for admin-1 regions in the Horn of Africa.  Every public function
returns a tidy DataFrame with columns ``[region_code, date, <value_cols>]``
and checks the parquet cache before issuing any GEE request.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Load .env for GEE_PROJECT_ID and credentials
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
except ImportError:
    pass

from src.config import (
    ALL_REGIONS,
    ANALYSIS_END,
    ANALYSIS_START,
    ERA5_BANDS,
    GAUL_ASSET,
    GAUL_COUNTRY_CODES,
    GEE_ASSETS,
    MODIS_EVI_BAND,
    MODIS_LST_DAY_BAND,
    MODIS_LST_NIGHT_BAND,
    MODIS_NDVI_BAND,
)
from src.data.cache import (
    cache_exists,
    load_from_cache,
    make_cache_key,
    save_to_cache,
)

logger = logging.getLogger(__name__)

# Lazy-imported so the module can be imported even without ``ee`` installed.
ee = None  # type: ignore[assignment]

# ── Batch / retry knobs ─────────────────────────────────────────────────
BATCH_SIZE = 12          # regions processed per GEE request
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0    # seconds


# =====================================================================
#  GEE initialisation
# =====================================================================

def initialize_gee() -> None:
    """Authenticate and initialise the Earth Engine API.

    Supports two modes:

    1. **Service account** -- set ``GEE_SERVICE_ACCOUNT`` and
       ``GEE_KEY_FILE`` environment variables.
    2. **Interactive** -- falls back to ``ee.Authenticate()`` /
       ``ee.Initialize()`` using default credentials.

    Raises
    ------
    RuntimeError
        If initialisation fails after all attempts.
    """
    global ee
    import ee as _ee  # type: ignore[import-untyped]

    ee = _ee

    service_account = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_file = os.environ.get("GEE_KEY_FILE")
    project_id = os.environ.get("GEE_PROJECT_ID")

    try:
        if service_account and key_file:
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials, project=project_id)
            logger.info("GEE initialised with service account %s.", service_account)
        else:
            # Prefer gcloud application-default credentials (avoids OAuth app blocks)
            try:
                import google.auth
                credentials, _ = google.auth.default(
                    scopes=[
                        "https://www.googleapis.com/auth/earthengine",
                        "https://www.googleapis.com/auth/cloud-platform",
                    ]
                )
                ee.Initialize(credentials=credentials, project=project_id)
                logger.info("GEE initialised with gcloud ADC (project=%s).", project_id)
            except Exception:
                # Fall back to ee's own auth flow
                try:
                    ee.Initialize(project=project_id)
                except Exception:
                    ee.Authenticate()
                    ee.Initialize(project=project_id)
                logger.info("GEE initialised with ee default credentials (project=%s).", project_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Google Earth Engine: {exc}") from exc


def _ensure_ee() -> None:
    """Guarantee that ``ee`` is initialised."""
    if ee is None:
        initialize_gee()


# =====================================================================
#  Geometry helpers
# =====================================================================

# In-memory geometry cache so we don't re-fetch GAUL boundaries every call.
_geometry_cache: Dict[str, object] = {}


def _country_iso3(region_code: str) -> str:
    """Map a region code prefix to ISO3 (e.g. ``'KE'`` -> ``'KEN'``)."""
    mapping = {"KE": "KEN", "ET": "ETH", "SO": "SOM"}
    return mapping[region_code[:2]]


def get_region_geometry(region_code: str):
    """Return a GEE ``ee.Geometry`` for an admin-1 region.

    Uses FAO GAUL 2015 level-1 boundaries filtered by country GAUL code
    and admin-1 name from :data:`src.config.ALL_REGIONS`.

    Parameters
    ----------
    region_code : str
        Code such as ``"KE007"`` (Garissa, Kenya).

    Returns
    -------
    ee.Geometry
        Polygon/MultiPolygon geometry of the admin-1 unit.
    """
    _ensure_ee()

    if region_code in _geometry_cache:
        return _geometry_cache[region_code]

    region_name = ALL_REGIONS.get(region_code)
    if region_name is None:
        raise ValueError(f"Unknown region code: {region_code}")

    iso3 = _country_iso3(region_code)
    gaul_code = GAUL_COUNTRY_CODES[iso3]

    gaul = ee.FeatureCollection(GAUL_ASSET)
    filtered = gaul.filter(ee.Filter.eq("ADM0_CODE", gaul_code)).filter(
        ee.Filter.eq("ADM1_NAME", region_name)
    )

    # Fallback: case-insensitive partial match
    count = filtered.size().getInfo()
    if count == 0:
        # Try matching with startsWith
        all_admin1 = (
            gaul.filter(ee.Filter.eq("ADM0_CODE", gaul_code))
            .aggregate_array("ADM1_NAME")
            .getInfo()
        )
        for name in all_admin1:
            if region_name.lower() in name.lower() or name.lower() in region_name.lower():
                filtered = gaul.filter(ee.Filter.eq("ADM0_CODE", gaul_code)).filter(
                    ee.Filter.eq("ADM1_NAME", name)
                )
                break

    geom = filtered.geometry()
    _geometry_cache[region_code] = geom
    return geom


# =====================================================================
#  Low-level extraction helpers
# =====================================================================

def _retry_with_backoff(func, *args, **kwargs):
    """Call *func* with exponential backoff on GEE rate-limit errors."""
    delay = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = any(
                tok in exc_str
                for tok in ("too many requests", "rate limit", "429", "quota")
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                logger.warning(
                    "GEE rate limit (attempt %d/%d). Retrying in %.1fs ...",
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise


def _reduce_region_monthly(
    collection_id: str,
    band: str,
    region_code: str,
    start: str,
    end: str,
    reducer_name: str = "mean",
    scale: int = 5000,
) -> pd.DataFrame:
    """Reduce an ImageCollection to monthly region-level means.

    Parameters
    ----------
    collection_id : str
        GEE asset ID (e.g. ``"UCSB-CHG/daily/CHIRPS/final"``).
    band : str
        Band name to extract.
    region_code : str
        Admin-1 region code.
    start, end : str
        ISO date bounds.
    reducer_name : str
        ``"mean"`` or ``"sum"``.
    scale : int
        Spatial resolution in metres for the reduction.

    Returns
    -------
    pd.DataFrame
        Columns ``[region_code, date, value]``.
    """
    _ensure_ee()

    geom = get_region_geometry(region_code)

    # Build monthly composites
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    months = pd.date_range(start_dt, end_dt, freq="MS")

    rows: list[dict] = []

    for month_start in months:
        month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        m_start_str = month_start.strftime("%Y-%m-%d")
        m_end_str = month_end.strftime("%Y-%m-%d")

        col = (
            ee.ImageCollection(collection_id)
            .filterDate(m_start_str, m_end_str)
            .select(band)
        )

        if reducer_name == "sum":
            image = col.sum()
        else:
            image = col.mean()

        def _extract():
            result = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=scale,
                maxPixels=1e9,
            ).getInfo()
            return result

        try:
            result = _retry_with_backoff(_extract)
        except Exception as exc:
            logger.warning(
                "Failed to extract %s/%s for %s %s: %s",
                collection_id,
                band,
                region_code,
                m_start_str,
                exc,
            )
            result = {}

        value = result.get(band) if result else None
        rows.append(
            {"region_code": region_code, "date": month_start, "value": value}
        )

    return pd.DataFrame(rows)


def _batch_extract(
    variable_name: str,
    collection_id: str,
    bands: List[str],
    region_codes: Sequence[str],
    start: str,
    end: str,
    reducer_name: str = "mean",
    scale: int = 5000,
    value_transforms: Optional[Dict[str, callable]] = None,
) -> pd.DataFrame:
    """Batch extraction across regions with caching.

    Processes regions in chunks of :data:`BATCH_SIZE`, checking the parquet
    cache for each region before hitting GEE.

    Parameters
    ----------
    variable_name : str
        Cache variable prefix (e.g. ``"chirps"``).
    collection_id : str
        GEE asset ID.
    bands : list of str
        Band names to extract.
    region_codes : sequence of str
        Region codes.
    start, end : str
        Date bounds.
    reducer_name : str
        ``"mean"`` or ``"sum"``.
    scale : int
        Pixel resolution (m).
    value_transforms : dict, optional
        ``{band: callable}`` applied to the raw value column after
        extraction (e.g. MODIS scale factors).

    Returns
    -------
    pd.DataFrame
        Columns ``[region_code, date, <band_1>, ...]``.
    """
    all_frames: list[pd.DataFrame] = []

    for i, rc in enumerate(region_codes):
        cache_key = make_cache_key(variable_name, rc, start, end)
        if cache_exists(cache_key):
            logger.debug("Cache hit: %s", cache_key)
            all_frames.append(load_from_cache(cache_key))
            continue

        logger.info(
            "Extracting %s for %s (%d/%d) ...",
            variable_name,
            rc,
            i + 1,
            len(region_codes),
        )

        band_frames: list[pd.DataFrame] = []
        for band in bands:
            bdf = _reduce_region_monthly(
                collection_id, band, rc, start, end, reducer_name, scale
            )
            bdf = bdf.rename(columns={"value": band})
            band_frames.append(bdf)

        if band_frames:
            merged = band_frames[0]
            for bdf in band_frames[1:]:
                merged = merged.merge(bdf, on=["region_code", "date"], how="outer")
        else:
            merged = pd.DataFrame(columns=["region_code", "date"])

        # Apply transforms (e.g. MODIS scale factors)
        if value_transforms:
            for col, func in value_transforms.items():
                if col in merged.columns:
                    merged[col] = merged[col].apply(
                        lambda v, f=func: f(v) if v is not None else None
                    )

        save_to_cache(cache_key, merged)
        all_frames.append(merged)

        # Pause between batches to avoid rate limits
        if (i + 1) % BATCH_SIZE == 0 and i + 1 < len(region_codes):
            logger.info("Batch pause (processed %d regions)...", i + 1)
            time.sleep(2)

    if not all_frames:
        return pd.DataFrame(columns=["region_code", "date"])

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["region_code", "date"]).reset_index(drop=True)


# =====================================================================
#  Public extraction functions
# =====================================================================

def get_monthly_precipitation(
    region_codes: Sequence[str],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Monthly precipitation totals from CHIRPS daily rainfall.

    Parameters
    ----------
    region_codes : sequence of str
        Admin-1 region codes (e.g. ``["KE007", "SO014"]``).
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, precipitation]``.
        Precipitation is in mm/month.
    """
    df = _batch_extract(
        variable_name="chirps",
        collection_id=GEE_ASSETS["chirps"],
        bands=["precipitation"],
        region_codes=region_codes,
        start=start,
        end=end,
        reducer_name="sum",
        scale=5000,
    )
    return df


def get_monthly_ndvi(
    region_codes: Sequence[str],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Monthly NDVI and EVI from MODIS MOD13A2 (16-day composites).

    MODIS stores NDVI/EVI as scaled integers; this function converts
    them to the standard [-1, 1] range by multiplying by 0.0001.

    Parameters
    ----------
    region_codes : sequence of str
        Admin-1 region codes.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, NDVI, EVI]``.
    """
    scale_factor = 0.0001
    df = _batch_extract(
        variable_name="ndvi",
        collection_id=GEE_ASSETS["ndvi"],
        bands=[MODIS_NDVI_BAND, MODIS_EVI_BAND],
        region_codes=region_codes,
        start=start,
        end=end,
        reducer_name="mean",
        scale=1000,
        value_transforms={
            MODIS_NDVI_BAND: lambda v: v * scale_factor if v is not None else None,
            MODIS_EVI_BAND: lambda v: v * scale_factor if v is not None else None,
        },
    )
    return df


def get_monthly_lst(
    region_codes: Sequence[str],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Monthly land surface temperature from MODIS MOD11A2 (8-day).

    Raw values are in Kelvin * 0.02; this converts to degrees Celsius.

    Parameters
    ----------
    region_codes : sequence of str
        Admin-1 region codes.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, LST_Day_1km, LST_Night_1km]``
        in degrees Celsius.
    """

    def _lst_to_celsius(v):
        if v is None:
            return None
        return v * 0.02 - 273.15

    df = _batch_extract(
        variable_name="lst",
        collection_id=GEE_ASSETS["lst"],
        bands=[MODIS_LST_DAY_BAND, MODIS_LST_NIGHT_BAND],
        region_codes=region_codes,
        start=start,
        end=end,
        reducer_name="mean",
        scale=1000,
        value_transforms={
            MODIS_LST_DAY_BAND: _lst_to_celsius,
            MODIS_LST_NIGHT_BAND: _lst_to_celsius,
        },
    )
    return df


def get_monthly_soil_moisture(
    region_codes: Sequence[str],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Monthly surface soil moisture from NASA SMAP SPL4SMGP.

    Parameters
    ----------
    region_codes : sequence of str
        Admin-1 region codes.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, sm_surface]``.
        Soil moisture is in m^3/m^3.
    """
    df = _batch_extract(
        variable_name="smap",
        collection_id=GEE_ASSETS["smap"],
        bands=["sm_surface"],
        region_codes=region_codes,
        start=start,
        end=end,
        reducer_name="mean",
        scale=9000,
    )
    return df


def get_monthly_era5(
    region_codes: Sequence[str],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.DataFrame:
    """Monthly ERA5-Land variables: temperature, humidity, wind, radiation, evaporation.

    Computes Tmax, Tmin, and Tmean from 2-m temperature, plus dewpoint
    (proxy for humidity), wind components, solar radiation, and
    evaporation.

    Parameters
    ----------
    region_codes : sequence of str
        Admin-1 region codes.
    start, end : str
        ISO date bounds.

    Returns
    -------
    pd.DataFrame
        Columns: ``[region_code, date, temperature_2m, dewpoint_temperature_2m,
        u_component_of_wind_10m, v_component_of_wind_10m,
        surface_solar_radiation_downwards, total_evaporation,
        total_precipitation_era5]``.
    """
    _ensure_ee()

    era5_bands = [
        ERA5_BANDS["temperature_2m"],
        ERA5_BANDS["dewpoint_2m"],
        ERA5_BANDS["u_wind_10m"],
        ERA5_BANDS["v_wind_10m"],
        ERA5_BANDS["surface_solar_radiation"],
        ERA5_BANDS["evaporation"],
    ]

    all_frames: list[pd.DataFrame] = []

    for i, rc in enumerate(region_codes):
        cache_key = make_cache_key("era5", rc, start, end)
        if cache_exists(cache_key):
            logger.debug("Cache hit: %s", cache_key)
            all_frames.append(load_from_cache(cache_key))
            continue

        logger.info("Extracting ERA5-Land for %s (%d/%d) ...", rc, i + 1, len(region_codes))

        geom = get_region_geometry(rc)
        months = pd.date_range(start, end, freq="MS")
        rows: list[dict] = []

        for month_start in months:
            month_end = month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
            m_start_str = month_start.strftime("%Y-%m-%d")
            m_end_str = month_end.strftime("%Y-%m-%d")

            col = (
                ee.ImageCollection(GEE_ASSETS["era5_land"])
                .filterDate(m_start_str, m_end_str)
                .select(era5_bands)
            )

            # Temperature: mean, max, min
            t_mean_img = col.select(ERA5_BANDS["temperature_2m"]).mean()
            t_max_img = col.select(ERA5_BANDS["temperature_2m"]).max()
            t_min_img = col.select(ERA5_BANDS["temperature_2m"]).min()

            # Other bands: monthly means (radiation/evap summed, but
            # GEE ERA5 stores cumulative values per hour so mean is
            # appropriate for a monthly aggregate)
            other_img = col.mean()

            combined = (
                t_mean_img.rename([ERA5_BANDS["temperature_2m"]])
                .addBands(t_max_img.rename(["temperature_2m_max"]))
                .addBands(t_min_img.rename(["temperature_2m_min"]))
                .addBands(other_img.select(era5_bands[1:]))  # non-temp bands
            )

            all_bands_list = (
                [ERA5_BANDS["temperature_2m"], "temperature_2m_max", "temperature_2m_min"]
                + era5_bands[1:]
            )

            def _extract():
                return combined.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geom,
                    scale=11000,
                    maxPixels=1e9,
                ).getInfo()

            try:
                result = _retry_with_backoff(_extract)
            except Exception as exc:
                logger.warning("ERA5 extraction failed for %s %s: %s", rc, m_start_str, exc)
                result = {}

            row = {"region_code": rc, "date": month_start}
            if result:
                for band_name in all_bands_list:
                    val = result.get(band_name)
                    # Convert temperatures from Kelvin to Celsius
                    if val is not None and "temperature" in band_name:
                        val = val - 273.15
                    row[band_name] = val
            rows.append(row)

        rc_df = pd.DataFrame(rows)
        save_to_cache(cache_key, rc_df)
        all_frames.append(rc_df)

        if (i + 1) % BATCH_SIZE == 0 and i + 1 < len(region_codes):
            logger.info("Batch pause (processed %d regions)...", i + 1)
            time.sleep(2)

    if not all_frames:
        return pd.DataFrame(columns=["region_code", "date"])

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["region_code", "date"]).reset_index(drop=True)
