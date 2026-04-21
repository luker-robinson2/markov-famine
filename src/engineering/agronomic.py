"""Agronomic index computation for the food security prediction system.

Implements standard drought, vegetation, and evapotranspiration indices used
in agricultural monitoring and early warning systems.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import (
    DROUGHT_INDEX_WINDOWS,
    GDD_BASE_TEMPS,
    PM_ALBEDO,
    PM_PSYCHROMETRIC,
    PM_STEFAN_BOLTZMANN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drought indices
# ---------------------------------------------------------------------------


def compute_spei(
    precip: pd.Series,
    et0: pd.Series,
    window: int,
) -> pd.Series:
    r"""Compute the Standardized Precipitation-Evapotranspiration Index (SPEI).

    The SPEI extends the SPI by accounting for evaporative demand:

    .. math::

        D_i = P_i - \mathrm{ET0}_i

    The water balance *D* is accumulated over *window* months, the resulting
    series is fit to a log-logistic distribution, and the CDF values are
    transformed to standard normal quantiles.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation totals (mm).
    et0 : pd.Series
        Monthly reference evapotranspiration (mm).
    window : int
        Accumulation window in months (typically 1, 3, 6, or 12).

    Returns
    -------
    pd.Series
        SPEI values (unitless, standard normal scale).

    Notes
    -----
    * The log-logistic distribution is fit via L-moments (approximated here
      using ``scipy.stats.fisk``).
    * The first ``window - 1`` values are ``NaN`` due to the rolling sum.
    """
    if window not in DROUGHT_INDEX_WINDOWS:
        warnings.warn(
            f"Window {window} not in standard DROUGHT_INDEX_WINDOWS "
            f"{DROUGHT_INDEX_WINDOWS}; proceeding anyway."
        )

    water_balance = precip - et0
    accumulated = water_balance.rolling(window=window, min_periods=window).sum()

    spei = pd.Series(np.nan, index=accumulated.index)

    valid = accumulated.dropna()
    if len(valid) < 30:
        logger.warning(
            "[agronomic.compute_spei] Only %d valid observations for window=%d; "
            "returning NaN series.",
            len(valid),
            window,
        )
        return spei

    # Shift so all values are positive (log-logistic requires > 0)
    shift = 0.0
    if valid.min() <= 0:
        shift = abs(valid.min()) + 1.0

    shifted = valid + shift

    # Fit log-logistic (scipy.stats.fisk)
    try:
        c, loc, scale = sp_stats.fisk.fit(shifted, floc=0)
    except Exception:
        logger.warning(
            "[agronomic.compute_spei] Log-logistic fit failed; "
            "falling back to normal standardization."
        )
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0:
            return spei
        spei.loc[valid.index] = (valid - mu) / sigma
        return spei

    cdf_vals = sp_stats.fisk.cdf(accumulated.loc[valid.index] + shift, c, loc=loc, scale=scale)
    # Clip to avoid infinite quantiles
    cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)
    spei.loc[valid.index] = sp_stats.norm.ppf(cdf_vals)

    return spei


def compute_spi(
    precip: pd.Series,
    window: int,
) -> pd.Series:
    r"""Compute the Standardized Precipitation Index (SPI).

    Accumulated precipitation over *window* months is fit to a gamma
    distribution, and CDF values are transformed to standard normal quantiles:

    .. math::

        \mathrm{SPI} = \Phi^{-1}\!\bigl[G(\text{accum})\bigr]

    where :math:`G` is the fitted gamma CDF and :math:`\Phi^{-1}` is the
    standard normal inverse CDF.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation totals (mm).
    window : int
        Accumulation window in months.

    Returns
    -------
    pd.Series
        SPI values (unitless, standard normal scale).
    """
    accumulated = precip.rolling(window=window, min_periods=window).sum()

    spi = pd.Series(np.nan, index=accumulated.index)

    valid = accumulated.dropna()
    positive = valid[valid > 0]

    if len(positive) < 30:
        logger.warning(
            "[agronomic.compute_spi] Only %d positive observations for "
            "window=%d; returning NaN series.",
            len(positive),
            window,
        )
        return spi

    # Proportion of zeros (mixed distribution for dry climates)
    q_zero = (valid == 0).sum() / len(valid)

    try:
        alpha, loc, beta = sp_stats.gamma.fit(positive, floc=0)
    except Exception:
        logger.warning(
            "[agronomic.compute_spi] Gamma fit failed; falling back to "
            "normal standardization."
        )
        mu, sigma = valid.mean(), valid.std()
        if sigma == 0:
            return spi
        spi.loc[valid.index] = (valid - mu) / sigma
        return spi

    # CDF accounting for zero mass
    cdf_vals = q_zero + (1 - q_zero) * sp_stats.gamma.cdf(
        accumulated.loc[valid.index], alpha, loc=loc, scale=beta
    )
    cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)
    spi.loc[valid.index] = sp_stats.norm.ppf(cdf_vals)

    return spi


# ---------------------------------------------------------------------------
# Growing Degree Days
# ---------------------------------------------------------------------------


def compute_gdd(
    tmax: pd.Series,
    tmin: pd.Series,
    tbase: float = 10.0,
) -> pd.Series:
    r"""Compute cumulative Growing Degree Days (GDD).

    .. math::

        \mathrm{GDD}_i = \max\!\left(0,\;
        \frac{T_{\max,i} + T_{\min,i}}{2} - T_{\mathrm{base}}\right)

    The returned series is the **cumulative** GDD within each calendar year
    (reset each January).

    Parameters
    ----------
    tmax, tmin : pd.Series
        Monthly mean of daily max/min temperatures (°C).
    tbase : float, default 10.0
        Base temperature (°C).  See ``config.GDD_BASE_TEMPS`` for
        crop-specific values.

    Returns
    -------
    pd.Series
        Cumulative GDD (°C·days equivalent at monthly resolution).
    """
    tavg = (tmax + tmin) / 2.0
    daily_gdd = np.maximum(0.0, tavg - tbase)

    # Cumulative within calendar year
    if hasattr(tmax.index, "year"):
        years = tmax.index.year
    elif hasattr(tmax.index, "to_timestamp"):
        years = tmax.index.to_timestamp().year
    else:
        # Fallback: no year grouping
        return daily_gdd.cumsum()

    cumulative = daily_gdd.groupby(years).cumsum()
    return cumulative


# ---------------------------------------------------------------------------
# Crop Water Stress Index
# ---------------------------------------------------------------------------


def compute_cwsi(
    lst_day: pd.Series,
    et0: pd.Series,
) -> pd.Series:
    r"""Compute an empirical Crop Water Stress Index (CWSI).

    Uses MODIS Land Surface Temperature (LST) as a proxy for canopy
    temperature and ET0 as a reference:

    .. math::

        \mathrm{CWSI} = \frac{T_{\mathrm{LST}} - T_{\mathrm{LST,min}}}
        {T_{\mathrm{LST,max}} - T_{\mathrm{LST,min}}}
        \times \left(1 - \frac{\mathrm{ET0}}{\mathrm{ET0_{max}}}\right)

    The index is clipped to [0, 1]:
    * 0 = no water stress (well-watered)
    * 1 = maximum water stress

    Parameters
    ----------
    lst_day : pd.Series
        MODIS daytime LST (°C or K — the formula is scale-invariant).
    et0 : pd.Series
        Reference evapotranspiration (mm/month).

    Returns
    -------
    pd.Series
        CWSI values clipped to [0, 1].
    """
    lst_min = lst_day.min()
    lst_max = lst_day.max()
    et0_max = et0.max()

    if lst_max == lst_min or et0_max == 0:
        return pd.Series(np.nan, index=lst_day.index)

    lst_norm = (lst_day - lst_min) / (lst_max - lst_min)
    et0_factor = 1.0 - (et0 / et0_max)

    cwsi = lst_norm * et0_factor
    return cwsi.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Reference Evapotranspiration
# ---------------------------------------------------------------------------


def compute_et0_penman_monteith(
    temp_mean: pd.Series,
    temp_max: pd.Series,
    temp_min: pd.Series,
    humidity: pd.Series,
    wind_speed: pd.Series,
    solar_rad: pd.Series,
    elevation: float = 1000.0,
) -> pd.Series:
    r"""FAO Penman-Monteith reference evapotranspiration (ET₀).

    Implements the FAO-56 standard equation:

    .. math::

        \mathrm{ET}_0 = \frac{
            0.408\,\Delta\,(R_n - G)
            + \gamma \frac{900}{T + 273}\,u_2\,(e_s - e_a)
        }{
            \Delta + \gamma\,(1 + 0.34\,u_2)
        }

    where:

    * :math:`\Delta` — slope of saturation vapour pressure curve (kPa/°C)
    * :math:`R_n` — net radiation (MJ m⁻² day⁻¹)
    * :math:`G` — soil heat flux density (≈ 0 at monthly scale)
    * :math:`\gamma` — psychrometric constant (kPa/°C)
    * :math:`u_2` — wind speed at 2 m (m/s)
    * :math:`e_s - e_a` — vapour pressure deficit (kPa)

    Parameters
    ----------
    temp_mean : pd.Series
        Mean temperature (°C).
    temp_max, temp_min : pd.Series
        Daily max/min temperature (°C).
    humidity : pd.Series
        Relative humidity (%).
    wind_speed : pd.Series
        Wind speed at 2 m height (m/s).
    solar_rad : pd.Series
        Incoming shortwave solar radiation (MJ m⁻² day⁻¹).
    elevation : float, default 1000
        Station elevation (m) — used for atmospheric pressure.

    Returns
    -------
    pd.Series
        ET₀ (mm/day).
    """
    # Atmospheric pressure (kPa)
    P = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26

    # Psychrometric constant γ (kPa/°C)
    gamma = PM_PSYCHROMETRIC * P

    # Saturation vapour pressure (kPa) via Tetens formula
    def _esat(T: pd.Series) -> pd.Series:
        return 0.6108 * np.exp(17.27 * T / (T + 237.3))

    e_s = (_esat(temp_max) + _esat(temp_min)) / 2.0
    e_a = e_s * (humidity / 100.0)

    # Slope of saturation vapour pressure curve
    delta = (4098 * _esat(temp_mean)) / (temp_mean + 237.3) ** 2

    # Net radiation (simplified)
    # Net shortwave
    Rns = (1 - PM_ALBEDO) * solar_rad

    # Net longwave (FAO-56 simplified formula)
    sigma = PM_STEFAN_BOLTZMANN  # MJ m-2 day-1 K-4
    Tmax_K4 = (temp_max + 273.16) ** 4
    Tmin_K4 = (temp_min + 273.16) ** 4

    # Clear-sky radiation approximation (Rso)
    Rso = (0.75 + 2e-5 * elevation) * solar_rad.clip(lower=0.1)
    Rs_over_Rso = (solar_rad / Rso).clip(0.0, 1.0)

    Rnl = sigma * ((Tmax_K4 + Tmin_K4) / 2.0) * (0.34 - 0.14 * np.sqrt(e_a)) * (
        1.35 * Rs_over_Rso - 0.35
    )

    Rn = Rns - Rnl

    # Soil heat flux G ≈ 0 for monthly time steps
    G = 0.0

    # FAO-56 Penman-Monteith
    numerator = 0.408 * delta * (Rn - G) + gamma * (900 / (temp_mean + 273)) * wind_speed * (e_s - e_a)
    denominator = delta + gamma * (1 + 0.34 * wind_speed)

    et0 = numerator / denominator
    return et0.clip(lower=0.0)


def compute_et0_hargreaves(
    temp_mean: pd.Series,
    temp_max: pd.Series,
    temp_min: pd.Series,
    latitude: float,
) -> pd.Series:
    r"""Hargreaves-Samani reference evapotranspiration.

    A simpler ET₀ estimate requiring only temperature and latitude:

    .. math::

        \mathrm{ET}_0 = 0.0023\;(T_{\mathrm{mean}} + 17.8)\;
        (T_{\max} - T_{\min})^{0.5}\;R_a

    where :math:`R_a` is extra-terrestrial radiation estimated from latitude
    and day of year.

    Parameters
    ----------
    temp_mean, temp_max, temp_min : pd.Series
        Monthly temperatures (°C).
    latitude : float
        Station latitude in degrees.

    Returns
    -------
    pd.Series
        ET₀ (mm/day).
    """
    # Extra-terrestrial radiation approximation (Ra, MJ m-2 day-1)
    # Use middle of month for day-of-year
    if hasattr(temp_mean.index, "to_timestamp"):
        doy = temp_mean.index.to_timestamp().dayofyear
    elif hasattr(temp_mean.index, "dayofyear"):
        doy = temp_mean.index.dayofyear
    else:
        # Fallback: assume uniformly spaced months
        doy = pd.Series(
            [15 + 30 * i for i in range(len(temp_mean))],
            index=temp_mean.index,
        )

    lat_rad = np.radians(latitude)
    # Solar declination
    decl = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    # Sunset hour angle
    ws = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    # Extra-terrestrial radiation (MJ m-2 day-1)
    Gsc = 0.0820  # solar constant
    Ra = (
        (24 * 60 / np.pi)
        * Gsc
        * dr
        * (ws * np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.sin(ws))
    )

    # Hargreaves equation
    temp_range = (temp_max - temp_min).clip(lower=0)
    et0 = 0.0023 * (temp_mean + 17.8) * np.sqrt(temp_range) * Ra

    return et0.clip(lower=0.0)


# ---------------------------------------------------------------------------
# Vegetation indices
# ---------------------------------------------------------------------------


def compute_vci(
    ndvi: pd.Series,
    ndvi_min: float,
    ndvi_max: float,
) -> pd.Series:
    r"""Compute the Vegetation Condition Index (VCI).

    .. math::

        \mathrm{VCI} = \frac{\mathrm{NDVI} - \mathrm{NDVI}_{\min}}
        {\mathrm{NDVI}_{\max} - \mathrm{NDVI}_{\min}} \times 100

    VCI expresses the current NDVI relative to the historical range:
    * 0 = worst historical vegetation condition
    * 100 = best historical vegetation condition

    Parameters
    ----------
    ndvi : pd.Series
        Current NDVI values.
    ndvi_min, ndvi_max : float
        Historical minimum and maximum NDVI for the corresponding month /
        region.

    Returns
    -------
    pd.Series
        VCI (0–100 scale).
    """
    denom = ndvi_max - ndvi_min
    if denom == 0:
        return pd.Series(np.nan, index=ndvi.index)

    vci = ((ndvi - ndvi_min) / denom) * 100.0
    return vci.clip(0.0, 100.0)


def compute_vhi(
    vci: pd.Series,
    tci: pd.Series,
) -> pd.Series:
    r"""Compute the Vegetation Health Index (VHI).

    .. math::

        \mathrm{VHI} = 0.5 \times \mathrm{VCI} + 0.5 \times \mathrm{TCI}

    Parameters
    ----------
    vci : pd.Series
        Vegetation Condition Index (0–100).
    tci : pd.Series
        Temperature Condition Index (0–100), computed analogously to VCI
        but using LST.

    Returns
    -------
    pd.Series
        VHI (0–100 scale).

    Notes
    -----
    VHI < 40 is commonly used as a drought indicator in agricultural
    monitoring (Kogan, 1995).
    """
    vhi = 0.5 * vci + 0.5 * tci
    return vhi.clip(0.0, 100.0)
