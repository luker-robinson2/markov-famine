"""Enhanced feature engineering for food security prediction.

Computes evidence-based features that the literature shows are most
predictive of IPC phase transitions:

- **VCI** (Vegetation Condition Index) — outperforms raw NDVI (Klisch 2016)
- **SPI-3** (Standardized Precipitation Index) — drought accumulation
- **Cumulative deficits** — rolling negative anomaly sums (Funk 2019)
- **Rate-of-change** — velocity of decline in key indicators
- **Consecutive stress** — duration of drought conditions
- **Interaction features** — compound stress indicators

All fitted statistics (VCI bounds, SPI gamma params) use ONLY training
data to prevent leakage.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.config import TRAIN_END

logger = logging.getLogger(__name__)


# =====================================================================
#  VCI — Vegetation Condition Index
# =====================================================================

def compute_vci_panel(
    panel: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """Add VCI column computed from training-period NDVI min/max.

    VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) × 100

    The historical min/max are computed per (region_code, month_of_year)
    using ONLY observations up to *train_end* to prevent leakage.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``ndvi_monthly``, ``region_code``, ``date``.
    train_end : str
        Cutoff date for computing historical statistics.

    Returns
    -------
    pd.DataFrame
        Input panel with ``vci`` column added.
    """
    df = panel.copy()
    df["_month"] = df["date"].dt.month
    train_mask = df["date"] <= train_end

    # Compute per-(region, month) NDVI min/max from training data only
    stats = (
        df.loc[train_mask]
        .groupby(["region_code", "_month"])["ndvi_monthly"]
        .agg(ndvi_min="min", ndvi_max="max")
        .reset_index()
    )

    df = df.merge(stats, on=["region_code", "_month"], how="left")
    denom = (df["ndvi_max"] - df["ndvi_min"]).replace(0, np.nan)
    df["vci"] = ((df["ndvi_monthly"] - df["ndvi_min"]) / denom * 100).clip(0, 100)
    df.drop(columns=["_month", "ndvi_min", "ndvi_max"], inplace=True)

    n_valid = df["vci"].notna().sum()
    logger.info("VCI computed: %d/%d valid (%.1f%%)", n_valid, len(df), n_valid / len(df) * 100)
    return df


# =====================================================================
#  SPI — Standardized Precipitation Index (3-month)
# =====================================================================

def compute_spi_panel(
    panel: pd.DataFrame,
    window: int = 3,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """Add SPI column using gamma distribution fitted on training data.

    Precipitation is accumulated over *window* months, then the gamma
    CDF is applied and transformed to a standard normal quantile.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain ``precip_monthly``, ``region_code``, ``date``.
    window : int
        Accumulation window in months.
    train_end : str
        Cutoff for gamma distribution fitting.

    Returns
    -------
    pd.DataFrame
        Input panel with ``spi_{window}mo`` column added.
    """
    col_name = f"spi_{window}mo"
    df = panel.copy()
    df = df.sort_values(["region_code", "date"])

    # Rolling sum of precipitation over window months
    df["_precip_acc"] = df.groupby("region_code")["precip_monthly"].transform(
        lambda x: x.rolling(window, min_periods=window).sum()
    )

    train_mask = df["date"] <= train_end
    result = pd.Series(np.nan, index=df.index, name=col_name)

    for rc in df["region_code"].unique():
        rc_mask = df["region_code"] == rc
        train_vals = df.loc[rc_mask & train_mask, "_precip_acc"].dropna()
        train_vals = train_vals[train_vals > 0]  # gamma requires positive values

        if len(train_vals) < 20:
            logger.warning("SPI: region %s has only %d training values, skipping", rc, len(train_vals))
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                shape, loc, scale = sp_stats.gamma.fit(train_vals, floc=0)
            except Exception:
                logger.warning("SPI gamma fit failed for %s", rc)
                continue

        all_vals = df.loc[rc_mask, "_precip_acc"]
        valid = all_vals.notna() & (all_vals > 0)
        cdf_vals = sp_stats.gamma.cdf(all_vals[valid], shape, loc=loc, scale=scale)
        # Clip to avoid infinite quantiles
        cdf_vals = np.clip(cdf_vals, 1e-6, 1 - 1e-6)
        spi_vals = sp_stats.norm.ppf(cdf_vals)
        result.loc[valid[valid].index] = spi_vals

    df[col_name] = result.values
    df.drop(columns=["_precip_acc"], inplace=True)

    n_valid = df[col_name].notna().sum()
    logger.info("SPI-%dmo computed: %d/%d valid", window, n_valid, len(df))
    return df


# =====================================================================
#  Cumulative deficits
# =====================================================================

def compute_cumulative_deficits(panel: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 3-month cumulative negative anomaly features.

    For each anomaly column, computes the rolling 3-month sum of
    min(0, anomaly) — capturing accumulated drought/stress.

    Returns columns: ``precip_deficit_3mo``, ``ndvi_deficit_3mo``,
    ``sm_deficit_3mo``.
    """
    df = panel.copy()
    df = df.sort_values(["region_code", "date"])

    for raw_col, out_col in [
        ("precip_anomaly", "precip_deficit_3mo"),
        ("ndvi_anomaly", "ndvi_deficit_3mo"),
        ("soil_moisture_anomaly", "sm_deficit_3mo"),
    ]:
        if raw_col not in df.columns:
            logger.warning("Column %s not found, skipping %s", raw_col, out_col)
            continue
        # Only accumulate negative anomalies (stress)
        negative = df[raw_col].clip(upper=0)
        df[out_col] = df.groupby("region_code")[raw_col].transform(
            lambda x: x.clip(upper=0).rolling(3, min_periods=1).sum()
        )

    logger.info("Cumulative deficit features computed")
    return df


# =====================================================================
#  Rate of change
# =====================================================================

def compute_rate_of_change(panel: pd.DataFrame) -> pd.DataFrame:
    """Add month-over-month and 3-month rate-of-change features.

    Returns columns: ``ndvi_roc_1mo``, ``ndvi_roc_3mo``,
    ``precip_roc_3mo``, ``sm_roc_1mo``, ``tot_velocity``.
    """
    df = panel.copy()
    df = df.sort_values(["region_code", "date"])

    for raw_col, lag, out_col in [
        ("ndvi_monthly", 1, "ndvi_roc_1mo"),
        ("ndvi_monthly", 3, "ndvi_roc_3mo"),
        ("precip_monthly", 3, "precip_roc_3mo"),
        ("soil_moisture", 1, "sm_roc_1mo"),
    ]:
        if raw_col not in df.columns:
            continue
        df[out_col] = df.groupby("region_code")[raw_col].transform(
            lambda x: x.diff(lag)
        )

    # Terms of Trade velocity (% change where available)
    if "tot_livestock_grain" in df.columns:
        df["tot_velocity"] = df.groupby("region_code")["tot_livestock_grain"].transform(
            lambda x: x.pct_change()
        )
    else:
        df["tot_velocity"] = np.nan

    logger.info("Rate-of-change features computed")
    return df


# =====================================================================
#  Consecutive stress duration
# =====================================================================

def compute_consecutive_stress(panel: pd.DataFrame) -> pd.DataFrame:
    """Add features counting consecutive months of stress conditions.

    Returns columns: ``dry_months_count``, ``vci_below_35_months``.
    """
    df = panel.copy()
    df = df.sort_values(["region_code", "date"])

    def _consecutive_count(series: pd.Series, condition: pd.Series) -> pd.Series:
        """Count consecutive True values, resetting on False."""
        groups = (~condition).cumsum()
        return condition.groupby(groups).cumcount() + condition.astype(int)

    # Consecutive dry months (precip anomaly < -0.5)
    if "precip_anomaly" in df.columns:
        counts = []
        for rc in df["region_code"].unique():
            mask = df["region_code"] == rc
            sub = df.loc[mask, "precip_anomaly"]
            is_dry = sub < -0.5
            counts.append(_consecutive_count(sub, is_dry))
        df["dry_months_count"] = pd.concat(counts)

    # Consecutive months with VCI below drought threshold
    if "vci" in df.columns:
        counts = []
        for rc in df["region_code"].unique():
            mask = df["region_code"] == rc
            sub = df.loc[mask, "vci"]
            is_stressed = sub < 35
            counts.append(_consecutive_count(sub, is_stressed))
        df["vci_below_35_months"] = pd.concat(counts)

    logger.info("Consecutive stress features computed")
    return df


# =====================================================================
#  Interaction features
# =====================================================================

def compute_interaction_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add compound stress interaction features.

    Returns columns: ``drought_x_price``, ``ndvi_x_season_gu``.
    """
    df = panel.copy()

    # Drought × price stress (compound shock)
    if "precip_anomaly" in df.columns and "tot_anomaly" in df.columns:
        df["drought_x_price"] = df["precip_anomaly"] * df["tot_anomaly"].fillna(0)
    else:
        df["drought_x_price"] = 0.0

    # NDVI stress during critical Gu (long rains) season
    if "ndvi_anomaly" in df.columns and "season_gu" in df.columns:
        df["ndvi_x_season_gu"] = df["ndvi_anomaly"] * df["season_gu"]
    else:
        df["ndvi_x_season_gu"] = 0.0

    logger.info("Interaction features computed")
    return df


# =====================================================================
#  Master function
# =====================================================================

def build_enhanced_panel(
    panel: pd.DataFrame,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """Compute all enhanced features and add to the panel.

    Fitted statistics (VCI bounds, SPI gamma params) use ONLY data up
    to *train_end* to prevent leakage into validation/test sets.

    Parameters
    ----------
    panel : pd.DataFrame
        Base panel with climate, market, and IPC columns.
    train_end : str
        Last date of training period (inclusive).

    Returns
    -------
    pd.DataFrame
        Panel with all enhanced feature columns added.
    """
    logger.info("Building enhanced feature panel (train_end=%s) ...", train_end)

    df = compute_vci_panel(panel, train_end=train_end)
    df = compute_spi_panel(df, window=3, train_end=train_end)
    df = compute_cumulative_deficits(df)
    df = compute_rate_of_change(df)
    df = compute_consecutive_stress(df)
    df = compute_interaction_features(df)

    new_cols = [
        "vci", "spi_3mo",
        "precip_deficit_3mo", "ndvi_deficit_3mo", "sm_deficit_3mo",
        "ndvi_roc_1mo", "ndvi_roc_3mo", "precip_roc_3mo", "sm_roc_1mo", "tot_velocity",
        "dry_months_count", "vci_below_35_months",
        "drought_x_price", "ndvi_x_season_gu",
    ]
    present = [c for c in new_cols if c in df.columns]
    logger.info(
        "Enhanced panel: %d new features added (%d/%d rows with VCI, %d/%d with SPI-3)",
        len(present),
        df["vci"].notna().sum(), len(df),
        df["spi_3mo"].notna().sum() if "spi_3mo" in df.columns else 0, len(df),
    )
    return df
