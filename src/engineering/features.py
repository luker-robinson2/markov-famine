"""Full covariate matrix assembly for the food security prediction system.

Merges climate, agronomic, teleconnection, market, IPC, and static features
into a unified panel indexed by ``(region_code, year_month)`` with 40+
columns.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_GROUPS, get_season
from src.engineering.temporal import encode_cyclical_month, encode_season

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_panel(
    climate_df: pd.DataFrame,
    agronomic_df: pd.DataFrame,
    teleconnection_df: pd.DataFrame,
    market_df: pd.DataFrame,
    ipc_df: pd.DataFrame,
    static_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble all features into a unified panel indexed by (region_code, year_month).

    Parameters
    ----------
    climate_df : pd.DataFrame
        Monthly climate features (precipitation, NDVI, soil moisture,
        temperature, etc.).  Must have ``region_code`` and ``year_month``.
    agronomic_df : pd.DataFrame
        Drought / vegetation indices (SPEI, SPI, GDD, VCI, VHI, CWSI, ET0).
    teleconnection_df : pd.DataFrame
        Climate indices (IOD, ONI, MJO).  May lack ``region_code`` if
        indices are global; they will be broadcast to all regions.
    market_df : pd.DataFrame
        Market features (terms of trade, price anomalies).
    ipc_df : pd.DataFrame
        IPC phase observations.  Must contain ``ipc_phase``.
    static_df : pd.DataFrame
        Time-invariant features (livelihood type, population density,
        conflict counts).

    Returns
    -------
    pd.DataFrame
        Unified panel with ``region_code``, ``year_month`` as index
        columns and 40+ feature columns.
    """
    idx = ["region_code", "year_month"]

    # ------------------------------------------------------------------
    # 1. Start with climate as backbone
    # ------------------------------------------------------------------
    panel = climate_df.copy()
    _ensure_columns(panel, idx, "climate_df")

    # ------------------------------------------------------------------
    # 2. Merge agronomic
    # ------------------------------------------------------------------
    if agronomic_df is not None and len(agronomic_df):
        _ensure_columns(agronomic_df, idx, "agronomic_df")
        panel = panel.merge(agronomic_df, on=idx, how="left", suffixes=("", "_agro"))

    # ------------------------------------------------------------------
    # 3. Merge teleconnection (broadcast if no region_code)
    # ------------------------------------------------------------------
    if teleconnection_df is not None and len(teleconnection_df):
        if "region_code" not in teleconnection_df.columns:
            panel = panel.merge(
                teleconnection_df, on="year_month", how="left", suffixes=("", "_tele")
            )
        else:
            panel = panel.merge(
                teleconnection_df, on=idx, how="left", suffixes=("", "_tele")
            )

    # ------------------------------------------------------------------
    # 4. Merge market
    # ------------------------------------------------------------------
    if market_df is not None and len(market_df):
        _ensure_columns(market_df, idx, "market_df")
        panel = panel.merge(market_df, on=idx, how="left", suffixes=("", "_mkt"))

    # ------------------------------------------------------------------
    # 5. Merge IPC + derive lagged-state features
    # ------------------------------------------------------------------
    if ipc_df is not None and len(ipc_df):
        _ensure_columns(ipc_df, idx, "ipc_df")
        panel = panel.merge(ipc_df, on=idx, how="left", suffixes=("", "_ipc"))
        panel = _compute_lagged_state(panel)

    # ------------------------------------------------------------------
    # 6. Merge static (broadcast across time)
    # ------------------------------------------------------------------
    if static_df is not None and len(static_df):
        if "region_code" not in static_df.columns:
            raise KeyError("static_df must have 'region_code' column.")
        static_cols = [c for c in static_df.columns if c != "year_month"]
        panel = panel.merge(
            static_df[static_cols].drop_duplicates(),
            on="region_code",
            how="left",
            suffixes=("", "_static"),
        )

    # ------------------------------------------------------------------
    # 7. Compute derived features
    # ------------------------------------------------------------------
    panel = _compute_anomalies(panel)
    panel = _compute_moving_averages(panel)
    panel = _compute_temporal_encoding(panel)
    panel = _compute_iod_lag(panel)

    return panel


def get_feature_matrix(
    panel: pd.DataFrame,
    feature_groups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Select features by group name from ``config.FEATURE_GROUPS``.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of :func:`build_feature_panel`.
    feature_groups : list[str], optional
        Group names to include (e.g. ``["climate_raw", "agronomic"]``).
        If ``None``, all groups are included.

    Returns
    -------
    pd.DataFrame
        Subset of *panel* with only the requested feature columns (plus
        ``region_code`` and ``year_month`` for indexing).
    """
    if feature_groups is None:
        feature_groups = list(FEATURE_GROUPS.keys())

    selected: list[str] = []
    for group in feature_groups:
        if group not in FEATURE_GROUPS:
            logger.warning(
                "[features.get_feature_matrix] Unknown feature group %r; skipping.",
                group,
            )
            continue
        selected.extend(FEATURE_GROUPS[group])

    # Keep only columns that actually exist in the panel
    available = [c for c in selected if c in panel.columns]
    missing = set(selected) - set(available)
    if missing:
        logger.info(
            "[features.get_feature_matrix] %d requested features not in panel: %s",
            len(missing),
            sorted(missing),
        )

    idx_cols = [c for c in ["region_code", "year_month"] if c in panel.columns]
    return panel[idx_cols + available].copy()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Raise ``KeyError`` if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _compute_anomalies(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute standardized anomalies relative to the long-term monthly mean.

    For each of ``precip_monthly``, ``ndvi_monthly``, ``soil_moisture``,
    ``temp_mean``: anomaly = (x - monthly_mean) / monthly_std.
    """
    # Extract month from year_month
    if "year_month" in panel.columns:
        if hasattr(panel["year_month"].iloc[0], "month"):
            panel["_month"] = panel["year_month"].apply(lambda p: p.month)
        else:
            panel["_month"] = pd.to_datetime(panel["year_month"].astype(str)).dt.month

    target_pairs = [
        ("precip_monthly", "precip_anomaly"),
        ("ndvi_monthly", "ndvi_anomaly"),
        ("soil_moisture", "soil_moisture_anomaly"),
        ("temp_mean", "temp_anomaly"),
    ]

    for raw_col, anom_col in target_pairs:
        if raw_col not in panel.columns:
            continue
        if anom_col in panel.columns:
            # Already computed upstream; skip
            continue

        # Group by region and calendar month for climatology
        group_cols = ["region_code", "_month"]
        stats = panel.groupby(group_cols)[raw_col].agg(["mean", "std"]).reset_index()
        stats.columns = group_cols + ["_clim_mean", "_clim_std"]

        panel = panel.merge(stats, on=group_cols, how="left")
        panel[anom_col] = (panel[raw_col] - panel["_clim_mean"]) / panel["_clim_std"].replace(0, np.nan)
        panel.drop(columns=["_clim_mean", "_clim_std"], inplace=True)

    if "_month" in panel.columns:
        panel.drop(columns=["_month"], inplace=True, errors="ignore")

    return panel


def _compute_moving_averages(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-month simple moving averages for precipitation and soil moisture."""
    panel = panel.sort_values(["region_code", "year_month"]).copy()

    sma_targets = [
        ("precip_monthly", "precip_3mo_sma"),
        ("soil_moisture", "soil_moisture_3mo_sma"),
    ]

    for raw_col, sma_col in sma_targets:
        if raw_col not in panel.columns:
            continue
        if sma_col in panel.columns:
            continue
        panel[sma_col] = (
            panel.groupby("region_code")[raw_col]
            .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )

    return panel


def _compute_lagged_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Derive lagged IPC state features.

    Produces:
    * ``prev_ipc_phase`` — IPC phase from the prior month.
    * ``prev_ipc_duration`` — consecutive months at the same phase.
    * ``phase_trend_3mo`` — direction of change over 3 months (-1 / 0 / +1).
    """
    if "ipc_phase" not in panel.columns:
        return panel

    panel = panel.sort_values(["region_code", "year_month"]).copy()

    # Previous phase
    panel["prev_ipc_phase"] = panel.groupby("region_code")["ipc_phase"].shift(1)

    # Duration at same phase (consecutive months)
    durations = []
    for _, group in panel.groupby("region_code"):
        dur = []
        count = 0
        prev = None
        for phase in group["ipc_phase"]:
            if pd.notna(phase) and phase == prev:
                count += 1
            else:
                count = 1
            dur.append(count)
            prev = phase
        durations.extend(dur)
    panel["prev_ipc_duration"] = durations
    # Shift so it represents the *previous* month's duration
    panel["prev_ipc_duration"] = panel.groupby("region_code")["prev_ipc_duration"].shift(1)

    # Phase trend over 3 months: sign(current - 3-months-ago)
    panel["_phase_3mo_ago"] = panel.groupby("region_code")["ipc_phase"].shift(3)
    panel["phase_trend_3mo"] = np.sign(panel["ipc_phase"] - panel["_phase_3mo_ago"])
    panel.drop(columns=["_phase_3mo_ago"], inplace=True)

    return panel


def _compute_temporal_encoding(panel: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical month encoding and season dummies."""
    if "year_month" not in panel.columns:
        return panel

    # Extract month
    if hasattr(panel["year_month"].iloc[0], "month"):
        months = panel["year_month"].apply(lambda p: p.month)
    else:
        months = pd.to_datetime(panel["year_month"].astype(str)).dt.month

    # Cyclical encoding
    panel["month_sin"] = months.apply(lambda m: encode_cyclical_month(m)[0])
    panel["month_cos"] = months.apply(lambda m: encode_cyclical_month(m)[1])

    # Season dummies
    if "region_code" in panel.columns:
        season_records = [
            encode_season(m, r)
            for m, r in zip(months, panel["region_code"])
        ]
        season_df = pd.DataFrame(season_records, index=panel.index)
        for col in season_df.columns:
            if col not in panel.columns:
                panel[col] = season_df[col]

    return panel


def _compute_iod_lag(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-month lagged IOD (Indian Ocean Dipole leads short rains by ~1-2 months)."""
    if "iod_dmi" not in panel.columns:
        return panel
    if "iod_3mo_lag" in panel.columns:
        return panel

    panel = panel.sort_values(["region_code", "year_month"]).copy()
    panel["iod_3mo_lag"] = panel.groupby("region_code")["iod_dmi"].shift(3)

    return panel
