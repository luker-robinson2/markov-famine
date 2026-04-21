"""Temporal alignment utilities for the food security prediction system.

Resamples heterogeneous time-series (daily, dekadal, 16-day, monthly) onto a
common monthly grid indexed by ``(region_code, year_month)`` and encodes
cyclical / seasonal features.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import ANALYSIS_END, ANALYSIS_START, SEASONS, get_season

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resample_to_monthly(
    df: pd.DataFrame,
    agg_method: str = "mean",
    date_col: str = "date",
    value_cols: Optional[list[str]] = None,
    region_col: str = "region_code",
) -> pd.DataFrame:
    """Resample any temporal DataFrame to calendar-monthly frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a parseable date column and optionally a region column.
    agg_method : str, default ``"mean"``
        Aggregation method — one of ``"mean"``, ``"sum"``, ``"median"``,
        ``"max"``, ``"min"``, ``"last"``, ``"first"``.
    date_col : str, default ``"date"``
        Name of the datetime column.
    value_cols : list[str], optional
        Columns to aggregate.  If ``None``, all numeric columns (excluding
        the region column) are used.
    region_col : str, default ``"region_code"``
        Column to group by (admin-1 region).  Set to ``None`` if the
        DataFrame has no spatial dimension.

    Returns
    -------
    pd.DataFrame
        Columns: ``region_code``, ``year_month`` (Period[M]), plus the
        aggregated value columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if value_cols is None:
        exclude = {date_col}
        if region_col and region_col in df.columns:
            exclude.add(region_col)
        value_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    agg_funcs = {
        "mean": "mean",
        "sum": "sum",
        "median": "median",
        "max": "max",
        "min": "min",
        "last": "last",
        "first": "first",
    }
    if agg_method not in agg_funcs:
        raise ValueError(f"Unknown agg_method={agg_method!r}. Choose from {list(agg_funcs)}")

    df["year_month"] = df[date_col].dt.to_period("M")

    group_cols = ["year_month"]
    if region_col and region_col in df.columns:
        group_cols = [region_col, "year_month"]

    result = (
        df.groupby(group_cols, as_index=False)[value_cols]
        .agg(agg_funcs[agg_method])
    )

    return result


def align_to_monthly_grid(
    dfs: dict[str, pd.DataFrame],
    start: Optional[str] = None,
    end: Optional[str] = None,
    region_col: str = "region_code",
    time_col: str = "year_month",
) -> pd.DataFrame:
    """Merge multiple DataFrames on a common ``(region_code, year_month)`` grid.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping of source name to DataFrame.  Each DataFrame must already be
        at monthly frequency with a ``year_month`` column (``Period[M]``) and
        an optional ``region_code`` column.
    start, end : str, optional
        ISO date strings bounding the grid.  Default to
        ``config.ANALYSIS_START`` / ``config.ANALYSIS_END``.
    region_col, time_col : str
        Column names used for the spatial and temporal index.

    Returns
    -------
    pd.DataFrame
        Outer join of all input DataFrames on ``(region_code, year_month)``.
    """
    start = start or ANALYSIS_START
    end = end or ANALYSIS_END

    # Build the full grid
    periods = pd.period_range(start=start, end=end, freq="M")

    # Collect all unique region codes across DataFrames
    all_regions: set[str] = set()
    for name, df in dfs.items():
        if region_col in df.columns:
            all_regions.update(df[region_col].dropna().unique())

    if not all_regions:
        logger.warning(
            "[temporal.align_to_monthly_grid] No region codes found in any "
            "DataFrame. Building grid from time dimension only."
        )
        grid = pd.DataFrame({time_col: periods})
    else:
        grid = pd.DataFrame(
            [
                (r, p)
                for r in sorted(all_regions)
                for p in periods
            ],
            columns=[region_col, time_col],
        )

    # Merge each source
    for name, df in dfs.items():
        df = df.copy()
        # Ensure period dtype
        if time_col in df.columns and not hasattr(df[time_col].dtype, "freq"):
            df[time_col] = pd.PeriodIndex(df[time_col], freq="M")

        merge_on = [time_col]
        if region_col in df.columns and region_col in grid.columns:
            merge_on = [region_col, time_col]

        # Prefix columns to avoid collisions (except merge keys)
        rename_map = {
            c: f"{name}_{c}"
            for c in df.columns
            if c not in merge_on
        }
        df = df.rename(columns=rename_map)

        grid = grid.merge(df, on=merge_on, how="left")

    return grid


def encode_season(month: int, region_code: str) -> dict[str, int]:
    """Create one-hot encoded season features for a given month and region.

    Uses ``config.get_season()`` to determine the dominant season, then
    returns binary indicators for the three seasons used as model features:
    ``gu``, ``deyr``, and ``kiremt``.

    Parameters
    ----------
    month : int
        Calendar month (1-12).
    region_code : str
        Internal region code (e.g. ``"KE007"``).

    Returns
    -------
    dict[str, int]
        Keys: ``"season_gu"``, ``"season_deyr"``, ``"season_kiremt"``.
        Exactly one key is ``1``; the rest are ``0`` — or all zeros if the
        month falls outside these three primary seasons.
    """
    season = get_season(month, region_code)
    return {
        "season_gu": int(season == "gu"),
        "season_deyr": int(season == "deyr"),
        "season_kiremt": int(season == "kiremt"),
    }


def encode_cyclical_month(month: int) -> tuple[float, float]:
    r"""Encode a calendar month as a cyclical sine/cosine pair.

    .. math::

        \text{month\_sin} = \sin\!\left(\frac{2\pi \cdot m}{12}\right), \qquad
        \text{month\_cos} = \cos\!\left(\frac{2\pi \cdot m}{12}\right)

    This preserves the circular continuity between December (12) and
    January (1).

    Parameters
    ----------
    month : int
        Calendar month (1-12).

    Returns
    -------
    tuple[float, float]
        ``(month_sin, month_cos)``.
    """
    angle = 2 * np.pi * month / 12
    return float(np.sin(angle)), float(np.cos(angle))
