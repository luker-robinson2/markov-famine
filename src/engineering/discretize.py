r"""IPC phase discretization utilities for the food security prediction system.

Handles validation, modal-phase selection, gap-filling, and binarization of
IPC Acute Food Insecurity phase classifications — the discrete state space
:math:`S = \{1, 2, 3, 4, 5\}` of our Markov chain model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import IPC_LABELS, IPCPhase, N_STATES

logger = logging.getLogger(__name__)

# Set of valid phase values
_VALID_PHASES = frozenset(int(p) for p in IPCPhase)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_ipc_phases(phases: pd.Series) -> pd.Series:
    """Validate that all non-null values are in the IPC phase set {1,2,3,4,5}.

    Parameters
    ----------
    phases : pd.Series
        Series of IPC phase values (int or float with NaN).

    Returns
    -------
    pd.Series
        Validated series cast to ``pd.Int64Dtype()`` (nullable integer).

    Raises
    ------
    ValueError
        If any non-null value is outside {1, 2, 3, 4, 5}.
    """
    non_null = phases.dropna()
    if len(non_null) == 0:
        return phases.astype(pd.Int64Dtype())

    int_vals = non_null.astype(int)
    invalid = int_vals[~int_vals.isin(_VALID_PHASES)]

    if len(invalid) > 0:
        bad_vals = sorted(invalid.unique())
        raise ValueError(
            f"Invalid IPC phase values found: {bad_vals}. "
            f"All values must be in {sorted(_VALID_PHASES)}."
        )

    result = phases.copy()
    result.loc[non_null.index] = int_vals
    return result.astype(pd.Int64Dtype())


def select_modal_phase(phase_populations: pd.DataFrame) -> pd.Series:
    """Select the modal (most common) IPC phase weighted by population.

    When IPC analyses report multiple phases within a single admin-1
    region (e.g. 40% Phase 2, 35% Phase 3, 25% Phase 4), this function
    selects the phase affecting the largest population.

    Parameters
    ----------
    phase_populations : pd.DataFrame
        Must contain columns:

        * ``region_code`` — admin-1 region identifier.
        * ``year_month`` — temporal index.
        * ``ipc_phase`` — phase value (1-5).
        * ``population`` — population count in that phase.

    Returns
    -------
    pd.Series
        One IPC phase per ``(region_code, year_month)`` pair, selected as
        the phase with the highest population weight.  Indexed by a
        MultiIndex of ``(region_code, year_month)``.
    """
    required = {"region_code", "year_month", "ipc_phase", "population"}
    missing = required - set(phase_populations.columns)
    if missing:
        raise KeyError(f"phase_populations is missing columns: {sorted(missing)}")

    df = phase_populations.copy()
    df["ipc_phase"] = validate_ipc_phases(df["ipc_phase"])

    # For each (region, month), pick phase with max population
    idx_cols = ["region_code", "year_month"]
    modal = (
        df.sort_values("population", ascending=False)
        .drop_duplicates(subset=idx_cols, keep="first")
        .set_index(idx_cols)["ipc_phase"]
    )

    return modal


def forward_fill_phases(
    df: pd.DataFrame,
    max_gap: int = 6,
) -> pd.DataFrame:
    """Forward-fill missing IPC months up to a maximum gap.

    IPC analyses are published at irregular intervals (often quarterly).
    Between publications, the last known phase is carried forward for up
    to ``max_gap`` months.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``region_code``, ``year_month``, and ``ipc_phase``.
    max_gap : int, default 6
        Maximum number of consecutive months to forward-fill.  Gaps longer
        than this remain NaN.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``ipc_phase`` forward-filled within each region.
    """
    required = {"region_code", "year_month", "ipc_phase"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"DataFrame is missing columns: {sorted(missing)}")

    df = df.sort_values(["region_code", "year_month"]).copy()

    filled_groups = []
    for region, group in df.groupby("region_code"):
        group = group.copy()
        group["ipc_phase"] = group["ipc_phase"].ffill(limit=max_gap)
        filled_groups.append(group)

    result = pd.concat(filled_groups, ignore_index=True)

    n_filled = result["ipc_phase"].notna().sum() - df["ipc_phase"].notna().sum()
    if n_filled > 0:
        logger.info(
            "[discretize.forward_fill_phases] Forward-filled %d missing IPC "
            "observations (max_gap=%d).",
            n_filled,
            max_gap,
        )

    return result


def binarize_crisis(
    phases: pd.Series,
    threshold: int = 3,
) -> pd.Series:
    r"""Convert IPC phases to a binary crisis indicator.

    .. math::

        y_i = \begin{cases}
            0 & \text{if phase} \in \{1, 2\} \\
            1 & \text{if phase} \in \{3, 4, 5\}
        \end{cases}

    Parameters
    ----------
    phases : pd.Series
        IPC phase values (1-5).  NaN values remain NaN.
    threshold : int, default 3
        Phase at or above which the indicator is set to 1.

    Returns
    -------
    pd.Series
        Binary series: 0 = no crisis (Phase 1-2), 1 = crisis (Phase 3+).
        Uses ``pd.Int64Dtype()`` to preserve NaN.
    """
    result = pd.Series(pd.NA, index=phases.index, dtype=pd.Int64Dtype())
    mask = phases.notna()
    result.loc[mask] = (phases.loc[mask].astype(int) >= threshold).astype(int)
    return result.astype(pd.Int64Dtype())
