"""Training and evaluation pipeline for food security prediction.

Provides temporal data splitting, sample weighting, model evaluation
using the project's existing metrics suite, and expanding-window
temporal cross-validation.

All evaluation uses :mod:`src.metrics` functions for consistency.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

from src.config import TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from src.metrics.ordinal import (
    quadratic_weighted_kappa,
    crisis_detection_metrics,
    asymmetric_cost_score,
    weighted_f1_score,
)
from src.metrics.probabilistic import ranked_probability_skill_score

logger = logging.getLogger(__name__)


# =====================================================================
#  Data splitting
# =====================================================================

def prepare_splits(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target: str = "delta",
    train_end: str = TRAIN_END,
    val_start: str = VALID_START,
    val_end: str = VALID_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> dict:
    """Create temporal train/val/test splits.

    Parameters
    ----------
    panel : pd.DataFrame
        Enhanced panel with ``date``, ``region_code``, ``ipc_phase``.
    feature_cols : list[str]
        Columns to use as features.
    target : str
        ``"delta"`` for phase change or ``"phase"`` for phase level.

    Returns
    -------
    dict
        Keys: ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test``, ``current_train``, ``current_val``,
        ``current_test``, ``meta``.
    """
    df = panel.sort_values(["region_code", "date"]).copy()

    # Create next-month target
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)
    df["delta"] = df["next_phase"] - df["ipc_phase"]

    # Temporal splits
    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]

    if target == "delta":
        y_col = "delta"
    else:
        y_col = "next_phase"

    def _extract(subset):
        X = subset[feature_cols].fillna(0).values
        y = subset[y_col].values
        current = subset["ipc_phase"].values
        return X, y, current

    X_train, y_train, cur_train = _extract(train)
    X_val, y_val, cur_val = _extract(val)
    X_test, y_test, cur_test = _extract(test)

    meta = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "target": target,
    }
    logger.info(
        "Splits: train=%d, val=%d, test=%d (%d features, target=%s)",
        meta["n_train"], meta["n_val"], meta["n_test"], meta["n_features"], target,
    )

    return {
        "X_train": X_train, "y_train": y_train, "current_train": cur_train,
        "X_val": X_val, "y_val": y_val, "current_val": cur_val,
        "X_test": X_test, "y_test": y_test, "current_test": cur_test,
        "meta": meta,
    }


# =====================================================================
#  Sample weights
# =====================================================================

def compute_sample_weights(
    y_train: np.ndarray,
    boost_factor: float = 15.0,
    zero_class: int = 0,
) -> np.ndarray:
    """Upweight non-zero delta samples by *boost_factor*.

    For delta target: zero_class should be 0 (no change).
    """
    weights = np.ones(len(y_train), dtype=np.float64)
    weights[y_train != zero_class] = boost_factor
    n_boosted = (y_train != zero_class).sum()
    logger.info(
        "Sample weights: %d/%d boosted by %.0fx",
        n_boosted, len(y_train), boost_factor,
    )
    return weights


# =====================================================================
#  Model evaluation
# =====================================================================

def evaluate_model(
    y_true_phases: np.ndarray,
    y_pred_phases: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    current_phases: Optional[np.ndarray] = None,
    model_name: str = "Model",
    n_states: int = 4,
) -> dict:
    """Compute comprehensive metrics using the project's metrics suite.

    Parameters
    ----------
    y_true_phases : array-like
        True next-phase values (1-indexed: 1-4).
    y_pred_phases : array-like
        Predicted next-phase values (1-indexed: 1-4).
    y_pred_proba : array-like, optional
        Predicted probabilities, shape ``(n, n_states)``.
    current_phases : array-like, optional
        Current-period phases (for transition analysis).
    model_name : str
        Label for the results row.
    n_states : int
        Number of IPC phases in data.

    Returns
    -------
    dict
        Metric name → value.
    """
    # Convert to 0-indexed for metrics that expect it
    yt = np.asarray(y_true_phases)
    yp = np.asarray(y_pred_phases)
    yt0 = yt - 1  # 0-indexed
    yp0 = yp - 1

    results = {"model": model_name}

    # Basic metrics
    results["accuracy"] = accuracy_score(yt, yp)
    results["r2"] = r2_score(yt, yp)
    results["mae"] = mean_absolute_error(yt, yp)

    # Ordinal metrics (from src/metrics/ordinal.py)
    results["qwk"] = quadratic_weighted_kappa(yt0, yp0, n_states=n_states)
    results["weighted_f1"] = weighted_f1_score(yt0, yp0, n_states=n_states)
    results["asymmetric_cost"] = asymmetric_cost_score(yt0, yp0)

    # Crisis detection (Phase 3+ = index 2+)
    crisis = crisis_detection_metrics(yt0, yp0, crisis_threshold=2)
    results["crisis_recall"] = crisis["recall"]
    results["crisis_precision"] = crisis["precision"]
    results["crisis_f1"] = crisis["f1"]
    results["false_alarm_rate"] = crisis["false_alarm_rate"]

    # Probabilistic metrics
    if y_pred_proba is not None:
        results["rpss"] = ranked_probability_skill_score(
            y_pred_proba, yt0, n_states=n_states,
        )

    # Transition-conditional metrics (how well do we detect changes?)
    if current_phases is not None:
        cp = np.asarray(current_phases)
        actual_transitions = yt != cp
        n_transitions = actual_transitions.sum()
        if n_transitions > 0:
            # Of actual transitions, how many did we predict correctly?
            transition_correct = (yp[actual_transitions] == yt[actual_transitions]).sum()
            results["transition_accuracy"] = transition_correct / n_transitions
            # Of actual transitions, how many did we predict as ANY change?
            predicted_change = yp[actual_transitions] != cp[actual_transitions]
            results["transition_detection_rate"] = predicted_change.sum() / n_transitions
        else:
            results["transition_accuracy"] = np.nan
            results["transition_detection_rate"] = np.nan
        results["n_transitions"] = int(n_transitions)
        results["n_total"] = len(yt)

    return results


# =====================================================================
#  Model comparison
# =====================================================================

def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """Format comparison table from a list of evaluate_model outputs."""
    df = pd.DataFrame(results_list)
    # Reorder columns
    priority = [
        "model", "accuracy", "r2", "qwk", "rpss", "mae",
        "crisis_recall", "crisis_precision", "crisis_f1",
        "transition_accuracy", "transition_detection_rate",
        "weighted_f1", "false_alarm_rate", "asymmetric_cost",
        "n_transitions", "n_total",
    ]
    cols = [c for c in priority if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


# =====================================================================
#  Temporal cross-validation
# =====================================================================

def temporal_cross_validate(
    panel: pd.DataFrame,
    model_factory: callable,
    feature_cols: list[str],
    n_folds: int = 5,
    min_train_months: int = 48,
    val_months: int = 12,
) -> pd.DataFrame:
    """Expanding-window temporal cross-validation.

    Folds:
      Fold 1: train [0, T₁], validate [T₁, T₁+12]
      Fold 2: train [0, T₂], validate [T₂, T₂+12]
      ...
    where T_k increases by ``val_months`` each fold.

    Parameters
    ----------
    panel : pd.DataFrame
        Full enhanced panel.
    model_factory : callable
        Function ``(X_tr, y_tr, X_val, y_val, current_tr) -> model``
        that trains and returns a model with ``.predict_phase(X, current)``
        and optionally ``.predict_proba(X, current)`` methods.
    feature_cols : list[str]
        Feature column names.
    n_folds : int
        Number of CV folds.
    min_train_months : int
        Minimum training window in months.
    val_months : int
        Size of validation window in months.

    Returns
    -------
    pd.DataFrame
        One row per fold with metrics.
    """
    df = panel.sort_values(["region_code", "date"]).copy()
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)

    dates = sorted(df["date"].unique())
    n_dates = len(dates)

    # Compute fold boundaries
    first_val_start_idx = min_train_months
    step = max(1, (n_dates - first_val_start_idx - val_months) // max(n_folds - 1, 1))

    fold_results = []

    for fold in range(n_folds):
        val_start_idx = first_val_start_idx + fold * step
        val_end_idx = min(val_start_idx + val_months, n_dates)
        if val_start_idx >= n_dates or val_end_idx > n_dates:
            break

        train_end_date = dates[val_start_idx - 1]
        val_start_date = dates[val_start_idx]
        val_end_date = dates[val_end_idx - 1]

        train = df[df["date"] <= train_end_date]
        val = df[(df["date"] >= val_start_date) & (df["date"] <= val_end_date)]

        if len(train) < 100 or len(val) < 20:
            continue

        X_tr = train[feature_cols].fillna(0).values
        y_tr = train["next_phase"].values
        cur_tr = train["ipc_phase"].values

        X_val = val[feature_cols].fillna(0).values
        y_val = val["next_phase"].values
        cur_val = val["ipc_phase"].values

        try:
            model = model_factory(X_tr, y_tr, X_val, y_val, cur_tr)
            y_pred = model.predict_phase(X_val, cur_val)

            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X_val, cur_val)
                except TypeError:
                    proba = model.predict_proba(X_val)

            metrics = evaluate_model(
                y_val, y_pred, proba, cur_val,
                model_name=f"Fold {fold + 1}",
            )
            metrics["fold"] = fold + 1
            metrics["train_end"] = str(train_end_date.date()) if hasattr(train_end_date, "date") else str(train_end_date)
            metrics["val_period"] = f"{val_start_date.date()} to {val_end_date.date()}" if hasattr(val_start_date, "date") else f"{val_start_date} to {val_end_date}"
            fold_results.append(metrics)
        except Exception as e:
            logger.warning("Fold %d failed: %s", fold + 1, e)

    if not fold_results:
        logger.error("All CV folds failed")
        return pd.DataFrame()

    result_df = pd.DataFrame(fold_results)
    logger.info(
        "Temporal CV: %d folds, mean accuracy=%.3f ± %.3f",
        len(result_df),
        result_df["accuracy"].mean(),
        result_df["accuracy"].std(),
    )
    return result_df
