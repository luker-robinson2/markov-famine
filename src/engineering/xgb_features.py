"""XGBoost-derived meta-features for the food security prediction system.

Uses XGBoost as a feature-engineering tool: SHAP-based importance, leaf-index
embeddings, anomaly scoring, and recursive feature elimination.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import FEATURE_GROUPS, RANDOM_STATE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_shap_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
) -> pd.DataFrame:
    """Compute SHAP feature importance using an XGBoost classifier.

    If no model is supplied, a preliminary ``XGBClassifier`` is trained on
    the provided data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric columns only).
    y : pd.Series
        Target variable (IPC phase or binary crisis indicator).
    model : xgboost.XGBClassifier, optional
        Pre-trained model.  If ``None`` a default model is fit.

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``feature`` — feature name.
        * ``mean_abs_shap`` — mean |SHAP value| across observations (global
          importance).
        * Per-observation SHAP values in columns ``shap_0 .. shap_N``.
    """
    import shap
    import xgboost as xgb

    if model is None:
        logger.info("[xgb_features.compute_shap_importance] Training preliminary XGBoost model.")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # shap_values may be a list (multi-class) or ndarray (binary)
    if isinstance(shap_values, list):
        # Average across classes
        shap_array = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        shap_array = np.abs(shap_values)

    mean_abs = shap_array.mean(axis=0)

    result = pd.DataFrame({
        "feature": X.columns.tolist(),
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Append per-observation SHAP values
    shap_df = pd.DataFrame(
        shap_array if not isinstance(shap_values, list) else np.mean(np.stack(shap_values, axis=0), axis=0),
        columns=[f"shap_{i}" for i in range(shap_array.shape[1])],
        index=X.index,
    )
    # Attach as attribute for downstream use
    result.attrs["shap_per_obs"] = shap_df

    return result


def extract_leaf_indices(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """Extract leaf node indices from a trained XGBoost model.

    Leaf indices serve as categorical features that capture non-linear
    interactions learned by the tree ensemble.

    Parameters
    ----------
    model : xgboost.XGBClassifier or xgboost.Booster
        Trained XGBoost model.
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples, n_trees)``.  Each entry is the leaf index
        for the corresponding sample in the corresponding tree.
    """
    leaf_indices = model.predict(X, pred_leaf=True)
    return leaf_indices


def compute_anomaly_scores(
    X: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.Series:
    """Compute anomaly scores using Isolation Forest on climate features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (typically climate-related columns).
    contamination : float, default 0.05
        Expected proportion of anomalies in the dataset.

    Returns
    -------
    pd.Series
        Anomaly scores — lower (more negative) values indicate stronger
        anomalies.  Indexed to match *X*.
    """
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Fill NaN for fitting
    X_filled = X.fillna(X.median())
    iso.fit(X_filled)

    # decision_function: lower = more anomalous
    scores = iso.decision_function(X_filled)

    return pd.Series(scores, index=X.index, name="anomaly_score")


def recursive_feature_elimination(
    X: pd.DataFrame,
    y: pd.Series,
    min_features: int = 10,
) -> list[str]:
    """XGBoost-based recursive feature elimination (RFE).

    Iteratively removes the least important feature (by XGBoost
    ``feature_importances_``) until ``min_features`` remain.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    min_features : int, default 10
        Minimum number of features to retain.

    Returns
    -------
    list[str]
        Ordered list of selected feature names (most important first).
    """
    import xgboost as xgb

    remaining = list(X.columns)

    while len(remaining) > min_features:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        model.fit(X[remaining], y)

        importances = pd.Series(model.feature_importances_, index=remaining)
        worst = importances.idxmin()
        remaining.remove(worst)
        logger.debug("[xgb_features.rfe] Removed %s (importance=%.4f)", worst, importances[worst])

    # Final ranking
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X[remaining], y)
    importances = pd.Series(model.feature_importances_, index=remaining)

    return importances.sort_values(ascending=False).index.tolist()


def build_xgb_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all XGBoost-based feature engineering.

    Adds the following columns to *panel*:

    * ``anomaly_score`` — Isolation Forest anomaly score on climate features.
    * ``leaf_idx_0 .. leaf_idx_N`` — XGBoost leaf indices (categorical).
    * ``shap_selected`` — 1 if the observation's features pass SHAP-based
      importance filtering, else 0.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of :func:`features.build_feature_panel`.

    Returns
    -------
    pd.DataFrame
        Enriched panel with XGBoost meta-features.
    """
    import xgboost as xgb

    panel = panel.copy()
    idx_cols = ["region_code", "year_month"]

    # Identify target
    target_col = "ipc_phase"
    if target_col not in panel.columns:
        logger.warning(
            "[xgb_features.build_xgb_features] No '%s' column found; "
            "returning panel unchanged.",
            target_col,
        )
        return panel

    # Identify numeric feature columns (exclude index and target)
    feature_cols = [
        c
        for c in panel.columns
        if c not in idx_cols + [target_col]
        and pd.api.types.is_numeric_dtype(panel[c])
    ]

    if not feature_cols:
        logger.warning("[xgb_features.build_xgb_features] No numeric feature columns found.")
        return panel

    # Drop rows with missing target
    mask = panel[target_col].notna()
    X = panel.loc[mask, feature_cols].fillna(0)
    y = panel.loc[mask, target_col].astype(int)

    # ------------------------------------------------------------------
    # 1. Anomaly scores (on climate features only)
    # ------------------------------------------------------------------
    climate_cols = [c for c in FEATURE_GROUPS.get("climate_raw", []) if c in X.columns]
    if climate_cols:
        panel.loc[mask, "anomaly_score"] = compute_anomaly_scores(X[climate_cols]).values
    else:
        panel.loc[mask, "anomaly_score"] = compute_anomaly_scores(X).values

    # ------------------------------------------------------------------
    # 2. Leaf indices from preliminary model
    # ------------------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X, y)

    leaves = extract_leaf_indices(model, X)
    for i in range(leaves.shape[1]):
        col_name = f"leaf_idx_{i}"
        panel.loc[mask, col_name] = leaves[:, i]

    # ------------------------------------------------------------------
    # 3. SHAP importance → binary selection flag
    # ------------------------------------------------------------------
    shap_result = compute_shap_importance(X, y, model=model)
    top_features = shap_result.head(max(10, len(shap_result) // 2))["feature"].tolist()

    # Flag: 1 if all top-SHAP features are non-null for this observation
    panel["shap_selected"] = 0
    top_available = [c for c in top_features if c in panel.columns]
    if top_available:
        panel.loc[panel[top_available].notna().all(axis=1), "shap_selected"] = 1

    return panel
