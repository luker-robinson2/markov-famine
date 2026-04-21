"""Regularized prediction models for IPC food security phases.

Three complementary models:

- **DeltaPredictor** — predicts phase *change* (delta), sidestepping
  persistence dominance (Westerveld 2021).
- **RegularizedPhasePredictor** — predicts phase level directly with
  aggressive regularisation to prevent memorisation.
- **HybridPredictor** — blends both for robust combined predictions
  and plugs into the NHMC framework.

All models use XGBoost with research-backed hyperparameters:
max_depth 3-4, heavy L1/L2, subsampling, and early stopping
(Machefer 2025 systematic review).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import RANDOM_STATE

logger = logging.getLogger(__name__)

DELTA_CLASSES = np.array([-2, -1, 0, 1, 2])
N_DELTA_CLASSES = len(DELTA_CLASSES)


# =====================================================================
#  Delta Predictor
# =====================================================================

class DeltaPredictor:
    """Predicts IPC phase change using class-weighted XGBoost.

    The target is ``delta = phase(t+1) - phase(t)`` mapped to classes
    0..4 corresponding to deltas {-2, -1, 0, +1, +2}.  Class weights
    boost rare transition events so the model doesn't collapse to
    "always predict zero change".
    """

    DEFAULT_PARAMS = {
        "max_depth": 3,
        "min_child_weight": 50,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "learning_rate": 0.02,
        "gamma": 2.0,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "objective": "multi:softprob",
        "num_class": N_DELTA_CLASSES,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.n_estimators_used: int = 0

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        delta_y_train: np.ndarray,
        X_val: np.ndarray,
        delta_y_val: np.ndarray,
        transition_boost: float = 15.0,
        early_stopping_rounds: int = 50,
        max_estimators: int = 1000,
    ) -> "DeltaPredictor":
        """Train the delta predictor with class-weighted early stopping.

        Parameters
        ----------
        X_train, X_val : array-like
            Feature matrices.
        delta_y_train, delta_y_val : array-like
            Delta targets in {-2, -1, 0, +1, +2}.
        transition_boost : float
            Weight multiplier for non-zero delta samples.
        early_stopping_rounds : int
            Stop if validation loss doesn't improve for this many rounds.
        max_estimators : int
            Upper bound on boosting rounds.
        """
        # Map deltas to class indices: -2→0, -1→1, 0→2, +1→3, +2→4
        y_train_idx = delta_y_train + 2
        y_val_idx = delta_y_val + 2

        # Compute sample weights: boost transitions
        weights = np.ones(len(y_train_idx), dtype=np.float64)
        weights[y_train_idx != 2] = transition_boost  # non-zero delta

        self.model = xgb.XGBClassifier(
            n_estimators=max_estimators,
            **self.params,
        )
        self.model.fit(
            X_train, y_train_idx,
            eval_set=[(X_val, y_val_idx)],
            sample_weight=weights,
            verbose=False,
        )
        self.n_estimators_used = self.model.best_iteration + 1 if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None else max_estimators
        logger.info(
            "DeltaPredictor fitted: %d estimators (early-stopped from %d)",
            self.n_estimators_used, max_estimators,
        )
        return self

    # ------------------------------------------------------------------
    #  Predict
    # ------------------------------------------------------------------

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely delta {-2, -1, 0, +1, +2}."""
        idx = self.model.predict(X)
        return DELTA_CLASSES[idx.astype(int)]

    def predict_delta_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probabilities over the 5 delta classes."""
        return self.model.predict_proba(X)

    def predict_phase(
        self, X: np.ndarray, current_phases: np.ndarray
    ) -> np.ndarray:
        """Predict next phase = current + delta, clipped to [1, 4]."""
        deltas = self.predict_delta(X)
        return np.clip(current_phases + deltas, 1, 4).astype(int)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return feature importance from the trained model."""
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# =====================================================================
#  Regularized Phase Predictor
# =====================================================================

class RegularizedPhasePredictor:
    """Predicts IPC phase level directly with aggressive regularisation.

    Includes ``prev_ipc_phase`` as a feature — the model learns when
    to deviate from persistence, not whether persistence works.
    """

    DEFAULT_PARAMS = {
        "max_depth": 3,
        "min_child_weight": 50,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "learning_rate": 0.02,
        "gamma": 2.0,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[xgb.XGBClassifier] = None
        self.n_estimators_used: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 50,
        max_estimators: int = 2000,
    ) -> "RegularizedPhasePredictor":
        """Train the phase predictor with early stopping.

        Parameters
        ----------
        y_train, y_val : array-like
            Phase labels 0-indexed (0=Phase1, 1=Phase2, 2=Phase3, 3=Phase4).
        """
        self.model = xgb.XGBClassifier(
            n_estimators=max_estimators,
            **self.params,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.n_estimators_used = self.model.best_iteration + 1 if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None else max_estimators
        logger.info(
            "RegularizedPhasePredictor fitted: %d estimators", self.n_estimators_used,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict phase (0-indexed: 0=Phase1 .. 3=Phase4)."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probabilities over 4 phases."""
        return self.model.predict_proba(X)

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# =====================================================================
#  Hybrid Predictor
# =====================================================================

class HybridPredictor:
    """Blends delta and phase predictors for robust output.

    The delta model contributes transition-detection skill; the phase
    model contributes stability.  The blend weight defaults to 0.6
    for the delta model (emphasising transition detection).
    """

    def __init__(
        self,
        delta_model: DeltaPredictor,
        phase_model: RegularizedPhasePredictor,
        delta_weight: float = 0.6,
        n_phases: int = 4,
    ):
        self.delta_model = delta_model
        self.phase_model = phase_model
        self.delta_weight = delta_weight
        self.phase_weight = 1.0 - delta_weight
        self.n_phases = n_phases

    def predict_proba(
        self, X: np.ndarray, current_phases: np.ndarray
    ) -> np.ndarray:
        """Blended probability distribution over phases.

        Converts delta probabilities to phase probabilities by
        shifting by current_phase, then blends with direct phase model.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_phases)`` probability matrix.
        """
        # Delta model → phase probabilities
        delta_proba = self.delta_model.predict_delta_proba(X)  # (n, 5)
        n = len(X)
        phase_from_delta = np.zeros((n, self.n_phases))

        for i in range(n):
            cp = int(current_phases[i])
            for di, delta in enumerate(DELTA_CLASSES):
                target_phase = cp + delta
                if 1 <= target_phase <= self.n_phases:
                    phase_from_delta[i, target_phase - 1] += delta_proba[i, di]
            # Normalise in case some deltas mapped outside [1,4]
            row_sum = phase_from_delta[i].sum()
            if row_sum > 0:
                phase_from_delta[i] /= row_sum

        # Phase model probabilities
        phase_direct = self.phase_model.predict_proba(X)  # (n, 4)

        # Blend
        blended = (
            self.delta_weight * phase_from_delta
            + self.phase_weight * phase_direct
        )
        # Renormalise
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = np.where(row_sums > 0, blended / row_sums, 1.0 / self.n_phases)

        return blended

    def predict_phase(
        self, X: np.ndarray, current_phases: np.ndarray
    ) -> np.ndarray:
        """Predict most likely phase (1-indexed)."""
        proba = self.predict_proba(X, current_phases)
        return proba.argmax(axis=1) + 1  # 1-indexed

    def transition_row(
        self, X_single: np.ndarray, origin_state: int
    ) -> np.ndarray:
        """Produce a single transition-matrix row for the NHMC.

        Parameters
        ----------
        X_single : np.ndarray
            Feature vector for one observation, shape ``(1, n_features)``.
        origin_state : int
            Current phase (1-indexed).

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(n_phases,)`` summing to 1.
        """
        current = np.array([origin_state])
        return self.predict_proba(X_single.reshape(1, -1), current)[0]
