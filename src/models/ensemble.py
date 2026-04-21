"""Ensemble ML models for IPC phase transition prediction.

Implements per-state classifiers using four gradient boosting variants
plus a stacking meta-learner. Each origin IPC phase gets its own set of
classifiers because the dynamics of transitioning FROM Phase 1 are
fundamentally different from transitioning FROM Phase 4.

Architecture:
    - Level 0: XGBoost, LightGBM, CatBoost, Random Forest (per-state)
    - Level 1: Logistic Regression meta-learner (stacking)
    - Quantile: XGBoost quantile regression for uncertainty intervals
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.config import N_STATES, RANDOM_STATE, N_FOLDS

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model training."""

    n_states: int = N_STATES
    random_state: int = RANDOM_STATE
    n_folds: int = N_FOLDS
    min_samples_per_state: int = 10  # Fall back to empirical if fewer samples

    # XGBoost hyperparameters
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
    })

    # LightGBM hyperparameters
    lgbm_params: dict = field(default_factory=lambda: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multiclass",
        "num_class": N_STATES,
        "verbose": -1,
    })

    # CatBoost hyperparameters
    catboost_params: dict = field(default_factory=lambda: {
        "depth": 6,
        "learning_rate": 0.1,
        "iterations": 200,
        "loss_function": "MultiClass",
        "verbose": 0,
        "random_seed": RANDOM_STATE,
    })

    # Random Forest hyperparameters
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": "balanced",
    })


class PerStateClassifier:
    """A single classifier trained on transitions from one origin IPC phase.

    For origin state i, this model predicts P(S_{t+1} = j | S_t = i, X_t).
    """

    def __init__(
        self,
        origin_state: int,
        model_type: str = "xgboost",
        config: Optional[EnsembleConfig] = None,
    ):
        self.origin_state = origin_state
        self.model_type = model_type
        self.config = config or EnsembleConfig()
        self.model: Optional[BaseEstimator] = None
        self.empirical_dist: Optional[np.ndarray] = None
        self.n_train_samples: int = 0
        self.classes_seen: Optional[np.ndarray] = None

    def _create_model(self) -> BaseEstimator:
        """Create the underlying sklearn-compatible classifier."""
        if self.model_type == "xgboost":
            params = {**self.config.xgb_params, "random_state": self.config.random_state}
            params["num_class"] = self.config.n_states
            return xgb.XGBClassifier(**params)
        elif self.model_type == "lightgbm":
            params = {**self.config.lgbm_params, "random_state": self.config.random_state}
            return lgb.LGBMClassifier(**params)
        elif self.model_type == "catboost":
            return cb.CatBoostClassifier(**self.config.catboost_params)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                **self.config.rf_params,
                random_state=self.config.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerStateClassifier":
        """Train classifier on transitions from this origin state.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
            Only samples where S_t = origin_state.
        y : np.ndarray
            Target states S_{t+1}, shape (n_samples,). Values in {0, ..., n_states-1}.
        """
        self.n_train_samples = len(y)

        # Always compute empirical distribution as fallback
        counts = np.zeros(self.config.n_states)
        for val in y:
            if 0 <= val < self.config.n_states:
                counts[int(val)] += 1
        total = counts.sum()
        self.empirical_dist = counts / total if total > 0 else np.ones(self.config.n_states) / self.config.n_states

        # Only train ML model if enough samples
        if self.n_train_samples >= self.config.min_samples_per_state:
            self.model = self._create_model()
            self.classes_seen = np.unique(y)

            try:
                self.model.fit(X, y)
            except Exception as e:
                logger.warning(
                    f"Failed to train {self.model_type} for state {self.origin_state}: {e}. "
                    f"Using empirical distribution."
                )
                self.model = None
        else:
            logger.info(
                f"State {self.origin_state}: only {self.n_train_samples} samples, "
                f"using empirical distribution"
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability distribution over next states.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_states) probability matrix.
        """
        n_samples = X.shape[0]

        if self.model is None:
            # Use empirical distribution for all samples
            return np.tile(self.empirical_dist, (n_samples, 1))

        try:
            raw_proba = self.model.predict_proba(X)
        except Exception:
            return np.tile(self.empirical_dist, (n_samples, 1))

        # Handle case where not all classes were seen during training
        if raw_proba.shape[1] < self.config.n_states:
            full_proba = np.zeros((n_samples, self.config.n_states))
            if self.classes_seen is not None:
                for idx, cls in enumerate(self.classes_seen):
                    full_proba[:, int(cls)] = raw_proba[:, idx]
            # Redistribute missing probability mass
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            full_proba = full_proba / row_sums
            return full_proba

        return raw_proba


class EnsembleSuite:
    """Full ensemble of per-state classifiers across multiple model types.

    For each origin state and each model type, trains a separate classifier.
    Optionally trains a stacking meta-learner that combines predictions.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.model_types = ["xgboost", "lightgbm", "catboost", "random_forest"]
        # Dict[model_type][origin_state] -> PerStateClassifier
        self.classifiers: dict[str, dict[int, PerStateClassifier]] = {}
        # Stacking meta-learner per origin state
        self.meta_learners: dict[int, LogisticRegression] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        state_from: np.ndarray,
        state_to: np.ndarray,
        fit_stacking: bool = True,
    ) -> "EnsembleSuite":
        """Train all models for all origin states.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        state_from : np.ndarray
            Origin states S_t, shape (n_samples,). 0-indexed.
        state_to : np.ndarray
            Target states S_{t+1}, shape (n_samples,). 0-indexed.
        fit_stacking : bool
            Whether to fit stacking meta-learner.
        """
        for model_type in self.model_types:
            self.classifiers[model_type] = {}
            for s in range(self.config.n_states):
                mask = state_from == s
                if mask.sum() == 0:
                    logger.warning(f"No samples for origin state {s+1}")
                    continue

                X_s = X[mask]
                y_s = state_to[mask]

                clf = PerStateClassifier(
                    origin_state=s + 1,
                    model_type=model_type,
                    config=self.config,
                )
                clf.fit(X_s, y_s)
                self.classifiers[model_type][s] = clf

                logger.info(
                    f"Trained {model_type} for state {s+1}: "
                    f"{clf.n_train_samples} samples"
                )

        if fit_stacking:
            self._fit_stacking(X, state_from, state_to)

        self._fitted = True
        return self

    def _fit_stacking(
        self,
        X: np.ndarray,
        state_from: np.ndarray,
        state_to: np.ndarray,
    ) -> None:
        """Fit stacking meta-learner using internal cross-validation.

        Level-0: Generate out-of-fold predictions from each base model.
        Level-1: Train logistic regression on concatenated level-0 outputs.
        """
        n_models = len(self.model_types)
        n_states = self.config.n_states

        for s in range(n_states):
            mask = state_from == s
            if mask.sum() < self.config.min_samples_per_state * 2:
                continue

            X_s = X[mask]
            y_s = state_to[mask]

            # Generate out-of-fold predictions
            meta_features = np.zeros((len(y_s), n_models * n_states))
            kf = StratifiedKFold(
                n_splits=min(self.config.n_folds, len(np.unique(y_s))),
                shuffle=True,
                random_state=self.config.random_state,
            )

            try:
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_s, y_s)):
                    X_train, X_val = X_s[train_idx], X_s[val_idx]
                    y_train = y_s[train_idx]

                    for m_idx, model_type in enumerate(self.model_types):
                        clf = PerStateClassifier(
                            origin_state=s + 1,
                            model_type=model_type,
                            config=self.config,
                        )
                        clf.fit(X_train, y_train)
                        proba = clf.predict_proba(X_val)
                        start_col = m_idx * n_states
                        end_col = start_col + n_states
                        meta_features[val_idx, start_col:end_col] = proba
            except ValueError:
                # Not enough classes per fold
                continue

            # Train meta-learner
            meta = LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
            )
            try:
                meta.fit(meta_features, y_s)
                self.meta_learners[s] = meta
            except Exception as e:
                logger.warning(f"Stacking failed for state {s+1}: {e}")

    def predict_proba(
        self,
        X: np.ndarray,
        origin_state: int,
        method: str = "stacking",
    ) -> np.ndarray:
        """Predict transition probabilities for a given origin state.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        origin_state : int
            Current IPC phase (1-indexed).
        method : str
            'stacking' (meta-learner), 'average' (simple average),
            or specific model type name.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_states) probability matrix.
        """
        s = origin_state - 1  # 0-indexed

        if method == "stacking" and s in self.meta_learners:
            return self._predict_stacking(X, s)
        elif method == "average":
            return self._predict_average(X, s)
        elif method in self.classifiers:
            if s in self.classifiers[method]:
                return self.classifiers[method][s].predict_proba(X)

        # Fallback to averaging
        return self._predict_average(X, s)

    def _predict_stacking(self, X: np.ndarray, state_idx: int) -> np.ndarray:
        """Predict using stacking meta-learner."""
        n_states = self.config.n_states
        n_models = len(self.model_types)
        meta_features = np.zeros((X.shape[0], n_models * n_states))

        for m_idx, model_type in enumerate(self.model_types):
            if state_idx in self.classifiers.get(model_type, {}):
                proba = self.classifiers[model_type][state_idx].predict_proba(X)
                start_col = m_idx * n_states
                meta_features[:, start_col:start_col + n_states] = proba

        return self.meta_learners[state_idx].predict_proba(meta_features)

    def _predict_average(self, X: np.ndarray, state_idx: int) -> np.ndarray:
        """Predict by averaging all base model predictions."""
        predictions = []
        for model_type in self.model_types:
            if state_idx in self.classifiers.get(model_type, {}):
                proba = self.classifiers[model_type][state_idx].predict_proba(X)
                predictions.append(proba)

        if not predictions:
            uniform = np.ones(self.config.n_states) / self.config.n_states
            return np.tile(uniform, (X.shape[0], 1))

        return np.mean(predictions, axis=0)

    def get_model(self, model_type: str, origin_state: int) -> Optional[PerStateClassifier]:
        """Get a specific per-state classifier."""
        s = origin_state - 1
        return self.classifiers.get(model_type, {}).get(s)

    def feature_importance(
        self, model_type: str = "xgboost", feature_names: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Get feature importance for each origin state.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance, origin_state
        """
        records = []
        for s, clf in self.classifiers.get(model_type, {}).items():
            if clf.model is None:
                continue
            try:
                importances = clf.model.feature_importances_
                for i, imp in enumerate(importances):
                    fname = feature_names[i] if feature_names else f"feature_{i}"
                    records.append({
                        "feature": fname,
                        "importance": imp,
                        "origin_state": s + 1,
                    })
            except AttributeError:
                continue

        return pd.DataFrame(records)


class QuantileEnsemble:
    """XGBoost quantile regression for prediction intervals.

    Trains separate models for p10, p50, p90 quantiles to provide
    uncertainty bounds on predicted IPC phase probabilities.
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        config: Optional[EnsembleConfig] = None,
    ):
        self.quantiles = quantiles
        self.config = config or EnsembleConfig()
        # Dict[quantile][origin_state] -> model
        self.models: dict[float, dict[int, xgb.XGBRegressor]] = {}

    def fit(
        self,
        X: np.ndarray,
        state_from: np.ndarray,
        state_to: np.ndarray,
    ) -> "QuantileEnsemble":
        """Train quantile regression models for each origin state and quantile."""
        for q in self.quantiles:
            self.models[q] = {}
            for s in range(self.config.n_states):
                mask = state_from == s
                if mask.sum() < self.config.min_samples_per_state:
                    continue

                X_s = X[mask]
                y_s = state_to[mask].astype(float)

                model = xgb.XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    max_depth=self.config.xgb_params.get("max_depth", 6),
                    learning_rate=self.config.xgb_params.get("learning_rate", 0.1),
                    n_estimators=self.config.xgb_params.get("n_estimators", 200),
                    random_state=self.config.random_state,
                )
                model.fit(X_s, y_s)
                self.models[q][s] = model

        return self

    def predict(
        self, X: np.ndarray, origin_state: int
    ) -> dict[float, np.ndarray]:
        """Predict quantile values for a given origin state.

        Returns
        -------
        dict
            Maps quantile -> predicted values array.
        """
        s = origin_state - 1
        results = {}
        for q in self.quantiles:
            if s in self.models.get(q, {}):
                results[q] = self.models[q][s].predict(X)
            else:
                results[q] = np.full(X.shape[0], float(s))
        return results
