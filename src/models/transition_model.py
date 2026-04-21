"""Bridge between ensemble ML models and the Markov chain transition matrix.

This module wraps any trained ensemble model to produce transition matrix rows,
implementing the TransitionPredictor protocol expected by NonHomogeneousMarkovChain.

The key equation:
    P_t(i, j) = f_i(X_t)

where f_i is the per-state classifier for origin state i, and the full row
[f_i(X_t)_1, ..., f_i(X_t)_5] gives the transition probabilities from state i.
"""

import numpy as np
import logging
from typing import Optional

from src.config import N_STATES
from src.models.ensemble import EnsembleSuite

logger = logging.getLogger(__name__)


class TransitionModel:
    """Wraps an ensemble model to produce Markov chain transition matrix rows.

    Implements the TransitionPredictor protocol for NonHomogeneousMarkovChain.
    """

    def __init__(
        self,
        ensemble: Optional[EnsembleSuite] = None,
        n_states: int = N_STATES,
        method: str = "stacking",
        empirical_fallback: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        ensemble : EnsembleSuite
            Trained ensemble model.
        n_states : int
            Number of IPC phases.
        method : str
            Prediction method ('stacking', 'average', or specific model type).
        empirical_fallback : np.ndarray, optional
            Shape (n_states, n_states) empirical transition matrix for fallback.
        """
        self.ensemble = ensemble
        self.n_states = n_states
        self.method = method
        self.empirical_fallback = empirical_fallback

    def predict_transition_row(
        self, origin_state: int, covariates: np.ndarray
    ) -> np.ndarray:
        """Predict P(next_state | origin_state, covariates).

        Parameters
        ----------
        origin_state : int
            Current IPC phase (1-indexed).
        covariates : np.ndarray
            Feature vector for current time step. Shape (n_features,).

        Returns
        -------
        np.ndarray
            Shape (n_states,) probability vector summing to 1.
        """
        if self.ensemble is not None:
            try:
                X = covariates.reshape(1, -1)
                proba = self.ensemble.predict_proba(
                    X, origin_state=origin_state, method=self.method
                )
                row = proba[0]

                # Ensure valid probability distribution
                row = np.maximum(row, 0)
                total = row.sum()
                if total > 0:
                    row = row / total
                else:
                    row = self._fallback(origin_state)

                return row

            except Exception as e:
                logger.warning(
                    f"Ensemble prediction failed for state {origin_state}: {e}. "
                    f"Using fallback."
                )

        return self._fallback(origin_state)

    def predict_transition_matrix(self, covariates: np.ndarray) -> np.ndarray:
        """Predict full transition matrix for given covariates.

        Convenience method that calls predict_transition_row for each state.

        Returns
        -------
        np.ndarray
            Shape (n_states, n_states) right-stochastic matrix.
        """
        P = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            P[i] = self.predict_transition_row(i + 1, covariates)
        return P

    def _fallback(self, origin_state: int) -> np.ndarray:
        """Return empirical transition probabilities or uniform distribution."""
        if self.empirical_fallback is not None:
            return self.empirical_fallback[origin_state - 1]
        return np.ones(self.n_states) / self.n_states

    @classmethod
    def from_ensemble(
        cls,
        ensemble: EnsembleSuite,
        method: str = "stacking",
        empirical_matrix: Optional[np.ndarray] = None,
    ) -> "TransitionModel":
        """Create TransitionModel from a trained EnsembleSuite."""
        return cls(
            ensemble=ensemble,
            method=method,
            empirical_fallback=empirical_matrix,
        )


class EmpiricalTransitionModel:
    """Simple transition model using only empirical (MLE) transition probabilities.

    Does not use covariates — returns the same transition row regardless of X_t.
    Useful as a wrapper to make HomogeneousMarkovChain work with NHMC interface.
    """

    def __init__(self, transition_matrix: np.ndarray):
        """
        Parameters
        ----------
        transition_matrix : np.ndarray
            Shape (n_states, n_states) stochastic matrix.
        """
        self.transition_matrix = transition_matrix
        self.n_states = transition_matrix.shape[0]

    def predict_transition_row(
        self, origin_state: int, covariates: np.ndarray
    ) -> np.ndarray:
        """Return the fixed empirical transition row (ignores covariates)."""
        return self.transition_matrix[origin_state - 1]


class HybridTransitionModel:
    """Wraps a HybridPredictor to produce transition matrix rows for NHMC.

    This adapter lets the new delta/phase hybrid model plug directly
    into the existing NonHomogeneousMarkovChain framework without
    modifying any Markov chain code.
    """

    def __init__(self, hybrid_predictor, n_states: int = 4):
        """
        Parameters
        ----------
        hybrid_predictor : HybridPredictor
            Trained hybrid model from ``src.models.delta_model``.
        n_states : int
            Number of IPC phases (4 in observed data).
        """
        self.predictor = hybrid_predictor
        self.n_states = n_states

    def predict_transition_row(
        self, origin_state: int, covariates: np.ndarray
    ) -> np.ndarray:
        """Produce a single transition-matrix row for the NHMC.

        Parameters
        ----------
        origin_state : int
            Current phase (1-indexed).
        covariates : np.ndarray
            Feature vector, shape ``(n_features,)`` or ``(1, n_features)``.

        Returns
        -------
        np.ndarray
            Probability vector of shape ``(n_states,)`` summing to 1.
        """
        X = covariates.reshape(1, -1) if covariates.ndim == 1 else covariates
        return self.predictor.transition_row(X, origin_state)

    def predict_transition_matrix(self, covariates: np.ndarray) -> np.ndarray:
        """Predict full transition matrix for given covariates.

        Returns
        -------
        np.ndarray
            Shape ``(n_states, n_states)`` right-stochastic matrix.
        """
        P = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            P[i] = self.predict_transition_row(i + 1, covariates)
        return P
