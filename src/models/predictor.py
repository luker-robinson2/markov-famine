"""Forecast pipeline orchestrator.

Brings together the data, feature engineering, ensemble model, and Markov
chain to produce end-to-end food security forecasts for a given region.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional
from dataclasses import dataclass

from src.config import N_STATES, LEAD_TIME_HORIZONS, IPC_LABELS
from src.models.markov_chain import NonHomogeneousMarkovChain, ForecastResult
from src.models.transition_model import TransitionModel

logger = logging.getLogger(__name__)


@dataclass
class RegionForecast:
    """Complete forecast output for a single region."""

    region_code: str
    region_name: str
    current_phase: int
    forecast_date: str
    horizon_months: int
    # Per-month forecast
    forecast: ForecastResult
    # Most likely phase at each horizon
    predicted_phases: list[int]
    # Probability of Phase 3+ (Crisis or worse) at each horizon
    crisis_probability: list[float]
    # Phase with highest probability at final horizon
    final_predicted_phase: int
    final_phase_probability: float


class FoodSecurityPredictor:
    """End-to-end forecast pipeline for IPC phase prediction.

    Usage:
        predictor = FoodSecurityPredictor(nhmc, transition_model)
        forecast = predictor.predict(
            region_code="KE023",  # Turkana
            current_phase=3,      # Crisis
            covariates_future=future_features,
            horizon=6,
        )
    """

    def __init__(
        self,
        nhmc: Optional[NonHomogeneousMarkovChain] = None,
        transition_model: Optional[TransitionModel] = None,
    ):
        if nhmc is None:
            nhmc = NonHomogeneousMarkovChain()
        if transition_model is not None:
            nhmc.set_transition_model(transition_model)
        self.nhmc = nhmc

    def predict(
        self,
        region_code: str,
        current_phase: int,
        covariates_future: np.ndarray,
        horizon: Optional[int] = None,
        n_simulations: int = 1000,
    ) -> RegionForecast:
        """Generate forecast for a single region.

        Parameters
        ----------
        region_code : str
            Admin-1 region identifier (e.g., "KE023" for Turkana).
        current_phase : int
            Current IPC phase (1-indexed).
        covariates_future : np.ndarray
            Shape (horizon, n_features). Future covariate vectors.
        horizon : int, optional
            Months to forecast (default: rows in covariates_future).
        n_simulations : int
            Monte Carlo simulations for confidence intervals.

        Returns
        -------
        RegionForecast
        """
        if horizon is None:
            horizon = covariates_future.shape[0]

        cov_seq = [covariates_future[t] for t in range(horizon)]

        forecast = self.nhmc.forecast(
            current_state=current_phase,
            covariates_sequence=cov_seq,
            horizon=horizon,
            n_simulations=n_simulations,
        )

        predicted_phases = forecast.predicted_states.tolist()

        # P(Phase >= 3) at each horizon
        crisis_probs = []
        for t in range(horizon):
            p_crisis = forecast.probabilities[t, 2:].sum()  # Phases 3,4,5
            crisis_probs.append(float(p_crisis))

        final_phase = int(forecast.predicted_states[-1])
        final_prob = float(forecast.probabilities[-1, final_phase - 1])

        from src.config import ALL_REGIONS
        region_name = ALL_REGIONS.get(region_code, region_code)

        return RegionForecast(
            region_code=region_code,
            region_name=region_name,
            current_phase=current_phase,
            forecast_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            horizon_months=horizon,
            forecast=forecast,
            predicted_phases=predicted_phases,
            crisis_probability=crisis_probs,
            final_predicted_phase=final_phase,
            final_phase_probability=final_prob,
        )

    def predict_all_regions(
        self,
        current_phases: dict[str, int],
        covariates_future: dict[str, np.ndarray],
        horizon: int = 6,
    ) -> dict[str, RegionForecast]:
        """Generate forecasts for all regions.

        Parameters
        ----------
        current_phases : dict
            Maps region_code -> current IPC phase.
        covariates_future : dict
            Maps region_code -> (horizon, n_features) covariate array.
        horizon : int
            Months to forecast.

        Returns
        -------
        dict mapping region_code -> RegionForecast
        """
        results = {}
        for region_code, phase in current_phases.items():
            if region_code not in covariates_future:
                logger.warning(f"No covariates for {region_code}, skipping")
                continue
            results[region_code] = self.predict(
                region_code=region_code,
                current_phase=phase,
                covariates_future=covariates_future[region_code],
                horizon=horizon,
            )
        return results

    def crisis_early_warning(
        self,
        forecasts: dict[str, RegionForecast],
        threshold: float = 0.5,
        lead_months: int = 3,
    ) -> pd.DataFrame:
        """Identify regions at risk of Phase 3+ within lead_months.

        Parameters
        ----------
        forecasts : dict
            Output of predict_all_regions.
        threshold : float
            Minimum P(Phase 3+) to trigger warning.
        lead_months : int
            Horizon to check.

        Returns
        -------
        pd.DataFrame
            Regions exceeding threshold, sorted by crisis probability.
        """
        warnings = []
        for region_code, forecast in forecasts.items():
            for t in range(min(lead_months, len(forecast.crisis_probability))):
                if forecast.crisis_probability[t] >= threshold:
                    warnings.append({
                        "region_code": region_code,
                        "region_name": forecast.region_name,
                        "current_phase": forecast.current_phase,
                        "current_phase_label": IPC_LABELS.get(forecast.current_phase, ""),
                        "months_ahead": t + 1,
                        "crisis_probability": forecast.crisis_probability[t],
                        "predicted_phase": forecast.predicted_phases[t],
                        "predicted_phase_label": IPC_LABELS.get(
                            forecast.predicted_phases[t], ""
                        ),
                    })
                    break  # Only report earliest warning per region

        df = pd.DataFrame(warnings)
        if not df.empty:
            df = df.sort_values("crisis_probability", ascending=False)
        return df
