#!/usr/bin/env python3
"""Multi-horizon NHMC evaluation (t+1, t+3, t+6).

The paper motivates a 3–6 month humanitarian lead time but only
evaluates one-step-ahead. This script uses the existing multi-step
product of transition matrices (``NonHomogeneousMarkovChain.forecast``)
to produce honest t+1 / t+3 / t+6 forecasts on the test set, with
skill metrics referenced to persistence.

For each test (region, start_month) and horizon h:
  1. Build π_t = one-hot on the observed S_t.
  2. Iterate: for k in 0..h-1, look up X_{t+k} from the panel and
     apply π_{t+k+1} = π_{t+k} · P_{t+k}, where each P_{t+k} row is
     produced by ``HybridPredictor.transition_row``.
  3. Predicted phase is ``argmax(π_{t+h}) + 1``.
  4. Compare to the observed S_{t+h}.

Note: using observed covariates at t+1..t+h-1 is a partial-information
evaluation. A fully out-of-sample forecast would also predict the
covariates. We follow Busker et al. 2024 and Westerveld et al. 2021,
which use observed covariates at each intermediate step — the lead-time
result is about propagating state uncertainty through P_t, not about
forecasting X_t itself.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/multi_horizon_eval.py
"""

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END, N_STATES
from src.engineering.enhanced_features import build_enhanced_panel
from src.models.delta_model import DeltaPredictor, RegularizedPhasePredictor, HybridPredictor
from src.models.evaluation import evaluate_model, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_horizon")

HORIZONS = (1, 3, 6)


def build_feature_cols(panel: pd.DataFrame) -> list[str]:
    """Exclude non-feature columns and the target."""
    skip = {
        "region_code", "date", "country", "ipc_phase",
        "next_phase", "delta",
    }
    return [c for c in panel.columns if c not in skip]


def train_models(panel: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train DeltaPredictor, PhasePredictor, Hybrid on the train/val split."""
    df = panel.sort_values(["region_code", "date"]).copy()
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)
    df["delta"] = df["next_phase"] - df["ipc_phase"]

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VALID_START) & (df["date"] <= VALID_END)]

    X_tr = train[feature_cols].fillna(0).values
    y_tr_phase = train["next_phase"].values
    y_tr_delta = train["delta"].values

    X_va = val[feature_cols].fillna(0).values
    y_va_phase = val["next_phase"].values
    y_va_delta = val["delta"].values

    logger.info("Training DeltaPredictor (boost=15.0)...")
    delta = DeltaPredictor().fit(
        X_tr, y_tr_delta, X_va, y_va_delta, transition_boost=15.0,
    )
    logger.info("Training RegularizedPhasePredictor (0-indexed targets)...")
    phase = RegularizedPhasePredictor().fit(
        X_tr, y_tr_phase - 1, X_va, y_va_phase - 1,
    )
    logger.info("Training HybridPredictor (0.6 delta + 0.4 phase)...")
    hybrid = HybridPredictor(delta_model=delta, phase_model=phase, delta_weight=0.6)

    return {"delta": delta, "phase": phase, "hybrid": hybrid}


def build_test_table(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Filter to region-months that have enough future data for the longest horizon."""
    df = panel.sort_values(["region_code", "date"]).copy()
    df["features"] = df[feature_cols].fillna(0).values.tolist()
    # For each (region, start_date), we need future rows at +1, +3, +6.
    out_rows = []
    max_h = max(HORIZONS)
    for rc, grp in df.groupby("region_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for idx in range(len(grp)):
            start_date = grp.loc[idx, "date"]
            if start_date < pd.Timestamp(TEST_START) or start_date > pd.Timestamp(TEST_END):
                continue
            if idx + max_h >= len(grp):
                continue  # not enough future rows
            row = {
                "region_code": rc,
                "start_date": start_date,
                "S_t": int(grp.loc[idx, "ipc_phase"]),
            }
            # Covariate sequence and observed states for h = 1..max_h
            for k in range(1, max_h + 1):
                row[f"S_tph_{k}"] = int(grp.loc[idx + k, "ipc_phase"])
                row[f"X_tph_{k - 1}"] = np.asarray(grp.loc[idx + k - 1, feature_cols].fillna(0).values, dtype=float)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def forecast_one(model: HybridPredictor, start_state: int, covariate_seq: list[np.ndarray], n_states: int = 4) -> np.ndarray:
    """Iteratively apply π_{t+k+1} = π_{t+k} · P_{t+k} for each covariate in the sequence.

    Returns an array of shape (len(covariate_seq), n_states) — one row per step.
    """
    pi = np.zeros(n_states)
    pi[start_state - 1] = 1.0
    traj = np.zeros((len(covariate_seq), n_states))
    for k, X in enumerate(covariate_seq):
        # Build P_k by asking the hybrid model for each origin row
        P = np.zeros((n_states, n_states))
        X2 = X.reshape(1, -1)
        for i in range(n_states):
            P[i] = model.transition_row(X2, origin_state=i + 1)
        pi = pi @ P
        traj[k] = pi
    return traj


def run_horizon(models: dict, test_table: pd.DataFrame, h: int, n_states: int = 4) -> dict:
    """Evaluate the hybrid, delta, and phase models at horizon h.

    Returns a dict with one result row per model.
    """
    hybrid = models["hybrid"]
    delta = models["delta"]
    phase = models["phase"]

    y_true = test_table[f"S_tph_{h}"].to_numpy()
    current = test_table["S_t"].to_numpy()

    # Hybrid via NHMC iteration
    pred_hybrid = np.zeros(len(test_table), dtype=int)
    proba_hybrid = np.zeros((len(test_table), n_states))
    for i, row in enumerate(test_table.itertuples(index=False)):
        cov_seq = [getattr(row, f"X_tph_{k}") for k in range(h)]
        traj = forecast_one(hybrid, int(getattr(row, "S_t")), cov_seq, n_states=n_states)
        pi_h = traj[-1]
        pred_hybrid[i] = int(np.argmax(pi_h)) + 1
        proba_hybrid[i] = pi_h

    # Phase model: direct predict on X_{t+h-1}. Returns 0-indexed phases.
    X_phase = np.stack([row[f"X_tph_{h - 1}"] for _, row in test_table.iterrows()])
    pred_phase = phase.predict(X_phase) + 1  # convert to 1-indexed
    try:
        proba_phase = phase.predict_proba(X_phase)
    except Exception:
        proba_phase = None

    # Delta model: applied iteratively (re-estimate δ each step from X_{t+k})
    pred_delta = current.copy()
    for k in range(h):
        X_k = np.stack([row[f"X_tph_{k}"] for _, row in test_table.iterrows()])
        pred_delta = delta.predict_phase(X_k, pred_delta)

    # Persistence baseline
    pred_persist = current.copy()

    # Evaluate
    results = []
    for name, pred, proba in [
        (f"Hybrid (NHMC) t+{h}", pred_hybrid, proba_hybrid),
        (f"PhasePredictor t+{h}", pred_phase, proba_phase),
        (f"DeltaPredictor t+{h}", pred_delta, None),
        (f"Persistence t+{h}", pred_persist, None),
    ]:
        res = evaluate_model(
            y_true_phases=y_true,
            y_pred_phases=pred,
            y_pred_proba=proba,
            current_phases=current,
            model_name=name,
            n_states=n_states,
        )
        res["horizon"] = h
        results.append(res)
    return results


def main() -> None:
    logger.info("Loading panel...")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    panel = build_enhanced_panel(panel, train_end=TRAIN_END)
    feature_cols = build_feature_cols(panel)
    logger.info("Features: %d columns", len(feature_cols))

    models = train_models(panel, feature_cols)

    logger.info("Building test table with future covariate sequences...")
    test_table = build_test_table(panel, feature_cols)
    logger.info("Test table: %d region-months", len(test_table))

    all_results = []
    for h in HORIZONS:
        logger.info("---- Horizon t+%d ----", h)
        all_results.extend(run_horizon(models, test_table, h))

    out = compare_models(all_results)
    print("\n" + "=" * 120)
    print("MULTI-HORIZON RESULTS (2024 test set, iterating observed covariates)")
    print("=" * 120)
    core_cols = [
        "model", "horizon",
        "transition_detection_rate", "transition_accuracy",
        "f1_macro", "qwk", "hss",
        "rpss_vs_persistence", "bss_vs_persistence",
        "accuracy", "r2",
        "n_transitions", "n_total",
    ]
    cols = [c for c in core_cols if c in out.columns]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))

    # Save to processed directory
    out_path = PROCESSED_DIR / "multi_horizon_results.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved results -> %s", out_path)


if __name__ == "__main__":
    main()
