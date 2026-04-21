#!/usr/bin/env python3
"""Compare one-hot vs. model-predicted π_t initialization in NHMC forecasts.

The peer review flagged π_t = one-hot on observed S_t as a tidy
simplification that understates initial-state uncertainty. This
compares the one-hot forecast against one that initializes π_t with
the PhasePredictor's own distribution over S_t (given X_{t-1}). Small
differences at t+1 are expected; we focus on t+3 and t+6 where
compounding may amplify.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/init_pi_comparison.py
"""

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END
from src.engineering.enhanced_features import build_enhanced_panel
from src.models.delta_model import DeltaPredictor, RegularizedPhasePredictor, HybridPredictor
from src.models.evaluation import evaluate_model, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_pi")

HORIZONS = (1, 3, 6)
N_STATES = 4


def feature_cols(panel: pd.DataFrame) -> list[str]:
    skip = {"region_code", "date", "country", "ipc_phase", "next_phase", "delta"}
    return [c for c in panel.columns if c not in skip]


def forecast_traj(
    hybrid: HybridPredictor, pi0: np.ndarray, cov_seq: list[np.ndarray]
) -> np.ndarray:
    pi = pi0.copy()
    traj = np.zeros((len(cov_seq), N_STATES))
    for k, X in enumerate(cov_seq):
        P = np.zeros((N_STATES, N_STATES))
        X2 = X.reshape(1, -1)
        for i in range(N_STATES):
            P[i] = hybrid.transition_row(X2, origin_state=i + 1)
        pi = pi @ P
        traj[k] = pi
    return traj


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    panel = build_enhanced_panel(panel, train_end=TRAIN_END)
    cols = feature_cols(panel)

    df = panel.sort_values(["region_code", "date"]).copy()
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)
    df["delta"] = df["next_phase"] - df["ipc_phase"]

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VALID_START) & (df["date"] <= VALID_END)]

    X_tr = train[cols].fillna(0).values
    X_va = val[cols].fillna(0).values

    delta = DeltaPredictor().fit(
        X_tr, train["delta"].values, X_va, val["delta"].values, transition_boost=15.0,
    )
    phase = RegularizedPhasePredictor().fit(
        X_tr, train["next_phase"].values - 1, X_va, val["next_phase"].values - 1,
    )
    hybrid = HybridPredictor(delta_model=delta, phase_model=phase, delta_weight=0.6)

    # Build test table
    max_h = max(HORIZONS)
    rows = []
    for rc, grp in df.groupby("region_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for idx in range(len(grp)):
            start = grp.loc[idx, "date"]
            if start < pd.Timestamp(TEST_START) or start > pd.Timestamp(TEST_END):
                continue
            if idx + max_h >= len(grp) or idx == 0:
                continue  # need X_{t-1} for predicted-π_t init
            rec = {
                "S_t": int(grp.loc[idx, "ipc_phase"]),
                "X_prev": np.asarray(grp.loc[idx - 1, cols].fillna(0).values, dtype=float),
            }
            for k in range(1, max_h + 1):
                rec[f"S_tph_{k}"] = int(grp.loc[idx + k, "ipc_phase"])
                rec[f"X_tph_{k - 1}"] = np.asarray(grp.loc[idx + k - 1, cols].fillna(0).values, dtype=float)
            rows.append(rec)
    tt = pd.DataFrame(rows)
    logger.info("Test table: %d rows", len(tt))

    current = tt["S_t"].to_numpy()

    all_results = []
    for h in HORIZONS:
        y_true = tt[f"S_tph_{h}"].to_numpy()

        # ---- One-hot init ----
        pred_oh = np.zeros(len(tt), dtype=int)
        proba_oh = np.zeros((len(tt), N_STATES))
        # ---- Predicted-π init (from PhasePredictor on X_{t-1}) ----
        pred_pp = np.zeros(len(tt), dtype=int)
        proba_pp = np.zeros((len(tt), N_STATES))

        for i, row in enumerate(tt.itertuples(index=False)):
            cov_seq = [getattr(row, f"X_tph_{k}") for k in range(h)]

            pi_oh = np.zeros(N_STATES)
            pi_oh[int(getattr(row, "S_t")) - 1] = 1.0

            X_prev = getattr(row, "X_prev").reshape(1, -1)
            pi_pp = phase.predict_proba(X_prev)[0]

            traj_oh = forecast_traj(hybrid, pi_oh, cov_seq)
            traj_pp = forecast_traj(hybrid, pi_pp, cov_seq)

            proba_oh[i] = traj_oh[-1]
            proba_pp[i] = traj_pp[-1]
            pred_oh[i] = int(np.argmax(proba_oh[i])) + 1
            pred_pp[i] = int(np.argmax(proba_pp[i])) + 1

        for name, pred, proba in [
            (f"OneHot π_t t+{h}", pred_oh, proba_oh),
            (f"Predicted π_t t+{h}", pred_pp, proba_pp),
        ]:
            res = evaluate_model(
                y_true_phases=y_true,
                y_pred_phases=pred,
                y_pred_proba=proba,
                current_phases=current,
                model_name=name,
            )
            res["horizon"] = h
            all_results.append(res)

    out = compare_models(all_results)
    print("\n" + "=" * 110)
    print("INITIALIZATION COMPARISON (Hybrid NHMC, 2024 test set)")
    print("=" * 110)
    show = [
        "model", "horizon",
        "transition_detection_rate",
        "f1_macro", "qwk",
        "rpss_vs_persistence", "bss_vs_persistence",
        "accuracy",
    ]
    show = [c for c in show if c in out.columns]
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))


if __name__ == "__main__":
    main()
