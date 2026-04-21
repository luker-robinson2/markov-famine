#!/usr/bin/env python3
"""Sensitivity sweep over DeltaPredictor.transition_boost.

The paper defaults transition_boost=15.0 without justification. This
script trains the DeltaPredictor with boost ∈ {1, 5, 15, 50} and
evaluates on the **validation set only** (2023) to avoid test-set
tuning. Reports F1-macro, QWK, RPSS-vs-persistence, and transition
detection rate to justify (or update) the published choice.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/sensitivity_transition_boost.py
"""

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TRAIN_END, VALID_START, VALID_END
from src.engineering.enhanced_features import build_enhanced_panel
from src.models.delta_model import DeltaPredictor
from src.models.evaluation import evaluate_model, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sensitivity")

BOOSTS = (1.0, 5.0, 15.0, 50.0)


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    panel = build_enhanced_panel(panel, train_end=TRAIN_END)

    skip = {"region_code", "date", "country", "ipc_phase"}
    feature_cols = [c for c in panel.columns if c not in skip]

    df = panel.sort_values(["region_code", "date"]).copy()
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)
    df["delta"] = df["next_phase"] - df["ipc_phase"]

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VALID_START) & (df["date"] <= VALID_END)]

    X_tr = train[feature_cols].fillna(0).values
    y_tr_delta = train["delta"].values

    X_va = val[feature_cols].fillna(0).values
    y_va_delta = val["delta"].values
    y_va_phase = val["next_phase"].values
    cur_va = val["ipc_phase"].values

    all_results = []
    for boost in BOOSTS:
        logger.info("Training with transition_boost=%.1f ...", boost)
        delta = DeltaPredictor().fit(
            X_tr, y_tr_delta, X_va, y_va_delta, transition_boost=boost,
        )
        pred = delta.predict_phase(X_va, cur_va)
        res = evaluate_model(
            y_true_phases=y_va_phase,
            y_pred_phases=pred,
            current_phases=cur_va,
            model_name=f"boost={boost:g}",
        )
        res["transition_boost"] = boost
        all_results.append(res)

    out = compare_models(all_results)
    print("\n" + "=" * 120)
    print("transition_boost SENSITIVITY (2023 VALIDATION SET)")
    print("=" * 120)
    cols = [
        "model", "transition_boost",
        "transition_detection_rate", "transition_accuracy",
        "f1_macro", "qwk", "hss",
        "accuracy", "r2",
        "crisis_recall", "crisis_f1",
        "n_transitions",
    ]
    cols = [c for c in cols if c in out.columns]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)))

    # Save
    out_path = PROCESSED_DIR / "transition_boost_sensitivity.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
