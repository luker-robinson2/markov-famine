#!/usr/bin/env python3
"""Rolling-origin cross-validation across 2021–2024 test years.

For each test year Y in {2021, 2022, 2023, 2024}:
    - Train on 2015..(Y-2), validate on Y-1, test on Y
    - Report skill metrics at horizons t+1, t+3, t+6

Addresses the peer-review concern that the original paper used only a
single (and relatively favorable) 2024 test year. Aggregating across
four test years — including the 2022 drought peak (crisis fraction
61%) and the 2021 La Niña onset — provides a much more credible
assessment of transition-detection skill.

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/rolling_origin_eval.py
"""

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.engineering.enhanced_features import build_enhanced_panel
from src.models.delta_model import DeltaPredictor, RegularizedPhasePredictor, HybridPredictor
from src.models.evaluation import evaluate_model, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rolling_origin")

TEST_YEARS = (2021, 2022, 2023, 2024)
HORIZONS = (1, 3, 6)


def build_feature_cols(panel: pd.DataFrame) -> list[str]:
    skip = {"region_code", "date", "country", "ipc_phase", "next_phase", "delta"}
    return [c for c in panel.columns if c not in skip]


def pad_missing_classes(
    X: np.ndarray, y_delta: np.ndarray, all_deltas=(-2, -1, 0, 1, 2), all_phases=(1, 2, 3, 4)
) -> tuple:
    """Append sentinel rows so every delta/phase class appears at least once.

    XGBoost requires class labels to span a contiguous 0..num_class-1 range.
    Shorter training windows can miss rare classes (e.g., delta = -2 is
    often absent from 5-year windows). Pad with one zero-feature row per
    missing class; weights downstream treat these as near-zero-weight via
    the existing transition_boost mechanism — but since zero-weight with
    XGBoost can drop rows, we use weight=1 for the padded rows and accept
    a tiny bias.
    """
    missing = [d for d in all_deltas if d not in set(y_delta)]
    if not missing:
        return X, y_delta
    pad_X = np.zeros((len(missing), X.shape[1]))
    pad_y = np.array(missing)
    return np.vstack([X, pad_X]), np.concatenate([y_delta, pad_y])


def pad_missing_phases(
    X: np.ndarray, y_phase: np.ndarray, all_phases=(1, 2, 3, 4)
) -> tuple:
    missing = [p for p in all_phases if p not in set(y_phase)]
    if not missing:
        return X, y_phase
    pad_X = np.zeros((len(missing), X.shape[1]))
    pad_y = np.array(missing)
    return np.vstack([X, pad_X]), np.concatenate([y_phase, pad_y])


def train_fold(panel: pd.DataFrame, feature_cols: list[str], train_end: str, val_start: str, val_end: str) -> dict:
    df = panel.sort_values(["region_code", "date"]).copy()
    df["next_phase"] = df.groupby("region_code")["ipc_phase"].shift(-1)
    df = df.dropna(subset=["next_phase"])
    df["next_phase"] = df["next_phase"].astype(int)
    df["delta"] = df["next_phase"] - df["ipc_phase"]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]

    X_tr = train[feature_cols].fillna(0).values
    y_tr_phase = train["next_phase"].values
    y_tr_delta = train["delta"].values
    X_va = val[feature_cols].fillna(0).values
    y_va_phase = val["next_phase"].values
    y_va_delta = val["delta"].values

    X_tr_d, y_tr_delta_pad = pad_missing_classes(X_tr, y_tr_delta)
    X_va_d, y_va_delta_pad = pad_missing_classes(X_va, y_va_delta)
    X_tr_p, y_tr_phase_pad = pad_missing_phases(X_tr, y_tr_phase)
    X_va_p, y_va_phase_pad = pad_missing_phases(X_va, y_va_phase)

    delta = DeltaPredictor().fit(X_tr_d, y_tr_delta_pad, X_va_d, y_va_delta_pad, transition_boost=15.0)
    phase = RegularizedPhasePredictor().fit(X_tr_p, y_tr_phase_pad - 1, X_va_p, y_va_phase_pad - 1)
    hybrid = HybridPredictor(delta_model=delta, phase_model=phase, delta_weight=0.6)
    return {"delta": delta, "phase": phase, "hybrid": hybrid}


def build_test_table(panel: pd.DataFrame, feature_cols: list[str], test_year: int, max_h: int = 6) -> pd.DataFrame:
    """Rows for start_date in year `test_year` with enough future data for horizon max_h."""
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")
    df = panel.sort_values(["region_code", "date"]).copy()
    out_rows = []
    for rc, grp in df.groupby("region_code"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for idx in range(len(grp)):
            start_date = grp.loc[idx, "date"]
            if start_date < test_start or start_date > test_end:
                continue
            if idx + max_h >= len(grp):
                continue
            row = {
                "region_code": rc,
                "start_date": start_date,
                "S_t": int(grp.loc[idx, "ipc_phase"]),
            }
            for k in range(1, max_h + 1):
                row[f"S_tph_{k}"] = int(grp.loc[idx + k, "ipc_phase"])
                row[f"X_tph_{k - 1}"] = np.asarray(grp.loc[idx + k - 1, feature_cols].fillna(0).values, dtype=float)
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def forecast_one(hybrid: HybridPredictor, start_state: int, cov_seq: list[np.ndarray], n_states: int = 4) -> np.ndarray:
    pi = np.zeros(n_states)
    pi[start_state - 1] = 1.0
    traj = np.zeros((len(cov_seq), n_states))
    for k, X in enumerate(cov_seq):
        P = np.zeros((n_states, n_states))
        X2 = X.reshape(1, -1)
        for i in range(n_states):
            P[i] = hybrid.transition_row(X2, origin_state=i + 1)
        pi = pi @ P
        traj[k] = pi
    return traj


def run_fold(
    models: dict,
    test_table: pd.DataFrame,
    horizons: tuple,
    test_year: int,
    n_states: int = 4,
) -> list[dict]:
    hybrid = models["hybrid"]
    phase = models["phase"]
    delta = models["delta"]

    current = test_table["S_t"].to_numpy()
    results = []
    for h in horizons:
        y_true = test_table[f"S_tph_{h}"].to_numpy()

        # Hybrid NHMC
        pred_h = np.zeros(len(test_table), dtype=int)
        proba_h = np.zeros((len(test_table), n_states))
        for i, row in enumerate(test_table.itertuples(index=False)):
            cov_seq = [getattr(row, f"X_tph_{k}") for k in range(h)]
            traj = forecast_one(hybrid, int(getattr(row, "S_t")), cov_seq, n_states=n_states)
            pi_h = traj[-1]
            pred_h[i] = int(np.argmax(pi_h)) + 1
            proba_h[i] = pi_h

        # PhasePredictor direct on X_{t+h-1}
        X_h = np.stack([row[f"X_tph_{h - 1}"] for _, row in test_table.iterrows()])
        pred_p = phase.predict(X_h) + 1
        try:
            proba_p = phase.predict_proba(X_h)
        except Exception:
            proba_p = None

        # DeltaPredictor iterated
        pred_d = current.copy()
        for k in range(h):
            X_k = np.stack([row[f"X_tph_{k}"] for _, row in test_table.iterrows()])
            pred_d = delta.predict_phase(X_k, pred_d)

        for name, pred, proba in [
            (f"Hybrid", pred_h, proba_h),
            (f"PhasePredictor", pred_p, proba_p),
            (f"DeltaPredictor", pred_d, None),
            (f"Persistence", current.copy(), None),
        ]:
            res = evaluate_model(
                y_true_phases=y_true,
                y_pred_phases=pred,
                y_pred_proba=proba,
                current_phases=current,
                model_name=name,
                n_states=n_states,
            )
            res["test_year"] = test_year
            res["horizon"] = h
            results.append(res)
    return results


def main() -> None:
    logger.info("Loading panel...")
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")

    all_results = []
    for test_year in TEST_YEARS:
        train_end = f"{test_year - 2}-12-31"
        val_start = f"{test_year - 1}-01-01"
        val_end = f"{test_year - 1}-12-31"
        logger.info(
            "============ FOLD test_year=%d (train ≤ %s, val %s..%s) ============",
            test_year, train_end, val_start, val_end,
        )
        enhanced = build_enhanced_panel(panel, train_end=train_end)
        feature_cols = build_feature_cols(enhanced)

        models = train_fold(enhanced, feature_cols, train_end, val_start, val_end)
        test_table = build_test_table(enhanced, feature_cols, test_year, max_h=max(HORIZONS))
        logger.info("  Test table: %d region-months", len(test_table))
        if len(test_table) == 0:
            logger.warning("  Empty test table — skipping")
            continue
        fold_results = run_fold(models, test_table, HORIZONS, test_year)
        all_results.extend(fold_results)

    out = compare_models(all_results)
    print("\n" + "=" * 130)
    print("ROLLING-ORIGIN × MULTI-HORIZON RESULTS")
    print("=" * 130)
    cols = [
        "model", "test_year", "horizon",
        "transition_detection_rate",
        "f1_macro", "qwk", "hss",
        "rpss_vs_persistence", "bss_vs_persistence",
        "accuracy", "n_transitions", "n_total",
    ]
    cols = [c for c in cols if c in out.columns]
    print(
        out[cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v),
        )
    )

    # Aggregate across folds
    print("\n" + "=" * 130)
    print("AGGREGATE ACROSS TEST YEARS (median, IQR)")
    print("=" * 130)
    numeric_cols = ["transition_detection_rate", "f1_macro", "qwk", "hss", "rpss_vs_persistence", "bss_vs_persistence", "accuracy"]
    for model in ["Hybrid", "PhasePredictor", "DeltaPredictor", "Persistence"]:
        for h in HORIZONS:
            sub = out[(out["model"] == model) & (out["horizon"] == h)]
            if sub.empty:
                continue
            pieces = [f"{model:15s} t+{h}:"]
            for col in numeric_cols:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    if not vals.empty:
                        med = vals.median()
                        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                        pieces.append(f"{col}={med:.3f}[{q1:.3f},{q3:.3f}]")
            print("  " + " ".join(pieces))

    out_path = PROCESSED_DIR / "rolling_origin_results.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Saved -> %s", out_path)


if __name__ == "__main__":
    main()
