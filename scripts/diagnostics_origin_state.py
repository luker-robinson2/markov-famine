#!/usr/bin/env python3
"""Per-origin-state sample-size diagnostics for the NHMC row-wise classifiers.

Reports, for the training window:
    - Rows per origin state i (S_t = i)
    - Full empirical transition-count matrix (i -> j)
    - Non-zero delta rows per origin state (how many transitions each
      per-origin classifier actually learns from)

This addresses the peer review's concern that the Phase-4 (Emergency)
XGBoost row is fit on too few rows to be identified. The output feeds
paper §4.1 and the Limitations discussion of row-wise classifiers
(see Bartolucci & Farcomeni 2019; Meira-Machado 2009).

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/diagnostics_origin_state.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import TRAIN_END, PROCESSED_DIR


def main() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    print(f"Panel shape: {panel.shape}")
    print(f"Date range : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Training cutoff (TRAIN_END): {TRAIN_END}")

    train = (
        panel[panel["date"] <= TRAIN_END]
        .sort_values(["region_code", "date"])
        .copy()
    )
    train["next_phase"] = train.groupby("region_code")["ipc_phase"].shift(-1)
    train = train.dropna(subset=["next_phase"])
    train["next_phase"] = train["next_phase"].astype(int)
    train["delta"] = train["next_phase"] - train["ipc_phase"]

    print(f"\nTraining region-months with next-month target: {len(train):,}")
    print(f"  Non-zero delta (transition) rows: {(train['delta'] != 0).sum():,}")
    print(f"  Transition rate: {(train['delta'] != 0).mean() * 100:.2f}%")

    print("\n-- Rows per origin state S_t --")
    origin = train.groupby("ipc_phase").size()
    for phase, n in origin.items():
        print(f"  S_t = {phase}: {n:>5,} rows")

    print("\n-- Empirical transition count matrix (i -> j) --")
    trans_counts = (
        train.groupby(["ipc_phase", "next_phase"])
        .size()
        .unstack(fill_value=0)
    )
    print(trans_counts.to_string())

    print("\n-- Row-stochastic P̂ (empirical transition probs) --")
    trans_probs = trans_counts.div(trans_counts.sum(axis=1), axis=0).round(4)
    print(trans_probs.to_string())

    print("\n-- Non-zero deltas per origin state (training signal per classifier) --")
    nonzero = train[train["delta"] != 0].groupby("ipc_phase").size()
    for phase, n in nonzero.items():
        total = origin.get(phase, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  S_t = {phase}: {n:>4} non-zero / {total:>4} total ({pct:.1f}%)")

    print("\n-- Phase-5 observations in panel --")
    n_p5 = (panel["ipc_phase"] == 5).sum()
    print(f"  {n_p5} rows (current panel is 2015-2024; no Phase 5 expected)")


if __name__ == "__main__":
    main()
