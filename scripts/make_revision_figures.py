#!/usr/bin/env python3
"""Generate figures for the revised paper.

Produces:
  fig_skill_vs_horizon.pdf      — Transition-detection & RPSS-vs-persistence
                                  vs horizon, aggregated across 4 test years
  fig_horizon_table.pdf         — Per-year × horizon heatmap
  fig_delta_confusion.pdf       — DeltaPredictor confusion matrix (2024 test)
  fig_boost_sensitivity.pdf     — transition_boost sweep curves

Usage:
    cd ~/Dropbox/school/probability/markov_famine
    venv/bin/python scripts/make_revision_figures.py
"""

from __future__ import annotations

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import PROCESSED_DIR

FIG_DIR = PROCESSED_DIR.parent.parent / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fig")


def fig_skill_vs_horizon() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "rolling_origin_results.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    models = ["Hybrid", "PhasePredictor", "DeltaPredictor", "Persistence"]
    colors = {"Hybrid": "#2b6cb0", "PhasePredictor": "#c05621", "DeltaPredictor": "#2f855a", "Persistence": "#718096"}
    markers = {"Hybrid": "o", "PhasePredictor": "s", "DeltaPredictor": "^", "Persistence": "x"}

    # Panel 1: transition detection
    ax = axes[0]
    for m in models:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        agg = sub.groupby("horizon")["transition_detection_rate"].agg(["median", "min", "max"]).reset_index()
        ax.plot(agg["horizon"], agg["median"], marker=markers[m], color=colors[m], label=m, linewidth=2)
        ax.fill_between(agg["horizon"], agg["min"], agg["max"], color=colors[m], alpha=0.15)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Transition detection rate")
    ax.set_title("(a) Transition detection (median; shaded = min/max across 4 test years)")
    ax.set_xticks([1, 3, 6])
    ax.set_ylim(-0.03, 0.85)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    # Panel 2: RPSS vs persistence
    ax = axes[1]
    for m in ["Hybrid", "PhasePredictor"]:
        sub = df[df["model"] == m].dropna(subset=["rpss_vs_persistence"])
        if sub.empty:
            continue
        agg = sub.groupby("horizon")["rpss_vs_persistence"].agg(["median", "min", "max"]).reset_index()
        ax.plot(agg["horizon"], agg["median"], marker=markers[m], color=colors[m], label=m, linewidth=2)
        ax.fill_between(agg["horizon"], agg["min"], agg["max"], color=colors[m], alpha=0.15)
    ax.axhline(0, color="#718096", linestyle="--", linewidth=1, label="Persistence reference")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("RPSS vs. persistence")
    ax.set_title("(b) Probabilistic skill vs. persistence (>0 = beats persistence)")
    ax.set_xticks([1, 3, 6])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "fig_skill_vs_horizon.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_horizon_table() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "rolling_origin_results.parquet")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    pivot = (
        df[df["model"] == "PhasePredictor"]
        .pivot(index="horizon", columns="test_year", values="transition_detection_rate")
    )
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"t+{h}" for h in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.45 else "black", fontsize=10)
    ax.set_title("PhasePredictor transition-detection rate by test year × horizon")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Horizon")
    plt.colorbar(im, ax=ax, label="Transition detection rate")
    plt.tight_layout()
    out = FIG_DIR / "fig_horizon_table.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_boost_sensitivity() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "transition_boost_sensitivity.parquet")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {
        "transition_detection_rate": "#2b6cb0",
        "f1_macro": "#c05621",
        "accuracy": "#718096",
        "crisis_recall": "#2f855a",
    }
    labels = {
        "transition_detection_rate": "Transition detection",
        "f1_macro": "F1-macro",
        "accuracy": "Accuracy",
        "crisis_recall": "Crisis recall (Phase 3+)",
    }
    df_sorted = df.sort_values("transition_boost")
    for col, color in colors.items():
        if col in df_sorted.columns:
            ax.plot(
                df_sorted["transition_boost"], df_sorted[col],
                marker="o", label=labels[col], color=color, linewidth=2,
            )
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 15, 50])
    ax.set_xticklabels([1, 5, 15, 50])
    ax.set_xlabel("transition_boost (log scale)")
    ax.set_ylabel("Metric value (2023 validation set)")
    ax.set_title("DeltaPredictor sensitivity to transition_boost")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "fig_boost_sensitivity.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def fig_phase_timeseries() -> None:
    """Mean phase time series per country — shows 2016-17 drought, 2022 peak, 2024 recovery."""
    panel = pd.read_parquet(PROCESSED_DIR / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["country"] = panel["region_code"].str[:2]

    by_country = (
        panel.groupby(["country", "date"])["ipc_phase"].mean().reset_index()
    )

    country_names = {"KE": "Kenya", "ET": "Ethiopia", "SO": "Somalia"}
    colors = {"KE": "#2b6cb0", "ET": "#c05621", "SO": "#2f855a"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for country, label in country_names.items():
        sub = by_country[by_country["country"] == country]
        ax.plot(sub["date"], sub["ipc_phase"], label=label, color=colors[country], linewidth=1.5)

    ax.axvspan(pd.Timestamp("2016-06-01"), pd.Timestamp("2017-06-01"),
               alpha=0.12, color="#e53e3e", label="2016-17 drought")
    ax.axvspan(pd.Timestamp("2020-10-01"), pd.Timestamp("2023-06-01"),
               alpha=0.12, color="#d69e2e", label="2020-23 multi-season drought")
    ax.axhline(3.0, color="#c53030", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(pd.Timestamp("2015-02-01"), 3.05, "Crisis threshold (Phase 3)",
            fontsize=8, color="#c53030")

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean IPC phase")
    ax.set_title("Mean IPC phase by country, 2015-2024 — study region covers 37 admin-1 regions")
    ax.set_ylim(1.5, 3.5)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "fig_phase_timeseries.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", out)


def main() -> None:
    fig_phase_timeseries()
    fig_skill_vs_horizon()
    fig_horizon_table()
    fig_boost_sensitivity()


if __name__ == "__main__":
    main()
