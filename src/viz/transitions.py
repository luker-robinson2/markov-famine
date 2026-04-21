"""Transition matrix visualization: heatmaps and comparison plots."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from src.config import IPC_LABELS, IPC_COLORS, N_STATES


def plot_transition_heatmap(
    P: np.ndarray,
    title: str = "Transition Matrix",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
    annot_fmt: str = ".2f",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot a transition matrix as an annotated heatmap.

    Parameters
    ----------
    P : np.ndarray
        Shape (n_states, n_states) transition matrix.
    title : str
        Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    labels = [f"Phase {i}" for i in range(1, P.shape[0] + 1)]

    sns.heatmap(
        P,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Probability"},
        linewidths=0.5,
    )

    ax.set_xlabel("Next Phase (t+1)", fontsize=12)
    ax.set_ylabel("Current Phase (t)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_transition_comparison(
    P_baseline: np.ndarray,
    P_nhmc: np.ndarray,
    title_baseline: str = "Homogeneous MC",
    title_nhmc: str = "Non-Homogeneous MC (Drought)",
    suptitle: str = "Transition Matrix Comparison",
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
) -> tuple:
    """Side-by-side comparison of two transition matrices."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    plot_transition_heatmap(P_baseline, title=title_baseline, ax=ax1)
    plot_transition_heatmap(P_nhmc, title=title_nhmc, ax=ax2)

    # Difference matrix
    diff = P_nhmc - P_baseline
    labels = [f"Phase {i}" for i in range(1, P_baseline.shape[0] + 1)]

    sns.heatmap(
        diff,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax3,
        cbar_kws={"label": "Difference"},
        linewidths=0.5,
    )
    ax3.set_xlabel("Next Phase (t+1)")
    ax3.set_ylabel("Current Phase (t)")
    ax3.set_title("Difference (NHMC - Baseline)", fontsize=14, fontweight="bold")

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2, ax3)


def plot_stationary_distribution(
    distributions: dict[str, np.ndarray],
    title: str = "Stationary Distribution Comparison",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Bar chart comparing stationary distributions across scenarios.

    Parameters
    ----------
    distributions : dict
        Maps scenario name -> (n_states,) stationary distribution.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    n_scenarios = len(distributions)
    x = np.arange(N_STATES)
    width = 0.8 / n_scenarios

    for i, (name, dist) in enumerate(distributions.items()):
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            dist[:N_STATES],
            width,
            label=name,
            alpha=0.8,
        )

    ax.set_xlabel("IPC Phase", fontsize=12)
    ax.set_ylabel("Stationary Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Phase {i+1}\n{IPC_LABELS[i+1]}" for i in range(N_STATES)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_transition_counts(
    counts: np.ndarray,
    title: str = "Observed Transition Counts",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot observed transition count matrix."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    labels = [f"Phase {i}" for i in range(1, counts.shape[0] + 1)]

    sns.heatmap(
        counts.astype(int),
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )

    ax.set_xlabel("Next Phase (t+1)")
    ax.set_ylabel("Current Phase (t)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
