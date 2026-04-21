"""Time series visualization with forecast fans and covariate overlays."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from typing import Optional

from src.config import IPC_COLORS, IPC_LABELS, N_STATES


def plot_ipc_timeseries(
    dates: pd.DatetimeIndex,
    phases: np.ndarray,
    region_name: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot IPC phase time series as colored step function.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Time points.
    phases : np.ndarray
        IPC phases (1-indexed) at each time point.
    region_name : str
        Region name for title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color each segment by phase
    for t in range(len(dates) - 1):
        phase = int(phases[t])
        color = IPC_COLORS.get(phase, "grey")
        ax.fill_between(
            [dates[t], dates[t + 1]],
            0, phase,
            color=color, alpha=0.7,
            step="post",
        )

    ax.step(dates, phases, where="post", color="black", linewidth=1, alpha=0.8)

    ax.set_ylim(0.5, 5.5)
    ax.set_yticks(range(1, N_STATES + 1))
    ax.set_yticklabels([f"{i}: {IPC_LABELS[i]}" for i in range(1, N_STATES + 1)])
    ax.set_ylabel("IPC Phase")
    ax.set_title(f"IPC Phase Timeline — {region_name}", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(axis="x", alpha=0.3)

    # Horizontal line at Phase 3 (crisis threshold)
    ax.axhline(y=2.5, color="red", linestyle="--", alpha=0.5, label="Crisis threshold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_forecast_fan(
    dates_history: pd.DatetimeIndex,
    phases_history: np.ndarray,
    dates_forecast: pd.DatetimeIndex,
    forecast_probs: np.ndarray,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    region_name: str = "",
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot historical IPC phases with probabilistic forecast fan.

    Parameters
    ----------
    dates_history : pd.DatetimeIndex
        Historical time points.
    phases_history : np.ndarray
        Historical IPC phases (1-indexed).
    dates_forecast : pd.DatetimeIndex
        Forecast time points.
    forecast_probs : np.ndarray
        Shape (horizon, n_states). Probability distribution at each step.
    ci_lower, ci_upper : np.ndarray, optional
        Shape (horizon, n_states). Confidence interval bounds.
    region_name : str
        Region name for title.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot history
    plot_ipc_timeseries(dates_history, phases_history, ax=ax)

    # Expected phase (weighted average)
    expected = np.sum(
        forecast_probs * np.arange(1, N_STATES + 1)[np.newaxis, :],
        axis=1,
    )

    # Plot forecast: shaded probability fans
    for phase in range(N_STATES):
        bottom = np.sum(forecast_probs[:, :phase], axis=1)
        ax.fill_between(
            dates_forecast,
            bottom + 0.5,
            bottom + forecast_probs[:, phase] + 0.5,
            color=IPC_COLORS[phase + 1],
            alpha=0.4,
            step="mid",
        )

    # Plot expected value line
    ax.plot(
        dates_forecast, expected,
        color="black", linewidth=2, linestyle="--",
        marker="o", markersize=4,
        label="Expected phase",
    )

    # Vertical line at forecast start
    ax.axvline(
        x=dates_forecast[0], color="grey", linestyle=":",
        linewidth=1.5, alpha=0.7,
    )
    ax.text(
        dates_forecast[0], 5.3, "Forecast start",
        ha="center", fontsize=9, color="grey",
    )

    ax.set_title(
        f"IPC Phase Forecast — {region_name}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_covariate_overlay(
    dates: pd.DatetimeIndex,
    phases: np.ndarray,
    covariate_values: np.ndarray,
    covariate_name: str = "SPEI-3",
    region_name: str = "",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> tuple:
    """Plot IPC phases with covariate overlay on secondary axis.

    Useful for visualizing the relationship between drought indices
    (SPEI, VCI) and food security outcomes.
    """
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # IPC phases on primary axis
    for t in range(len(dates) - 1):
        phase = int(phases[t])
        color = IPC_COLORS.get(phase, "grey")
        ax1.fill_between(
            [dates[t], dates[t + 1]], 0, phase,
            color=color, alpha=0.4, step="post",
        )
    ax1.step(dates, phases, where="post", color="black", linewidth=1, alpha=0.6)
    ax1.set_ylim(0.5, 5.5)
    ax1.set_yticks(range(1, N_STATES + 1))
    ax1.set_yticklabels([f"{i}: {IPC_LABELS[i]}" for i in range(1, N_STATES + 1)])
    ax1.set_ylabel("IPC Phase", fontsize=11)

    # Covariate on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        dates, covariate_values,
        color="navy", linewidth=1.5, alpha=0.8,
        label=covariate_name,
    )
    ax2.set_ylabel(covariate_name, fontsize=11, color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")

    # Zero line for standardized indices
    if "spei" in covariate_name.lower() or "spi" in covariate_name.lower():
        ax2.axhline(y=0, color="navy", linestyle=":", alpha=0.3)
        ax2.axhline(y=-1, color="red", linestyle="--", alpha=0.3, label="Moderate drought")
        ax2.axhline(y=-2, color="darkred", linestyle="--", alpha=0.3, label="Severe drought")

    ax1.set_title(
        f"{region_name}: IPC Phase vs {covariate_name}",
        fontsize=13, fontweight="bold",
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_leadtime_curves(
    leadtime_df: pd.DataFrame,
    metrics: list[str] = ["rpss", "crisis_recall", "accuracy"],
    title: str = "Forecast Skill by Lead Time",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot accuracy metrics as a function of forecast lead time.

    Parameters
    ----------
    leadtime_df : pd.DataFrame
        Output of leadtime_accuracy_curve(). Must have 'lead_time' column.
    metrics : list of str
        Column names to plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    markers = ["o", "s", "^", "D", "v"]
    for i, metric in enumerate(metrics):
        if metric in leadtime_df.columns:
            ax.plot(
                leadtime_df["lead_time"],
                leadtime_df[metric],
                marker=markers[i % len(markers)],
                linewidth=2,
                label=metric.replace("_", " ").title(),
            )

    ax.set_xlabel("Lead Time (months)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
