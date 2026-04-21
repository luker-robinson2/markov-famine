"""Choropleth maps for IPC food security phase visualization.

Uses the official IPC color palette:
    Phase 1 (Minimal):   #C6FECE (light green)
    Phase 2 (Stressed):  #FAE61E (yellow)
    Phase 3 (Crisis):    #E67800 (orange)
    Phase 4 (Emergency): #C80000 (red)
    Phase 5 (Famine):    #640000 (dark maroon)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Optional

from src.config import IPC_COLORS, IPC_LABELS, N_STATES


def get_ipc_cmap():
    """Create a discrete colormap using the official IPC palette."""
    colors = [IPC_COLORS[i] for i in range(1, N_STATES + 1)]
    cmap = mcolors.ListedColormap(colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def ipc_legend_handles():
    """Create legend handles for IPC phases."""
    return [
        Patch(facecolor=IPC_COLORS[i], edgecolor="black", linewidth=0.5,
              label=f"Phase {i}: {IPC_LABELS[i]}")
        for i in range(1, N_STATES + 1)
    ]


def plot_ipc_map(
    gdf,
    phase_column: str = "ipc_phase",
    title: str = "IPC Acute Food Insecurity Classification",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (12, 10),
    show_legend: bool = True,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot choropleth map of IPC phases.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have geometry and a column with IPC phases (1-5).
    phase_column : str
        Column name containing IPC phase values.
    title : str
        Map title.
    ax : plt.Axes, optional
        Existing axes to plot on.
    figsize : tuple
        Figure size if creating new figure.
    show_legend : bool
        Whether to show IPC phase legend.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap, norm = get_ipc_cmap()

    gdf.plot(
        column=phase_column,
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    if show_legend:
        handles = ipc_legend_handles()
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=9,
            title="IPC Phase",
            frameon=True,
            framealpha=0.9,
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def plot_predicted_vs_actual(
    gdf,
    actual_column: str = "ipc_phase",
    predicted_column: str = "predicted_phase",
    title_actual: str = "Observed IPC Phase",
    title_predicted: str = "Predicted IPC Phase",
    suptitle: str = "",
    figsize: tuple = (20, 10),
    save_path: Optional[str] = None,
) -> tuple:
    """Side-by-side choropleth maps comparing actual and predicted phases.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have geometry, actual_column, and predicted_column.

    Returns
    -------
    tuple of (fig, (ax1, ax2))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_ipc_map(gdf, phase_column=actual_column, title=title_actual, ax=ax1, show_legend=False)
    plot_ipc_map(gdf, phase_column=predicted_column, title=title_predicted, ax=ax2, show_legend=False)

    # Shared legend
    handles = ipc_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=N_STATES,
        fontsize=10,
        title="IPC Phase",
        bbox_to_anchor=(0.5, 0.02),
    )

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_crisis_probability_map(
    gdf,
    prob_column: str = "crisis_probability",
    title: str = "Probability of Phase 3+ (Crisis or Worse)",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Plot continuous probability of crisis (Phase 3+).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must have geometry and a column with P(Phase >= 3).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf.plot(
        column=prob_column,
        cmap="YlOrRd",
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        legend_kwds={"label": "P(Phase 3+)", "shrink": 0.6},
        vmin=0,
        vmax=1,
        missing_kwds={"color": "lightgrey"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_axis_off()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax
