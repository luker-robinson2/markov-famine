"""SHAP (SHapley Additive exPlanations) visualization for ensemble models.

Provides interpretability for the XGBoost/LightGBM/CatBoost models that
parameterize the Markov chain transition probabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional

from src.config import N_STATES, IPC_LABELS


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> shap.Explanation:
    """Compute SHAP values for a trained model.

    Parameters
    ----------
    model : trained sklearn-compatible model
        Must have predict or predict_proba method.
    X : np.ndarray
        Feature matrix.
    feature_names : list of str, optional
        Feature names for display.

    Returns
    -------
    shap.Explanation
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    if feature_names is not None:
        shap_values.feature_names = feature_names

    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    title: str = "SHAP Feature Importance",
    max_display: int = 20,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """SHAP beeswarm summary plot.

    Shows the distribution of SHAP values for each feature across samples.
    Features are ranked by mean absolute SHAP value.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    shap.summary_plot(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_shap_bar(
    shap_values: shap.Explanation,
    title: str = "Mean |SHAP| Feature Importance",
    max_display: int = 20,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Bar plot of mean absolute SHAP values."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature: str,
    interaction_feature: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """SHAP dependence plot for a single feature.

    Shows how the SHAP value of a feature varies with the feature value,
    colored by an interaction feature.

    Parameters
    ----------
    feature : str
        Feature name to plot.
    interaction_feature : str, optional
        Feature for color coding. Auto-detected if None.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    shap.dependence_plot(
        feature,
        shap_values.values if hasattr(shap_values, 'values') else shap_values,
        features=shap_values.data if hasattr(shap_values, 'data') else None,
        feature_names=shap_values.feature_names if hasattr(shap_values, 'feature_names') else None,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_shap_per_state(
    models: dict[int, object],
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
) -> None:
    """SHAP importance comparison across origin states.

    Shows which features matter most for each origin IPC phase,
    highlighting that different covariates drive transitions from
    different phases.

    Parameters
    ----------
    models : dict
        Maps origin_state (1-indexed) -> trained model.
    X : np.ndarray
        Feature matrix.
    feature_names : list of str
        Feature names.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for idx, (state, model) in enumerate(sorted(models.items())):
        ax = axes[idx]
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X)
            mean_abs_shap = np.abs(sv.values).mean(axis=0)

            # Handle multi-output (get mean across outputs)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.mean(axis=1)

            top_indices = np.argsort(mean_abs_shap)[-max_display:]

            ax.barh(
                range(len(top_indices)),
                mean_abs_shap[top_indices],
                color=list(IPC_COLORS.values())[min(state - 1, N_STATES - 1)],
                alpha=0.8,
            )
            ax.set_yticks(range(len(top_indices)))
            ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
            ax.set_title(f"From Phase {state}\n({IPC_LABELS[state]})", fontsize=11)
            ax.set_xlabel("Mean |SHAP|")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(f"Phase {state}")

    fig.suptitle(
        "Feature Importance by Origin IPC Phase",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
