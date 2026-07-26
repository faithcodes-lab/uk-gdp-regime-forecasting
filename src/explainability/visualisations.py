"""Publication figures for the regime-aware SHAP analysis.

Each public function returns a matplotlib Figure; the caller saves it.
The figures report the analysis as it is: the global importance bar and
the per-regime heatmap show that the model concentrates on a small set
of GDP-history features, and the stability heatmap shows the resulting
cross-regime rank agreement. Nothing is added to make the result look
richer than it is.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

_STABLE = 0.6
_MODERATE = 0.3


def plot_global_importance(importance: pd.Series) -> Figure:
    """Horizontal bar chart of global mean absolute SHAP per feature, most important at the top."""
    ordered = importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ordered.index, ordered.values, color="tab:blue")
    ax.set_xlabel("Mean absolute SHAP value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Global SHAP feature importance", fontsize=14)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_regime_importance(importance: pd.DataFrame) -> Figure:
    """Heatmap of mean absolute SHAP per feature (rows) within each regime (columns)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(importance.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(importance.columns)))
    ax.set_xticklabels(importance.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(importance.index)))
    ax.set_yticklabels(importance.index, fontsize=8)
    ax.set_title("Per-regime SHAP feature importance (mean absolute SHAP)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Mean absolute SHAP value")
    fig.tight_layout()
    return fig


def plot_stability_heatmap(matrix: pd.DataFrame, small_regimes: set[str]) -> Figure:
    """6x6 Spearman heatmap coloured by the Akoglu bands, small-sample regimes flagged with an asterisk."""
    labels = [f"{r} *" if r in small_regimes else r for r in matrix.columns]
    # Bands: unstable (<= 0.3) red, moderately stable (0.3 to 0.6) orange, stable (> 0.6) green.
    cmap = ListedColormap(["#d73027", "#fdae61", "#1a9850"])
    norm = BoundaryNorm([-1.0, _MODERATE, _STABLE, 1.0], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.values, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
    ax.set_title("SHAP ranking stability across regimes (Spearman rho)", fontsize=13)
    cbar = fig.colorbar(
        im, ax=ax, ticks=[(-1 + _MODERATE) / 2, (_MODERATE + _STABLE) / 2, (_STABLE + 1) / 2]
    )
    cbar.ax.set_yticklabels(["unstable\n(<= 0.3)", "moderate\n(0.3 to 0.6)", "stable\n(> 0.6)"])
    # An asterisk on a regime label means a small-sample regime (n = 6); read those rows and
    # columns with the bootstrap confidence intervals in results/shap/bootstrap_cis.csv.
    fig.tight_layout()
    return fig
