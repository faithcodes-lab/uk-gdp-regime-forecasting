"""Publication figures for the evaluation: model comparison bar, regime heatmap, fold variance.

Each public function returns a matplotlib Figure; the caller saves it to
disk. CP7 writes PNG at 300 dpi and a vector PDF for LaTeX inclusion.
Palette is viridis for the heatmap (continuous data) and tab10 for the
bar and box plots (4 categorical models). Titles, axis labels, and the
small-sample annotation are self-contained so the figures read on their
own.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

_MODEL_ORDER = ["arima", "ridge", "xgboost", "lightgbm"]
_METRIC_LABELS = {"rmse": "RMSE", "mae": "MAE", "mase": "MASE", "r2": "R2"}
_METRIC_UNITS = {
    "rmse": "% quarter on quarter",
    "mae": "% quarter on quarter",
    "mase": "unitless",
    "r2": "unitless",
}


def _ordered_models(present: pd.Index | pd.Series) -> list[str]:
    """Returns the model names that appear in present, in the canonical order."""
    present_set = set(present)
    return [m for m in _MODEL_ORDER if m in present_set]


def plot_model_comparison_bar(
    aggregated_results: pd.DataFrame, scheme: str, metric: str = "rmse"
) -> Figure:
    """Horizontal bar chart of one metric across the four models, with std error bars."""
    filtered = aggregated_results[aggregated_results["scheme"] == scheme].set_index("model")
    filtered = filtered.reindex(_ordered_models(filtered.index))

    means = filtered[f"mean_{metric}"]
    stds = filtered[f"std_{metric}"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(filtered.index, means, xerr=stds, color="tab:blue", capsize=4)
    ax.set_xlabel(f"{_METRIC_LABELS[metric]} ({_METRIC_UNITS[metric]})", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Model comparison: {_METRIC_LABELS[metric]} on {scheme.replace('_', '-')} CV",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_regime_performance_heatmap(
    per_regime_results: pd.DataFrame, scheme: str, metric: str = "rmse"
) -> Figure:
    """Heatmap of one metric across (model, regime) for one CV scheme."""
    filtered = per_regime_results[per_regime_results["scheme"] == scheme]
    pivot = filtered.pivot(index="model", columns="regime", values=metric)
    pivot = pivot.reindex(_ordered_models(pivot.index))

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Regime", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Per-regime {_METRIC_LABELS[metric]} ({scheme.replace('_', '-')} CV)",
        fontsize=14,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{_METRIC_LABELS[metric]} ({_METRIC_UNITS[metric]})", fontsize=12)

    small_sample_regimes = filtered.loc[filtered["small_sample"], "regime"].unique()
    if len(small_sample_regimes) > 0:
        regimes_str = ", ".join(small_sample_regimes)
        fig.text(
            0.5,
            0.02,
            f"Small sample (n < 10 quarters): {regimes_str}",
            ha="center",
            fontsize=9,
            style="italic",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    return fig


def plot_cv_fold_variance(
    per_fold_metrics: pd.DataFrame, scheme: str, metric: str = "rmse"
) -> Figure:
    """Box plot of one metric across folds, one box per model."""
    filtered = per_fold_metrics[per_fold_metrics["scheme"] == scheme]
    models = _ordered_models(filtered["model"].unique())
    grouped = [filtered.loc[filtered["model"] == m, metric].to_numpy() for m in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        grouped,
        tick_labels=models,
        orientation="vertical",
        patch_artist=True,
        boxprops={"facecolor": "tab:blue", "alpha": 0.5},
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(f"{_METRIC_LABELS[metric]} ({_METRIC_UNITS[metric]})", fontsize=12)
    ax.set_title(
        f"Per-fold {_METRIC_LABELS[metric]} distribution ({scheme.replace('_', '-')} CV)",
        fontsize=14,
    )
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
