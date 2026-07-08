"""Tests for src/explainability/visualisations.py."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure

from src.explainability.visualisations import (
    plot_global_importance,
    plot_per_regime_importance,
    plot_stability_heatmap,
)


def test_plot_global_importance_returns_figure_and_saves(tmp_path):
    importance = pd.Series({"a": 0.8, "b": 0.2, "c": 0.0}, name="mean_abs_shap")
    fig = plot_global_importance(importance)
    assert isinstance(fig, Figure)
    out = tmp_path / "global.png"
    fig.savefig(out, dpi=300)
    assert out.stat().st_size > 5000


def test_plot_per_regime_importance_returns_figure(tmp_path):
    importance = pd.DataFrame(
        {"Pre-GFC Stability": [0.8, 0.2, 0.0], "COVID-19 Shock": [1.4, 0.3, 0.0]},
        index=["a", "b", "c"],
    )
    fig = plot_per_regime_importance(importance)
    assert isinstance(fig, Figure)
    out = tmp_path / "per_regime.png"
    fig.savefig(out, dpi=300)
    assert out.stat().st_size > 5000


def test_plot_stability_heatmap_flags_small_regimes(tmp_path):
    regimes = ["Pre-GFC Stability", "Global Financial Crisis", "COVID-19 Shock"]
    matrix = pd.DataFrame(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], index=regimes, columns=regimes
    )
    small = {"Global Financial Crisis", "COVID-19 Shock"}
    fig = plot_stability_heatmap(matrix, small)
    assert isinstance(fig, Figure)
    # Small-sample regimes carry an asterisk on their tick labels.
    ax = fig.axes[0]
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    assert "Global Financial Crisis *" in xlabels
    assert "Pre-GFC Stability" in xlabels
    out = tmp_path / "heatmap.png"
    fig.savefig(out, dpi=300)
    assert out.stat().st_size > 5000
