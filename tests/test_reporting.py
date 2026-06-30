"""Tests for src/reporting/tables.py and src/reporting/figures.py.

Tests that matter most: test_per_regime_table_marks_small_sample_regimes
(footnote flag on small samples), test_dm_table_formats_significant_p_value_with_three_decimals
and test_dm_table_shows_p_below_0001_as_less_than (p-value formatting,
including the < 0.001 threshold), and test_figures_save_at_300_dpi_png_and_pdf
(figure persistence at correct dpi and format).
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.reporting.figures import (
    plot_cv_fold_variance,
    plot_model_comparison_bar,
    plot_regime_performance_heatmap,
)
from src.reporting.tables import (
    make_dm_test_table,
    make_overall_performance_table,
    make_per_regime_table,
)


def _synthetic_aggregated_results() -> pd.DataFrame:
    """Synthetic aggregate DataFrame matching CP5's aggregate_cv_results output."""
    rows = []
    for model in ["ridge", "xgboost", "lightgbm", "arima"]:
        for scheme in ["expanding_window", "regime_aligned"]:
            row = {"model": model, "scheme": scheme, "n_folds": 8}
            for metric in ("rmse", "mae", "mase", "r2"):
                row[f"mean_{metric}"] = 0.5
                row[f"std_{metric}"] = 0.05
                row[f"median_{metric}"] = 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_per_regime_results() -> pd.DataFrame:
    """Synthetic per-regime DataFrame matching CP4's evaluate_per_regime output."""
    regimes = [
        ("Pre-GFC", 33, False),
        ("GFC", 6, True),
        ("Post-GFC", 27, False),
        ("Brexit", 14, False),
        ("COVID", 6, True),
        ("Post-COVID", 18, False),
    ]
    rows = []
    for model in ["ridge", "xgboost", "lightgbm", "arima"]:
        for scheme in ["expanding_window", "regime_aligned"]:
            for regime, n, small in regimes:
                rows.append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "regime": regime,
                        "n_observations": n,
                        "rmse": 0.4,
                        "mae": 0.3,
                        "mase": 0.85,
                        "r2": 0.32,
                        "small_sample": small,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_per_fold_metrics() -> pd.DataFrame:
    """Synthetic per-fold DataFrame matching CP5's compute_per_fold_metrics output."""
    rows = []
    for model in ["ridge", "xgboost", "lightgbm", "arima"]:
        for scheme in ["expanding_window", "regime_aligned"]:
            n_folds = 8 if scheme == "expanding_window" else 5
            for fold in range(1, n_folds + 1):
                rows.append(
                    {
                        "model": model,
                        "scheme": scheme,
                        "fold_idx": fold,
                        "n_observations": 4,
                        "rmse": 0.4 + 0.01 * fold,
                        "mae": 0.3 + 0.01 * fold,
                        "mase": 0.85 + 0.01 * fold,
                        "r2": 0.3 + 0.01 * fold,
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_dm_results() -> pd.DataFrame:
    """Synthetic DM results DataFrame for testing."""
    pairs = [
        ("ridge", "xgboost"),
        ("ridge", "lightgbm"),
        ("ridge", "arima"),
        ("xgboost", "lightgbm"),
        ("xgboost", "arima"),
        ("lightgbm", "arima"),
    ]
    rows = []
    for i, (a, b) in enumerate(pairs):
        for scheme in ["expanding_window", "regime_aligned"]:
            p_raw = 0.023 if i == 0 else 0.5
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "scheme": scheme,
                    "statistic": -1.5 + 0.3 * i,
                    "p_value": p_raw,
                    "p_value_bonferroni": min(p_raw * 6, 1.0),
                    "n_observations": 32,
                }
            )
    return pd.DataFrame(rows)


def test_overall_table_returns_dict_with_markdown_and_latex():
    """Both keys present, both values non-empty strings."""
    agg = _synthetic_aggregated_results()
    result = make_overall_performance_table(agg, scheme="expanding_window")
    assert set(result.keys()) == {"markdown", "latex"}
    assert len(result["markdown"]) > 0
    assert len(result["latex"]) > 0


def test_overall_table_includes_all_models():
    """The markdown lists all four model names."""
    agg = _synthetic_aggregated_results()
    result = make_overall_performance_table(agg, scheme="expanding_window")
    for model in ["ridge", "xgboost", "lightgbm", "arima"]:
        assert model in result["markdown"]


def test_overall_table_rounds_metrics_to_three_decimal_places():
    """A synthetic 0.5 metric renders as '0.500' in the markdown."""
    agg = _synthetic_aggregated_results()
    result = make_overall_performance_table(agg, scheme="expanding_window")
    assert "0.500" in result["markdown"]


def test_per_regime_table_marks_small_sample_regimes():
    """Small-sample regimes (GFC, COVID) carry an asterisk and the footnote text appears."""
    per_regime = _synthetic_per_regime_results()
    result = make_per_regime_table(per_regime, scheme="expanding_window")
    assert "GFC*" in result["markdown"]
    assert "COVID*" in result["markdown"]
    assert "Small sample" in result["markdown"]


def test_per_regime_table_does_not_mark_large_regimes():
    """Large regimes (no small_sample flag) have no asterisk."""
    per_regime = _synthetic_per_regime_results()
    result = make_per_regime_table(per_regime, scheme="expanding_window")
    assert "Pre-GFC*" not in result["markdown"]
    assert "Post-GFC*" not in result["markdown"]
    assert "Brexit*" not in result["markdown"]
    assert "Post-COVID*" not in result["markdown"]


def test_dm_table_formats_significant_p_value_with_three_decimals():
    """A p-value of 0.023 renders as '0.023' in the markdown."""
    dm = _synthetic_dm_results()
    result = make_dm_test_table(dm, scheme="expanding_window")
    assert "0.023" in result["markdown"]


def test_dm_table_shows_p_below_0001_as_less_than():
    """A p-value of 1e-5 renders as '< 0.001' in the markdown."""
    dm = _synthetic_dm_results().copy()
    dm.loc[0, "p_value"] = 1e-5
    result = make_dm_test_table(dm, scheme="expanding_window")
    assert "< 0.001" in result["markdown"]


def test_dm_table_includes_bonferroni_column():
    """The Bonferroni column header appears in the markdown."""
    dm = _synthetic_dm_results()
    result = make_dm_test_table(dm, scheme="expanding_window")
    assert "Bonferroni" in result["markdown"]


def test_plot_model_comparison_bar_returns_figure():
    """Returns a matplotlib Figure; title contains the scheme and metric label."""
    agg = _synthetic_aggregated_results()
    fig = plot_model_comparison_bar(agg, scheme="expanding_window", metric="rmse")
    assert isinstance(fig, matplotlib.figure.Figure)
    title = fig.axes[0].get_title()
    assert "RMSE" in title
    assert "expanding" in title.lower()
    plt.close(fig)


def test_plot_regime_performance_heatmap_returns_figure():
    """Returns a Figure; the colorbar axis exists (so the figure has at least 2 axes)."""
    per_regime = _synthetic_per_regime_results()
    fig = plot_regime_performance_heatmap(per_regime, scheme="expanding_window", metric="rmse")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_cv_fold_variance_returns_figure():
    """Returns a Figure; one box per model appears as an xtick label."""
    per_fold = _synthetic_per_fold_metrics()
    fig = plot_cv_fold_variance(per_fold, scheme="expanding_window", metric="rmse")
    assert isinstance(fig, matplotlib.figure.Figure)
    xtick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert set(xtick_labels) == {"ridge", "xgboost", "lightgbm", "arima"}
    plt.close(fig)


def test_figures_save_at_300_dpi_png_and_pdf(tmp_path):
    """A returned Figure persists to PNG (with dpi=300 metadata) and PDF on disk."""
    agg = _synthetic_aggregated_results()
    fig = plot_model_comparison_bar(agg, scheme="expanding_window")
    png_path = tmp_path / "test.png"
    pdf_path = tmp_path / "test.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    assert png_path.exists()
    assert pdf_path.exists()

    with Image.open(png_path) as img:
        dpi = img.info.get("dpi")
    assert dpi is not None
    assert round(dpi[0]) == 300
    assert round(dpi[1]) == 300
