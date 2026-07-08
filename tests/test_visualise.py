"""Tests for src/regimes/visualise.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from src.regimes.visualise import plot_gdp_with_regimes  # noqa: E402


def _real_regimes() -> list[dict]:
    """The six project regimes (mirrors config/regimes.yaml)."""
    return [
        {
            "id": 1,
            "label": "Pre-GFC Stability",
            "start": pd.Timestamp("2000-01-01"),
            "end": pd.Timestamp("2007-12-31"),
        },
        {
            "id": 2,
            "label": "Global Financial Crisis",
            "start": pd.Timestamp("2008-01-01"),
            "end": pd.Timestamp("2009-12-31"),
        },
        {
            "id": 3,
            "label": "Post-GFC Recovery",
            "start": pd.Timestamp("2010-01-01"),
            "end": pd.Timestamp("2016-06-30"),
        },
        {
            "id": 4,
            "label": "Brexit Transition",
            "start": pd.Timestamp("2016-07-01"),
            "end": pd.Timestamp("2019-12-31"),
        },
        {
            "id": 5,
            "label": "COVID-19 Shock",
            "start": pd.Timestamp("2020-01-01"),
            "end": pd.Timestamp("2021-06-30"),
        },
        {
            "id": 6,
            "label": "Post-COVID Recovery",
            "start": pd.Timestamp("2021-07-01"),
            "end": pd.Timestamp("2025-12-31"),
        },
    ]


def _synthetic_df() -> pd.DataFrame:
    """A 104-quarter synthetic dataset shaped like the real parquet."""
    dates = pd.period_range("2000Q1", periods=104, freq="Q").to_timestamp(how="end").normalize()
    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": [0.5] * 104,
        }
    )


def test_function_returns_matplotlib_figure():
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes())
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_png_file_created_with_min_size(tmp_path: Path):
    png = tmp_path / "regime_visualisation.png"
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes(), png_path=png)
    plt.close(fig)
    assert png.exists()
    assert png.stat().st_size > 50_000


def test_pdf_file_created(tmp_path: Path):
    pdf = tmp_path / "regime_visualisation.pdf"
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes(), pdf_path=pdf)
    plt.close(fig)
    assert pdf.exists()
    # PDF can be small for a simple line plot; just confirm non-trivial.
    assert pdf.stat().st_size > 1_000


def test_png_dpi_is_300(tmp_path: Path):
    png = tmp_path / "regime_visualisation.png"
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes(), png_path=png, dpi=300)
    plt.close(fig)
    with Image.open(png) as img:
        dpi = img.info.get("dpi")
    assert dpi is not None
    # PIL returns (xdpi, ydpi); both should be 300 (allow tiny float drift).
    assert round(dpi[0]) == 300
    assert round(dpi[1]) == 300


def test_legend_has_six_regime_entries():
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes())
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 6
    plt.close(fig)


def test_legend_entries_include_date_ranges():
    """Each legend label must combine the regime name and its date range."""
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=_real_regimes())
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Pre-GFC Stability (2000 Q1 - 2007 Q4)" in labels
    assert "Global Financial Crisis (2008 Q1 - 2009 Q4)" in labels
    assert "COVID-19 Shock (2020 Q1 - 2021 Q2)" in labels
    assert "Post-COVID Recovery (2021 Q3 - 2025 Q4)" in labels
    plt.close(fig)


def test_function_does_not_mutate_input_df():
    df = _synthetic_df()
    before_cols = list(df.columns)
    before_len = len(df)
    fig = plot_gdp_with_regimes(df, regimes=_real_regimes())
    plt.close(fig)
    assert list(df.columns) == before_cols
    assert len(df) == before_len


def test_missing_gdp_growth_column_raises():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"])})
    with pytest.raises(ValueError, match="gdp_growth"):
        plot_gdp_with_regimes(df, regimes=_real_regimes())


def test_missing_date_column_raises():
    df = pd.DataFrame({"gdp_growth": [0.5]})
    with pytest.raises(ValueError, match="date"):
        plot_gdp_with_regimes(df, regimes=_real_regimes())


def test_default_regimes_loaded_when_none_passed():
    """When regimes is None, the function loads from regimes_config()."""
    fig = plot_gdp_with_regimes(_synthetic_df(), regimes=None)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 6
    plt.close(fig)


def test_main_writes_png_and_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End-to-end: main() writes both artefacts under a mocked repo root."""
    from src.regimes import visualise as vis

    monkeypatch.setattr(vis, "_load_dataset", _synthetic_df)
    monkeypatch.setattr(vis, "_repo_root", lambda: tmp_path)

    vis.main()

    png = tmp_path / "results" / "figures" / "regime_visualisation.png"
    pdf = tmp_path / "results" / "figures" / "regime_visualisation.pdf"
    assert png.exists()
    assert pdf.exists()
    assert png.stat().st_size > 50_000
