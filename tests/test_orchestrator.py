"""Smoke tests for src/evaluation/orchestrator.py.

Tests mock CP2's generate_predictions to short-circuit the expensive
per-fold retraining (which has its own coverage in tests/test_predictions.py)
and exercise the orchestrator's wiring, artifact writing, and determinism
on synthetic predictions. The methodological heart is the determinism
test: two runs of run_evaluation produce byte-identical per-fold CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation import orchestrator


def _synthetic_dataset(tmp_path: Path, n: int = 40, seed: int = 42) -> Path:
    """Writes a tiny parquet matching the dataset schema the orchestrator reads."""
    rng = np.random.default_rng(seed)
    regimes = (["A"] * 14) + (["B"] * 13) + (["C"] * 13)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=n, freq="QS"),
            "regime": regimes[:n],
            "gdp_growth": rng.normal(size=n),
        }
    )
    path = tmp_path / "dataset.parquet"
    df.to_parquet(path)
    return path


def _synthetic_predictions(seed: int = 42) -> pd.DataFrame:
    """Returns a fake predictions DataFrame matching CP2's long-format schema."""
    rng = np.random.default_rng(seed)
    models = ["arima", "ridge", "xgboost", "lightgbm"]
    schemes = ["expanding_window", "regime_aligned"]
    regimes = ["A", "B", "C"]
    rows: list[dict] = []
    for model in models:
        for scheme in schemes:
            for fold_idx in range(1, 5):
                for k in range(4):
                    pos = (fold_idx - 1) * 4 + k
                    rows.append(
                        {
                            "model": model,
                            "quarter": pd.Timestamp("2010-01-01") + pd.Timedelta(days=pos * 90),
                            "regime": regimes[pos % 3],
                            "y_true": float(rng.normal()),
                            "y_pred": float(rng.normal()),
                            "fold_idx": fold_idx,
                            "scheme": scheme,
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def patched_generate_predictions(monkeypatch):
    """Monkeypatches the orchestrator's generate_predictions to return synthetic rows."""
    df = _synthetic_predictions()

    def _fake(_df, output_path=None):  # noqa: ARG001 (signature mirrors the real function)
        return df.copy()

    monkeypatch.setattr(orchestrator, "generate_predictions", _fake)
    return df


def test_run_evaluation_writes_predictions_parquet_with_expected_columns(
    tmp_path, patched_generate_predictions
):
    """The cached predictions parquet has CP2's long-format columns."""
    dataset = _synthetic_dataset(tmp_path)
    paths = orchestrator.run_evaluation(dataset, tmp_path / "out", quiet=True)
    assert paths["predictions"].exists()
    df = pd.read_parquet(paths["predictions"])
    assert set(df.columns) == {
        "model",
        "quarter",
        "regime",
        "y_true",
        "y_pred",
        "fold_idx",
        "scheme",
    }


def test_run_evaluation_writes_metric_csvs_with_expected_columns(
    tmp_path, patched_generate_predictions
):
    """All five metric CSVs are written with the right column sets."""
    dataset = _synthetic_dataset(tmp_path)
    paths = orchestrator.run_evaluation(dataset, tmp_path / "out", quiet=True)
    metrics_dir = paths["metrics"]

    aggregated = pd.read_csv(metrics_dir / "aggregated.csv")
    per_fold = pd.read_csv(metrics_dir / "per_fold.csv")
    per_regime = pd.read_csv(metrics_dir / "per_regime.csv")
    dm = pd.read_csv(metrics_dir / "dm_test.csv")

    assert "mean_rmse" in aggregated.columns
    assert {"model", "scheme", "fold_idx", "rmse"}.issubset(per_fold.columns)
    assert {"model", "scheme", "regime", "small_sample"}.issubset(per_regime.columns)
    assert {"model_a", "model_b", "scheme", "p_value_bonferroni"}.issubset(dm.columns)
    assert (metrics_dir / "small_sample_cis.csv").exists()


def test_run_evaluation_writes_tables_for_each_scheme(tmp_path, patched_generate_predictions):
    """Markdown and LaTeX exist for all three tables on both schemes."""
    dataset = _synthetic_dataset(tmp_path)
    paths = orchestrator.run_evaluation(dataset, tmp_path / "out", quiet=True)
    tables_dir = paths["tables"]
    for stem in ("overall_performance", "per_regime", "dm_test"):
        for scheme in ("expanding_window", "regime_aligned"):
            assert (tables_dir / f"{stem}_{scheme}.md").exists()
            assert (tables_dir / f"{stem}_{scheme}.tex").exists()


def test_run_evaluation_writes_figures_at_300_dpi_png_and_pdf(
    tmp_path, patched_generate_predictions
):
    """Each figure trio exists as PNG (dpi metadata 300) and PDF for each scheme."""
    dataset = _synthetic_dataset(tmp_path)
    paths = orchestrator.run_evaluation(dataset, tmp_path / "out", quiet=True)
    figures_dir = paths["figures"]
    for stem in ("model_comparison", "regime_heatmap", "cv_fold_variance"):
        for scheme in ("expanding_window", "regime_aligned"):
            png = figures_dir / f"{stem}_{scheme}_rmse.png"
            pdf = figures_dir / f"{stem}_{scheme}_rmse.pdf"
            assert png.exists()
            assert pdf.exists()
            with Image.open(png) as img:
                dpi = img.info.get("dpi")
            assert dpi is not None
            assert round(dpi[0]) == 300


def test_run_evaluation_is_deterministic(tmp_path, patched_generate_predictions):
    """Two runs with the same seed produce byte-identical per-fold and aggregated CSVs."""
    dataset = _synthetic_dataset(tmp_path)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    orchestrator.run_evaluation(dataset, out_a, quiet=True)
    orchestrator.run_evaluation(dataset, out_b, quiet=True)

    for csv_name in ("per_fold.csv", "aggregated.csv", "per_regime.csv", "dm_test.csv"):
        bytes_a = (out_a / "metrics" / csv_name).read_bytes()
        bytes_b = (out_b / "metrics" / csv_name).read_bytes()
        assert bytes_a == bytes_b, f"determinism failed: {csv_name}"
