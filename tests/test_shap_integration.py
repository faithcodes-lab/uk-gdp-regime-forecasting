"""End-to-end integration test for the SHAP master script.

Runs the real pipeline on the frozen dataset with a small bootstrap count
so it stays fast, checks every output is written, the matrix is 6x6, and
the persisted stability matrix is byte-identical across two runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.run_shap_analysis import run_shap_analysis

# Every test here runs the full pipeline on the real frozen dataset, which is
# gitignored and absent in CI, so the whole module is integration-only.
pytestmark = pytest.mark.integration

_EXPECTED_CSVS = [
    "global_importance.csv",
    "per_regime_importance.csv",
    "per_regime_rankings.csv",
    "stability_matrix.csv",
    "stability_pairs.csv",
    "bootstrap_cis.csv",
]
_EXPECTED_FIGS = [
    "shap_global_importance",
    "shap_per_regime_importance",
    "shap_stability_heatmap",
]


def test_run_shap_analysis_writes_all_outputs(tmp_path):
    out = tmp_path / "shap"
    figs = tmp_path / "figures"
    run_shap_analysis(output_dir=out, figures_dir=figs, n_bootstrap=20, seed=42)

    for name in _EXPECTED_CSVS:
        assert (out / name).stat().st_size > 0, name
    for stem in _EXPECTED_FIGS:
        assert (figs / f"{stem}.png").stat().st_size > 5000, stem
        assert (figs / f"{stem}.pdf").stat().st_size > 0, stem


def test_stability_matrix_is_six_by_six(tmp_path):
    out = tmp_path / "shap"
    run_shap_analysis(output_dir=out, figures_dir=tmp_path / "figures", n_bootstrap=20, seed=42)
    matrix = pd.read_csv(out / "stability_matrix.csv", index_col=0)
    assert matrix.shape == (6, 6)


def test_stability_matrix_reproducible_across_runs(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    run_shap_analysis(output_dir=a, figures_dir=a, n_bootstrap=20, seed=42)
    run_shap_analysis(output_dir=b, figures_dir=b, n_bootstrap=20, seed=42)
    assert (
        Path(a / "stability_matrix.csv").read_bytes()
        == Path(b / "stability_matrix.csv").read_bytes()
    )
    assert Path(a / "bootstrap_cis.csv").read_bytes() == Path(b / "bootstrap_cis.csv").read_bytes()
