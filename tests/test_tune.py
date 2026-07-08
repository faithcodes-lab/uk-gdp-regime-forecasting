"""Tests for src/models/tune.py.

The test that matters most is test_tune_search_never_touches_data_past_75_percent:
it captures the X handed to the splitter and asserts it matches the first 75%
of the input exactly, so the search can never touch the held-out evaluation
window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import tune as tune_module
from src.models.tune import (
    load_tuning_result,
    save_tuning_result,
    tune_lightgbm,
    tune_ridge,
    tune_xgboost,
)


def _synthetic_Xy(n: int = 60, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Builds synthetic X (three features) and y for a deterministic linear relationship plus noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "a": rng.normal(scale=0.01, size=n),
            "b": rng.normal(scale=1.0, size=n),
            "c": rng.normal(scale=1000.0, size=n),
        }
    )
    y = pd.Series(0.5 + 2 * X["b"] + 0.001 * X["c"] + rng.normal(scale=0.1, size=n))
    return X, y


def test_tune_ridge_returns_best_params_with_expected_keys():
    """tune_ridge returns a result dict with the ridge__alpha key in best_params."""
    X, y = _synthetic_Xy(n=60)
    result = tune_ridge(X, y, n_splits=3, n_iter=3)
    assert result["model"] == "ridge"
    assert "ridge__alpha" in result["best_params"]
    assert isinstance(result["best_score"], float)


def test_tune_xgboost_returns_best_params_with_expected_keys():
    """tune_xgboost returns a result dict with the expected XGBoost param keys."""
    X, y = _synthetic_Xy(n=60)
    result = tune_xgboost(X, y, n_splits=3, n_iter=3)
    assert result["model"] == "xgboost"
    for key in ["xgboost__max_depth", "xgboost__learning_rate", "xgboost__n_estimators"]:
        assert key in result["best_params"], f"missing param {key}"


def test_tune_lightgbm_returns_best_params_with_expected_keys():
    """tune_lightgbm returns a result dict with the expected LightGBM param keys including min_child_samples."""
    X, y = _synthetic_Xy(n=60)
    result = tune_lightgbm(X, y, n_splits=3, n_iter=3)
    assert result["model"] == "lightgbm"
    for key in [
        "lightgbm__max_depth",
        "lightgbm__learning_rate",
        "lightgbm__n_estimators",
        "lightgbm__min_child_samples",
    ]:
        assert key in result["best_params"], f"missing param {key}"


def test_tune_ridge_deterministic_with_fixed_seed():
    """Two tune_ridge runs with the same seed and data return identical best_params and best_score."""
    X, y = _synthetic_Xy(n=60)
    a = tune_ridge(X, y, n_splits=3, n_iter=3, random_state=42)
    b = tune_ridge(X, y, n_splits=3, n_iter=3, random_state=42)
    assert a["best_params"] == b["best_params"]
    assert a["best_score"] == b["best_score"]


def test_tune_result_dict_has_all_metadata_fields():
    """The result dict carries model, scoring, n_iter, n_splits, random_state, and timestamp."""
    X, y = _synthetic_Xy(n=60)
    result = tune_ridge(X, y, n_splits=3, n_iter=3, random_state=42)
    for key in [
        "model",
        "best_params",
        "best_score",
        "scoring",
        "n_iter",
        "n_splits",
        "random_state",
        "timestamp",
    ]:
        assert key in result, f"missing key {key}"
    assert result["scoring"] == "neg_root_mean_squared_error"
    assert result["n_iter"] == 3
    assert result["n_splits"] == 3
    assert result["random_state"] == 42


def test_save_tuning_result_writes_valid_json(tmp_path: Path):
    """save_tuning_result writes a JSON file with all dict keys preserved."""
    cache_path = tmp_path / "ridge_best_params.json"
    result = {
        "model": "ridge",
        "best_params": {"ridge__alpha": 1.5},
        "best_score": -0.45,
        "scoring": "neg_root_mean_squared_error",
        "n_iter": 20,
        "n_splits": 5,
        "random_state": 42,
        "timestamp": "2026-06-30T13:00:00+00:00",
    }
    path = save_tuning_result(result, cache_path)
    assert path == cache_path
    assert cache_path.exists()
    assert json.loads(cache_path.read_text()) == result


def test_load_tuning_result_reads_saved_file(tmp_path: Path):
    """load_tuning_result returns the same dict that was saved."""
    cache_path = tmp_path / "x.json"
    original = {"model": "ridge", "best_params": {"ridge__alpha": 2.0}}
    save_tuning_result(original, cache_path)
    assert load_tuning_result(cache_path) == original


def test_load_tuning_result_returns_none_when_file_missing(tmp_path: Path):
    """load_tuning_result returns None for a missing cache file."""
    assert load_tuning_result(tmp_path / "nope.json") is None


def test_tune_search_never_touches_data_past_75_percent(monkeypatch):
    """The X handed to the splitter is exactly the first 75% of the input.

    Captures expanding_window_splits via monkeypatch and asserts the
    captured X equals X.iloc[:int(0.75 * n)] in both length and content.
    Anything past the 75% boundary stays held out for evaluation.
    """
    captured: list[pd.DataFrame] = []
    from src.models.cv import expanding_window_splits as original

    def capturing(X, *args, **kwargs):
        captured.append(X)
        return original(X, *args, **kwargs)

    monkeypatch.setattr(tune_module, "expanding_window_splits", capturing)

    n_input = 60
    X, y = _synthetic_Xy(n=n_input)

    tune_ridge(X, y, n_splits=3, n_iter=3)

    expected_tune_n = int(0.75 * n_input)
    assert len(captured) == 1, f"expected 1 call to expanding_window_splits, got {len(captured)}"
    assert (
        len(captured[0]) == expected_tune_n
    ), f"expected {expected_tune_n} rows in tuning window, got {len(captured[0])}"
    pd.testing.assert_frame_equal(captured[0], X.iloc[:expected_tune_n])
