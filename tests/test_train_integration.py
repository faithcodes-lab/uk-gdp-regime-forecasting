"""Tests for src/models/train_all.py.

Two tests matter most: the alignment proof (X at row i is paired with the
next row's gdp_growth, with the last row dropped) and the reproducibility
proof (two consecutive runs produce byte-identical .joblib files).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import train_all as train_module


def _synthetic_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic dataset shaped like data/processed/final_dataset.parquet."""
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2010-01-01", periods=n, freq="Q").to_timestamp(how="end").normalize()
    half = n // 2
    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": rng.normal(scale=0.5, size=n),
            "regime": ["A"] * half + ["B"] * (n - half),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )


def _save_dataset(tmp_path: Path, df: pd.DataFrame) -> Path:
    """Writes the synthetic df to tmp_path/data.parquet and returns the path."""
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    return path


def _patch_environment(monkeypatch, tmp_path: Path) -> None:
    """Redirects data path, tuning dir, and models dir into tmp_path."""
    monkeypatch.setattr(train_module, "_DATA_PATH", tmp_path / "data.parquet")
    monkeypatch.setattr(train_module, "_TUNING_DIR", tmp_path / "tuning")
    monkeypatch.setattr(train_module, "_MODELS_DIR", tmp_path / "models")


def _stub_tunes(monkeypatch):
    """Replaces tune_* and select_arima_order with fast deterministic stubs."""

    def stub_ridge(X, y, **kwargs):
        return {
            "model": "ridge",
            "best_params": {"ridge__alpha": 1.0},
            "best_score": -1.0,
            "scoring": "neg_root_mean_squared_error",
            "n_iter": 1,
            "n_splits": 1,
            "random_state": 42,
            "timestamp": "2026-06-30T00:00:00+00:00",
        }

    def stub_xgb(X, y, **kwargs):
        return {
            "model": "xgboost",
            "best_params": {
                "xgboost__max_depth": 3,
                "xgboost__learning_rate": 0.1,
                "xgboost__n_estimators": 50,
            },
            "best_score": -1.0,
            "scoring": "neg_root_mean_squared_error",
            "n_iter": 1,
            "n_splits": 1,
            "random_state": 42,
            "timestamp": "2026-06-30T00:00:00+00:00",
        }

    def stub_lgbm(X, y, **kwargs):
        return {
            "model": "lightgbm",
            "best_params": {
                "lightgbm__max_depth": 3,
                "lightgbm__learning_rate": 0.1,
                "lightgbm__n_estimators": 50,
                "lightgbm__min_child_samples": 5,
            },
            "best_score": -1.0,
            "scoring": "neg_root_mean_squared_error",
            "n_iter": 1,
            "n_splits": 1,
            "random_state": 42,
            "timestamp": "2026-06-30T00:00:00+00:00",
        }

    def stub_arima_order(y, **kwargs):
        return (1, 0, 0)

    monkeypatch.setattr(train_module, "tune_ridge", stub_ridge)
    monkeypatch.setattr(train_module, "tune_xgboost", stub_xgb)
    monkeypatch.setattr(train_module, "tune_lightgbm", stub_lgbm)
    monkeypatch.setattr(train_module, "select_arima_order", stub_arima_order)


def test_prepare_sklearn_xy_aligns_x_at_t_with_y_at_t_plus_1():
    """y at row i equals gdp_growth at row i+1, and the last row is dropped."""
    df = _synthetic_df(n=10)
    X, y = train_module._prepare_sklearn_Xy(df)
    assert len(X) == 9
    assert len(y) == 9
    for i in range(9):
        assert y.iloc[i] == df["gdp_growth"].iloc[i + 1], f"misalignment at row {i}"


def test_prepare_sklearn_xy_drops_regime_and_date():
    """The sklearn X frame contains neither date nor regime; gdp_growth stays as a feature."""
    df = _synthetic_df(n=10)
    X, _ = train_module._prepare_sklearn_Xy(df)
    assert "date" not in X.columns
    assert "regime" not in X.columns
    assert "gdp_growth" in X.columns


def test_prepare_arima_y_is_unshifted():
    """ARIMA's y is the full gdp_growth series, no shift, no drops."""
    df = _synthetic_df(n=10)
    y = train_module._prepare_arima_y(df)
    assert len(y) == 10
    np.testing.assert_array_equal(y.to_numpy(), df["gdp_growth"].to_numpy())


def test_train_all_produces_all_four_joblib_files(tmp_path, monkeypatch):
    """After main(), each of the four model joblib files exists on disk."""
    df = _synthetic_df(n=60)
    _save_dataset(tmp_path, df)
    _patch_environment(monkeypatch, tmp_path)
    _stub_tunes(monkeypatch)

    train_module.main(retune=False)

    for name in ["ridge", "xgboost", "lightgbm", "arima"]:
        assert (tmp_path / "models" / f"{name}.joblib").exists(), f"missing {name}.joblib"


def test_train_all_writes_metadata_alongside_each_model(tmp_path, monkeypatch):
    """Each model's metadata JSON has the documented keys."""
    df = _synthetic_df(n=60)
    _save_dataset(tmp_path, df)
    _patch_environment(monkeypatch, tmp_path)
    _stub_tunes(monkeypatch)

    train_module.main(retune=False)

    expected_keys = {
        "model",
        "best_params",
        "library_versions",
        "training_timestamp",
        "dataset_hash_md5",
        "n_training_rows",
        "random_state",
    }
    for name in ["ridge", "xgboost", "lightgbm", "arima"]:
        meta_path = tmp_path / "models" / f"{name}_meta.json"
        assert meta_path.exists(), f"missing {name}_meta.json"
        meta = json.loads(meta_path.read_text())
        assert expected_keys.issubset(
            meta.keys()
        ), f"{name}_meta.json missing keys: {expected_keys - meta.keys()}"


def test_train_all_uses_cache_when_present_and_retune_false(tmp_path, monkeypatch):
    """When cache files exist, tune_* and select_arima_order are not called."""
    df = _synthetic_df(n=60)
    _save_dataset(tmp_path, df)
    _patch_environment(monkeypatch, tmp_path)
    _stub_tunes(monkeypatch)

    # Pre-seed all four cache files with minimal valid contents
    (tmp_path / "tuning").mkdir(parents=True)
    cached_results = {
        "ridge": {"model": "ridge", "best_params": {"ridge__alpha": 2.0}},
        "xgboost": {
            "model": "xgboost",
            "best_params": {
                "xgboost__max_depth": 3,
                "xgboost__learning_rate": 0.1,
                "xgboost__n_estimators": 50,
            },
        },
        "lightgbm": {
            "model": "lightgbm",
            "best_params": {
                "lightgbm__max_depth": 3,
                "lightgbm__learning_rate": 0.1,
                "lightgbm__n_estimators": 50,
                "lightgbm__min_child_samples": 5,
            },
        },
        "arima": {
            "model": "arima",
            "best_params": {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0]},
        },
    }
    for name, result in cached_results.items():
        (tmp_path / "tuning" / f"{name}_best_params.json").write_text(json.dumps(result))

    calls: dict[str, int] = {"ridge": 0, "xgboost": 0, "lightgbm": 0, "arima_order": 0}

    def make_counter(key, original):
        def counted(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        return counted

    monkeypatch.setattr(train_module, "tune_ridge", make_counter("ridge", train_module.tune_ridge))
    monkeypatch.setattr(
        train_module, "tune_xgboost", make_counter("xgboost", train_module.tune_xgboost)
    )
    monkeypatch.setattr(
        train_module, "tune_lightgbm", make_counter("lightgbm", train_module.tune_lightgbm)
    )
    monkeypatch.setattr(
        train_module,
        "select_arima_order",
        make_counter("arima_order", train_module.select_arima_order),
    )

    train_module.main(retune=False)

    assert calls == {"ridge": 0, "xgboost": 0, "lightgbm": 0, "arima_order": 0}


def test_train_all_retunes_when_retune_flag_set(tmp_path, monkeypatch):
    """With retune=True, tune_* runs even if cache exists."""
    df = _synthetic_df(n=60)
    _save_dataset(tmp_path, df)
    _patch_environment(monkeypatch, tmp_path)
    _stub_tunes(monkeypatch)

    # Pre-seed a cache file that should be ignored
    (tmp_path / "tuning").mkdir(parents=True)
    (tmp_path / "tuning" / "ridge_best_params.json").write_text(
        '{"model":"ridge","best_params":{"ridge__alpha":99.0}}'
    )

    calls: dict[str, int] = {"ridge": 0}
    # capture the stub before monkeypatching, otherwise the counter would
    # call the counter itself (the same name) and infinitely recurse
    stub_ridge = train_module.tune_ridge

    def counted_ridge(*args, **kwargs):
        calls["ridge"] += 1
        return stub_ridge(*args, **kwargs)

    monkeypatch.setattr(train_module, "tune_ridge", counted_ridge)

    train_module.main(retune=True)

    assert calls["ridge"] == 1


def test_train_all_joblib_bytes_are_reproducible(tmp_path, monkeypatch):
    """Two consecutive runs with the same seed produce byte-identical joblib files."""
    df = _synthetic_df(n=60)
    _save_dataset(tmp_path, df)
    _patch_environment(monkeypatch, tmp_path)
    _stub_tunes(monkeypatch)

    train_module.main(retune=False)
    first_bytes = {
        name: (tmp_path / "models" / f"{name}.joblib").read_bytes()
        for name in ["ridge", "xgboost", "lightgbm", "arima"]
    }

    train_module.main(retune=False)
    second_bytes = {
        name: (tmp_path / "models" / f"{name}.joblib").read_bytes()
        for name in ["ridge", "xgboost", "lightgbm", "arima"]
    }

    for name, value in first_bytes.items():
        assert value == second_bytes[name], f"joblib bytes differ for {name}"


def test_train_all_fails_clearly_when_data_missing(tmp_path, monkeypatch):
    """When the dataset is missing, raises FileNotFoundError mentioning `make data`."""
    _patch_environment(monkeypatch, tmp_path)
    # deliberately do not write the parquet
    with pytest.raises(FileNotFoundError, match="make data"):
        train_module.main(retune=False)
