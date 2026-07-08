"""Tests for src/evaluation/predictions.py.

The tests that matter most are test_each_prediction_target_quarter_matches_y_true
(alignment proof), test_sklearn_models_refit_per_fold_not_full_data (per-fold
retraining proof), and test_no_target_quarter_appears_in_its_own_training_fold
(evaluation-side leakage proof).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import predictions as predictions_module
from src.models.cv import expanding_window_splits, regime_aligned_splits


def _balance_regime_sizes(n: int, num_regimes: int) -> list[int]:
    """Splits n rows across num_regimes regimes as evenly as possible."""
    base = n // num_regimes
    rem = n % num_regimes
    return [base + (1 if i < rem else 0) for i in range(num_regimes)]


def _synthetic_df(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic dataset shaped like data/processed/final_dataset.parquet."""
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2005-01-01", periods=n, freq="Q").to_timestamp(how="end").normalize()
    sizes = _balance_regime_sizes(n, num_regimes=6)
    labels = ["A", "B", "C", "D", "E", "F"]
    regimes: list[str] = []
    for label, sz in zip(labels, sizes):
        regimes.extend([label] * sz)
    return pd.DataFrame(
        {
            "date": dates,
            "gdp_growth": rng.normal(scale=0.5, size=n),
            "regime": regimes,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )


def _stub_load_tuning_result(path: Path | str) -> dict[str, Any]:
    """Returns minimal best-params for the requested model."""
    name = Path(path).stem.replace("_best_params", "")
    return {
        "ridge": {"best_params": {"ridge__alpha": 1.0}},
        "xgboost": {
            "best_params": {
                "xgboost__max_depth": 3,
                "xgboost__learning_rate": 0.1,
                "xgboost__n_estimators": 50,
            }
        },
        "lightgbm": {
            "best_params": {
                "lightgbm__max_depth": 3,
                "lightgbm__learning_rate": 0.1,
                "lightgbm__n_estimators": 50,
                "lightgbm__min_child_samples": 5,
            }
        },
        "arima": {"best_params": {"order": [1, 0, 0], "seasonal_order": [0, 0, 0, 0]}},
    }[name]


class _StubModel:
    """Trivial sklearn model substitute: records fit calls, returns zero predictions."""

    fit_calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        type(self).fit_calls.append({"size": len(X), "training_indices": list(X.index)})
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _stub_cross_validate_arima(y, splits, order=None):
    """Returns zero predictions, one per test position across all folds."""
    total = sum(len(test_idx) for _, test_idx in splits)
    return np.zeros(total)


def _setup_stubs(monkeypatch) -> None:
    """Wires up stubs for load_tuning_result, the three sklearn factories, and cross_validate_arima."""
    _StubModel.fit_calls.clear()
    monkeypatch.setattr(predictions_module, "load_tuning_result", _stub_load_tuning_result)
    monkeypatch.setattr(predictions_module, "RidgeForecastingModel", _StubModel)
    monkeypatch.setattr(predictions_module, "XGBForecastingModel", _StubModel)
    monkeypatch.setattr(predictions_module, "LGBMForecastingModel", _StubModel)
    monkeypatch.setattr(predictions_module, "cross_validate_arima", _stub_cross_validate_arima)


def test_generate_predictions_returns_expected_columns(monkeypatch):
    """The output DataFrame has exactly the seven documented columns."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    result = predictions_module.generate_predictions(df)
    assert set(result.columns) == {
        "model",
        "quarter",
        "regime",
        "y_true",
        "y_pred",
        "fold_idx",
        "scheme",
    }


def test_each_prediction_target_quarter_matches_y_true(monkeypatch):
    """Every row's y_true equals the observed growth at that quarter's date.

    The alignment proof: catches off-by-one bugs in the shift logic between
    target_df, X_sklearn, and y_arima.
    """
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    result = predictions_module.generate_predictions(df)
    for _, row in result.iterrows():
        matching = df.loc[df["date"] == row["quarter"], "gdp_growth"]
        assert len(matching) == 1, f"quarter {row['quarter']} not unique in df"
        assert row["y_true"] == matching.iloc[0], (
            f"y_true {row['y_true']} does not match df[gdp_growth] at {row['quarter']} "
            f"which is {matching.iloc[0]}"
        )


def test_each_prediction_regime_matches_target_quarter_regime(monkeypatch):
    """Every row's regime equals the regime label at that quarter's date."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    result = predictions_module.generate_predictions(df)
    for _, row in result.iterrows():
        matching = df.loc[df["date"] == row["quarter"], "regime"]
        assert row["regime"] == matching.iloc[0]


def test_sklearn_models_refit_per_fold_not_full_data(monkeypatch):
    """Each sklearn fit receives the fold's training portion, never the full data.

    Catches the bug where the persisted .joblib model would be reused instead
    of refitting per fold (which would leak the entire dataset into every fit).
    """
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    target_size = len(df) - 1  # target_df is df.iloc[1:]; 79 rows here

    predictions_module.generate_predictions(df)

    fit_sizes = [call["size"] for call in _StubModel.fit_calls]
    # 8 expanding-window folds + 5 regime-aligned folds per model, 3 sklearn models
    assert len(fit_sizes) == 39, f"expected 39 fits, got {len(fit_sizes)}"
    assert max(fit_sizes) < target_size, (
        f"a fit received {max(fit_sizes)} rows but target_df has {target_size}; "
        "no fold should ever cover the full data"
    )


def test_arima_uses_cross_validate_arima(monkeypatch):
    """ARIMA predictions flow through cross_validate_arima, not direct .fit on the persisted model."""
    arima_calls: list[dict[str, Any]] = []

    def tracking_arima(y, splits, order=None):
        arima_calls.append({"y_size": len(y), "n_splits": len(splits), "order": order})
        return np.zeros(sum(len(test_idx) for _, test_idx in splits))

    _setup_stubs(monkeypatch)
    monkeypatch.setattr(predictions_module, "cross_validate_arima", tracking_arima)

    df = _synthetic_df(n=80)
    predictions_module.generate_predictions(df)

    # one call per scheme: expanding_window and regime_aligned
    assert len(arima_calls) == 2
    assert all(call["y_size"] == 80 for call in arima_calls)
    assert all(call["order"] == (1, 0, 0) for call in arima_calls)


def test_expanding_window_produces_32_predictions_per_model(monkeypatch):
    """Default n_splits=8 and test_size=4 produce 32 predictions per model per expanding scheme."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    result = predictions_module.generate_predictions(df)

    for model_name in ["ridge", "xgboost", "lightgbm", "arima"]:
        per_model = result[
            (result["model"] == model_name) & (result["scheme"] == "expanding_window")
        ]
        assert (
            len(per_model) == 32
        ), f"{model_name} expanding_window: {len(per_model)} predictions, expected 32"


def test_regime_aligned_produces_expected_predictions_per_model(monkeypatch):
    """Regime-aligned scheme produces sum-of-test-sizes predictions per model."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    target_df = df.iloc[1:].reset_index(drop=True)

    splits = regime_aligned_splits(target_df, regime_column="regime")
    expected_per_model = sum(len(test_idx) for _, test_idx in splits)

    result = predictions_module.generate_predictions(df)
    for model_name in ["ridge", "xgboost", "lightgbm", "arima"]:
        per_model = result[(result["model"] == model_name) & (result["scheme"] == "regime_aligned")]
        assert (
            len(per_model) == expected_per_model
        ), f"{model_name} regime_aligned: {len(per_model)}, expected {expected_per_model}"


def test_predictions_tagged_with_fold_idx_from_one(monkeypatch):
    """fold_idx is 1-indexed; 1 to 8 for expanding, 1 to 5 for regime-aligned."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    result = predictions_module.generate_predictions(df)

    expanding_fold_ids = set(
        result.loc[result["scheme"] == "expanding_window", "fold_idx"].unique()
    )
    regime_fold_ids = set(result.loc[result["scheme"] == "regime_aligned", "fold_idx"].unique())

    assert expanding_fold_ids == {1, 2, 3, 4, 5, 6, 7, 8}
    assert regime_fold_ids == {1, 2, 3, 4, 5}


def test_generate_predictions_writes_parquet_when_output_path_given(tmp_path, monkeypatch):
    """When output_path is given, the DataFrame is written as parquet and round-trips."""
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    out = tmp_path / "preds.parquet"
    result = predictions_module.generate_predictions(df, output_path=out)
    assert out.exists()
    loaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(result, loaded)


def test_no_target_quarter_appears_in_its_own_training_fold(monkeypatch):
    """For every prediction, the target quarter's row index is not in that fold's training set.

    The evaluation-side leakage proof: confirms that the per-fold splits are
    honoured by the prediction-generation loop, so no model sees its own
    test quarter during training.
    """
    _setup_stubs(monkeypatch)
    df = _synthetic_df(n=80)
    target_df = df.iloc[1:].reset_index(drop=True)

    splits_by_scheme = {
        "expanding_window": expanding_window_splits(target_df, n_splits=8, test_size=4),
        "regime_aligned": regime_aligned_splits(target_df, regime_column="regime"),
    }

    result = predictions_module.generate_predictions(df)

    for _, row in result.iterrows():
        scheme_splits = splits_by_scheme[row["scheme"]]
        train_idx, test_idx = scheme_splits[row["fold_idx"] - 1]
        target_pos_arr = target_df.index[target_df["date"] == row["quarter"]].to_numpy()
        assert len(target_pos_arr) == 1
        target_pos = int(target_pos_arr[0])

        assert target_pos in set(test_idx), (
            f"target {row['quarter']} not in test_idx of fold {row['fold_idx']} "
            f"({row['scheme']})"
        )
        assert target_pos not in set(train_idx), (
            f"LEAKAGE: target {row['quarter']} in train_idx of fold {row['fold_idx']} "
            f"({row['scheme']}) for {row['model']}"
        )
