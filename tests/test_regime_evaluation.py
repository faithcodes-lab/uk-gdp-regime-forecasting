"""Tests for src/evaluation/regime_evaluation.py.

Tests that matter most: test_evaluate_per_regime_metrics_match_compute_all_metrics
(grouping correctness), test_evaluate_per_regime_flags_small_sample_below_threshold
and the large-regime twin (correct threshold behaviour), and
test_bootstrap_regime_metrics_wider_ci_for_smaller_n (bootstrap reflects sample size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.regime_evaluation import bootstrap_regime_metrics, evaluate_per_regime


def _synthetic_predictions_df(n_per_regime: dict[str, int], seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic predictions DataFrame matching CP2's shape, for testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for model in ["ridge", "xgboost", "lightgbm", "arima"]:
        for scheme in ["expanding_window", "regime_aligned"]:
            for regime, n in n_per_regime.items():
                for i in range(n):
                    rows.append(
                        {
                            "model": model,
                            "quarter": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 90),
                            "regime": regime,
                            "y_true": float(rng.normal()),
                            "y_pred": float(rng.normal()),
                            "fold_idx": 1,
                            "scheme": scheme,
                        }
                    )
    return pd.DataFrame(rows)


def _synthetic_regime_aligned_predictions_df(
    n_quarters: int, n_folds: int, seed: int = 42
) -> pd.DataFrame:
    """Builds a predictions DataFrame where n_quarters distinct quarters are each
    repeated across n_folds folds, matching how regime-aligned CV re-tests a
    later regime in every subsequent fold (Section 3.6). n_observations for
    the regime is n_quarters * n_folds, but n_quarters stays n_quarters.
    """
    rng = np.random.default_rng(seed)
    quarters = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i * 90) for i in range(n_quarters)]
    rows: list[dict] = []
    for fold_idx in range(n_folds):
        for quarter in quarters:
            rows.append(
                {
                    "model": "xgboost",
                    "quarter": quarter,
                    "regime": "COVID-19 Shock",
                    "y_true": float(rng.normal()),
                    "y_pred": float(rng.normal()),
                    "fold_idx": fold_idx,
                    "scheme": "regime_aligned",
                }
            )
    return pd.DataFrame(rows)


def _y_train_series(n: int = 50, seed: int = 42) -> pd.Series:
    """Builds a synthetic training y series for MASE scaling."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n))


def test_evaluate_per_regime_returns_expected_columns():
    """The output DataFrame has exactly the nine documented columns."""
    df = _synthetic_predictions_df({"A": 10, "B": 6})
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    assert set(result.columns) == {
        "model",
        "scheme",
        "regime",
        "n_observations",
        "n_quarters",
        "rmse",
        "mae",
        "mase",
        "r2",
        "small_sample",
    }


def test_evaluate_per_regime_one_row_per_model_scheme_regime_combination():
    """For 4 models, 2 schemes, 6 regimes, the result has 48 rows."""
    regimes = {f"R{i}": 12 for i in range(6)}
    df = _synthetic_predictions_df(regimes)
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    assert len(result) == 4 * 2 * 6


def test_evaluate_per_regime_n_observations_matches_predictions_subset():
    """For each row, n_observations equals the count of predictions in that group."""
    df = _synthetic_predictions_df({"Pre-GFC": 33, "GFC": 6, "Post-GFC": 27})
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    for _, row in result.iterrows():
        expected_n = len(
            df[
                (df["model"] == row["model"])
                & (df["scheme"] == row["scheme"])
                & (df["regime"] == row["regime"])
            ]
        )
        assert row["n_observations"] == expected_n


def test_evaluate_per_regime_metrics_match_compute_all_metrics():
    """For one (model, scheme, regime) group, metrics match an independent compute_all_metrics call."""
    df = _synthetic_predictions_df({"Pre-GFC": 33, "GFC": 6})
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)

    row = result.iloc[0]
    subset = df[
        (df["model"] == row["model"])
        & (df["scheme"] == row["scheme"])
        & (df["regime"] == row["regime"])
    ]
    expected = compute_all_metrics(
        subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), y_train
    )
    assert row["rmse"] == pytest.approx(expected["rmse"])
    assert row["mae"] == pytest.approx(expected["mae"])
    assert row["mase"] == pytest.approx(expected["mase"])
    assert row["r2"] == pytest.approx(expected["r2"])


def test_evaluate_per_regime_flags_small_sample_below_threshold():
    """A regime with n=6 (default threshold 10) is flagged as small_sample."""
    df = _synthetic_predictions_df({"COVID": 6, "Pre-GFC": 33})
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    covid_rows = result[result["regime"] == "COVID"]
    assert covid_rows["small_sample"].all()


def test_evaluate_per_regime_does_not_flag_large_regime():
    """A regime with n=33 (default threshold 10) is not flagged."""
    df = _synthetic_predictions_df({"Pre-GFC": 33})
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    assert not result["small_sample"].any()


def test_evaluate_per_regime_n_quarters_counts_distinct_quarters_not_rows():
    """n_quarters is the count of distinct quarters, not the row count, when a
    regime's quarters are repeated across folds (regime-aligned scheme)."""
    df = _synthetic_regime_aligned_predictions_df(n_quarters=6, n_folds=4)
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    row = result.iloc[0]
    assert row["n_observations"] == 24
    assert row["n_quarters"] == 6


def test_evaluate_per_regime_flags_small_sample_by_unique_quarters_even_when_n_observations_is_large():
    """A regime with only 6 unique quarters, tested across 4 folds (n_observations=24,
    which is >= the default threshold of 10), must still be flagged small_sample.
    This is the regression test for the bug where the regime_heatmap figure
    silently missed COVID-19 Shock as small-sample under regime-aligned CV,
    because the flag was keyed on n_observations instead of n_quarters."""
    df = _synthetic_regime_aligned_predictions_df(n_quarters=6, n_folds=4)
    y_train = _y_train_series()
    result = evaluate_per_regime(df, y_train)
    row = result.iloc[0]
    assert row["n_observations"] >= 10
    assert row["small_sample"]


def test_evaluate_per_regime_threshold_is_parameterisable():
    """A custom small_sample_threshold changes the flagging behaviour."""
    df = _synthetic_predictions_df({"Brexit": 14})
    y_train = _y_train_series()

    result_default = evaluate_per_regime(df, y_train)
    assert not result_default["small_sample"].any()

    result_strict = evaluate_per_regime(df, y_train, small_sample_threshold=20)
    assert result_strict["small_sample"].all()


def test_bootstrap_regime_metrics_returns_expected_keys():
    """Bootstrap returns a dict with rmse, mae, mase, r2 keys, each a (lower, upper) tuple."""
    df = _synthetic_predictions_df({"R": 30})
    y_train = _y_train_series()
    subset = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "R")
    ]
    cis = bootstrap_regime_metrics(subset, y_train, n_bootstrap=100, random_state=42)
    assert set(cis.keys()) == {"rmse", "mae", "mase", "r2"}
    for ci in cis.values():
        assert isinstance(ci, tuple)
        assert len(ci) == 2


def test_bootstrap_regime_metrics_lower_less_than_upper():
    """For every metric, the lower CI bound is at most the upper bound."""
    df = _synthetic_predictions_df({"R": 30})
    y_train = _y_train_series()
    subset = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "R")
    ]
    cis = bootstrap_regime_metrics(subset, y_train, n_bootstrap=200, random_state=42)
    for metric_name, (lower, upper) in cis.items():
        if np.isfinite(lower) and np.isfinite(upper):
            assert lower <= upper, f"{metric_name}: lower {lower} > upper {upper}"


def test_bootstrap_regime_metrics_deterministic_with_fixed_seed():
    """Two bootstrap calls with the same seed return identical CIs."""
    df = _synthetic_predictions_df({"R": 30})
    y_train = _y_train_series()
    subset = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "R")
    ]
    cis_a = bootstrap_regime_metrics(subset, y_train, n_bootstrap=100, random_state=42)
    cis_b = bootstrap_regime_metrics(subset, y_train, n_bootstrap=100, random_state=42)
    for metric_name in cis_a:
        assert cis_a[metric_name] == cis_b[metric_name]


def test_bootstrap_regime_metrics_wider_ci_for_smaller_n():
    """Bootstrap on a small (n=6) subset produces a wider RMSE CI than on a large (n=30) subset."""
    df = _synthetic_predictions_df({"small": 6, "large": 30})
    y_train = _y_train_series()
    subset_small = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "small")
    ]
    subset_large = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "large")
    ]
    cis_small = bootstrap_regime_metrics(subset_small, y_train, n_bootstrap=500, random_state=42)
    cis_large = bootstrap_regime_metrics(subset_large, y_train, n_bootstrap=500, random_state=42)
    width_small = cis_small["rmse"][1] - cis_small["rmse"][0]
    width_large = cis_large["rmse"][1] - cis_large["rmse"][0]
    assert width_small > width_large, (
        f"expected smaller-n bootstrap to produce wider CI; "
        f"got width_small={width_small}, width_large={width_large}"
    )


def test_bootstrap_regime_metrics_raises_on_too_few_observations():
    """Bootstrap with n=1 raises ValueError."""
    df = _synthetic_predictions_df({"R": 1})
    y_train = _y_train_series()
    subset = df[
        (df["model"] == "ridge") & (df["scheme"] == "expanding_window") & (df["regime"] == "R")
    ]
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_regime_metrics(subset, y_train)
