"""Tests for :mod:`src.features.engineer`.

Includes the critical look-ahead-leakage tests required by the task file's
acceptance criteria. These tests are the load-bearing safeguard for the
project's methodological integrity, a passing look-ahead test does not
guarantee correctness, but a failing one means the model would be
trained on future information.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.engineer import (
    _cumulative_compounded_growth,
    engineer_features,
)


def _full_features_cfg() -> dict:
    """A complete features config covering all transformation paths."""
    return {
        "engineered": [
            {
                "name": "gdp_lag_1",
                "source": "gdp_growth",
                "transformation": "lag",
                "periods": 1,
            },
            {
                "name": "gdp_lag_4",
                "source": "gdp_growth",
                "transformation": "lag",
                "periods": 4,
            },
            {
                "name": "gdp_rolling_mean_4q",
                "source": "gdp_growth",
                "transformation": "rolling_mean",
                "window": 4,
            },
            {
                "name": "gdp_yoy",
                "source": "gdp_growth",
                "transformation": "year_over_year",
                "periods": 4,
            },
            {
                "name": "business_confidence_rolling_mean_4q",
                "source": "business_confidence",
                "transformation": "rolling_mean",
                "window": 4,
            },
            {
                "name": "yield_curve_slope",
                "sources": ["gilt_10y_yield", "gilt_2y_yield"],
                "transformation": "subtract",
            },
        ],
        "intermediate_only_then_dropped": ["gilt_2y_yield", "gilt_10y_yield"],
    }


def _basic_input_frame(n: int = 12) -> pd.DataFrame:
    """Synthetic quarterly frame with deterministic raw values."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=n, freq="QE"),
            "gdp_growth": [float(i) for i in range(1, n + 1)],
            "business_confidence": [100.0 + i for i in range(n)],
            "gilt_2y_yield": [0.5 + i * 0.01 for i in range(n)],
            "gilt_10y_yield": [1.5 + i * 0.01 for i in range(n)],
        }
    )


# Look-ahead leakage


def test_lag_features_are_truly_lagged() -> None:
    df = _basic_input_frame()
    out = engineer_features(df, features_cfg=_full_features_cfg())
    growth = df["gdp_growth"].to_numpy()

    assert pd.isna(out["gdp_lag_1"].iloc[0])
    for i in range(1, len(df)):
        assert out["gdp_lag_1"].iloc[i] == growth[i - 1]

    for i in range(4):
        assert pd.isna(out["gdp_lag_4"].iloc[i])
    for i in range(4, len(df)):
        assert out["gdp_lag_4"].iloc[i] == growth[i - 4]


def test_rolling_uses_only_past_and_present() -> None:
    """Mutating ``gdp_growth[t+1]`` must not change ``gdp_rolling_mean_4q[t]``."""
    df_clean = _basic_input_frame()
    df_mut = df_clean.copy()
    df_mut.loc[5, "gdp_growth"] = 999.0  # mutate one future row

    out_clean = engineer_features(df_clean, features_cfg=_full_features_cfg())
    out_mut = engineer_features(df_mut, features_cfg=_full_features_cfg())

    # gdp_rolling_mean_4q at t in {0..4} uses only gdp_growth[t-3..t];
    # mutating gdp_growth[5] must not affect those.
    for t in range(5):
        a = out_clean["gdp_rolling_mean_4q"].iloc[t]
        b = out_mut["gdp_rolling_mean_4q"].iloc[t]
        assert (pd.isna(a) and pd.isna(b)) or a == b


def test_no_future_leakage_across_pipeline() -> None:
    """End-to-end: poisoning a future row must not influence any past row's features."""
    df_clean = _basic_input_frame()
    df_poisoned = df_clean.copy()
    last_index = df_poisoned.index[-1]
    df_poisoned.loc[
        last_index,
        ["gdp_growth", "business_confidence", "gilt_2y_yield", "gilt_10y_yield"],
    ] = 9999.0

    out_clean = engineer_features(df_clean, features_cfg=_full_features_cfg())
    out_poisoned = engineer_features(df_poisoned, features_cfg=_full_features_cfg())

    n = len(df_clean)
    engineered_cols = [
        "gdp_lag_1",
        "gdp_lag_4",
        "gdp_rolling_mean_4q",
        "gdp_yoy",
        "business_confidence_rolling_mean_4q",
        "yield_curve_slope",
    ]
    for col in engineered_cols:
        for t in range(n - 1):
            a = out_clean[col].iloc[t]
            b = out_poisoned[col].iloc[t]
            assert (
                pd.isna(a) and pd.isna(b)
            ) or a == b, f"{col} at t={t} differs after poisoning last row: {a} vs {b}"


# gdp_yoy: compounded formula


def test_gdp_yoy_matches_explicit_compounded_formula() -> None:
    df = _basic_input_frame(n=12)
    growth = df["gdp_growth"].to_numpy()
    g_dec = growth / 100.0

    out = engineer_features(df, features_cfg=_full_features_cfg())

    # Window at t covers t-3..t; first valid t is 3.
    for t in range(3, 12):
        expected_decimal = (1 + g_dec[t - 3]) * (1 + g_dec[t - 2]) * (1 + g_dec[t - 1]) * (
            1 + g_dec[t]
        ) - 1
        expected_pct = expected_decimal * 100.0
        assert abs(out["gdp_yoy"].iloc[t] - expected_pct) < 1e-9


def test_gdp_yoy_first_three_quarters_are_nan_first_four_minus_one_valid() -> None:
    """With window=4 and min_periods=4, indices 0,1,2 are NaN; index 3 is the first valid output."""
    df = _basic_input_frame()
    out = engineer_features(df, features_cfg=_full_features_cfg())
    assert out["gdp_yoy"].iloc[:3].isna().all()
    assert not pd.isna(out["gdp_yoy"].iloc[3])


def test_compounded_growth_helper_isolated() -> None:
    """Direct test of :func:`_cumulative_compounded_growth` with a known answer."""
    series = pd.Series([2.0, 2.0, 2.0, 2.0])
    # (1.02)^4 - 1 = 0.08243216; * 100 = ~8.2432%
    result = _cumulative_compounded_growth(series, periods=4)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert abs(result.iloc[3] - 8.24321600) < 1e-6


# Intermediate only columns are dropped


def test_drops_intermediate_only_columns() -> None:
    df = _basic_input_frame()
    out = engineer_features(df, features_cfg=_full_features_cfg())
    assert "gilt_2y_yield" not in out.columns
    assert "gilt_10y_yield" not in out.columns


def test_engineered_feature_names_match_active_spec() -> None:
    df = _basic_input_frame()
    out = engineer_features(df, features_cfg=_full_features_cfg())
    expected_engineered = {
        "gdp_lag_1",
        "gdp_lag_4",
        "gdp_rolling_mean_4q",
        "gdp_yoy",
        "business_confidence_rolling_mean_4q",
        "yield_curve_slope",
    }
    new_cols = set(out.columns) - {"date", "gdp_growth", "business_confidence"}
    assert new_cols == expected_engineered


def test_rejects_unknown_transformation() -> None:
    bad_cfg = {
        "engineered": [
            {
                "name": "bad",
                "source": "gdp_growth",
                "transformation": "fourier_transform",
            },
        ],
        "intermediate_only_then_dropped": [],
    }
    df = _basic_input_frame()
    with pytest.raises(ValueError, match="Unknown engineered transformation"):
        engineer_features(df, features_cfg=bad_cfg)
