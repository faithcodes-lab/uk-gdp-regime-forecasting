"""Tests for scripts/feature_ablation.py.

Only the pure partitioning logic is unit-tested here; the full CV run is
covered by the reproduction check in the script's own main() (it logs the
full-feature RMSE against the recorded main-evaluation value).
"""

from __future__ import annotations

from scripts.feature_ablation import _GDP_HISTORY_FEATURES, _feature_sets


def test_feature_sets_full_is_all_columns():
    """The 'full' variant is every column, unchanged and in order."""
    columns = ["gdp_growth", "unemployment_rate", "gdp_lag_1", "bank_rate"]
    variants = _feature_sets(columns)
    assert variants["full"] == columns


def test_feature_sets_gdp_history_matches_constant():
    """The 'gdp_history_only' variant is exactly _GDP_HISTORY_FEATURES."""
    columns = _GDP_HISTORY_FEATURES + ["unemployment_rate", "bank_rate"]
    variants = _feature_sets(columns)
    assert variants["gdp_history_only"] == _GDP_HISTORY_FEATURES


def test_feature_sets_macro_only_excludes_gdp_history():
    """The 'macro_only' variant contains none of the GDP-history columns."""
    columns = _GDP_HISTORY_FEATURES + ["unemployment_rate", "bank_rate", "yield_curve_slope"]
    variants = _feature_sets(columns)
    assert set(variants["macro_only"]).isdisjoint(_GDP_HISTORY_FEATURES)


def test_feature_sets_gdp_history_and_macro_only_partition_full():
    """gdp_history_only and macro_only together reconstruct 'full' with no overlap and no gaps."""
    columns = [
        "gdp_growth",
        "unemployment_rate",
        "cpi_inflation",
        "gdp_lag_1",
        "gdp_lag_4",
        "gdp_rolling_mean_4q",
        "gdp_yoy",
        "bank_rate",
        "yield_curve_slope",
    ]
    variants = _feature_sets(columns)
    combined = set(variants["gdp_history_only"]) | set(variants["macro_only"])
    assert combined == set(columns)
    assert set(variants["gdp_history_only"]).isdisjoint(variants["macro_only"])


def test_feature_sets_seventeen_columns_matches_recorded_split():
    """With the real 17-column feature matrix, the split is 5 GDP-history / 12 macro."""
    real_columns = [
        "gdp_growth",
        "unemployment_rate",
        "cpi_inflation",
        "trade_balance",
        "gfcf_growth",
        "govt_consumption_growth",
        "bank_rate",
        "gbp_usd_rate",
        "brent_oil",
        "business_confidence",
        "consumer_confidence",
        "gdp_lag_1",
        "gdp_lag_4",
        "gdp_rolling_mean_4q",
        "gdp_yoy",
        "business_confidence_rolling_mean_4q",
        "yield_curve_slope",
    ]
    variants = _feature_sets(real_columns)
    assert len(variants["full"]) == 17
    assert len(variants["gdp_history_only"]) == 5
    assert len(variants["macro_only"]) == 12
