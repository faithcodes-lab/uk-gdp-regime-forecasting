"""Tests for the configuration loaders in :mod:`src.data.config`."""

from __future__ import annotations

import datetime as dt

from src.data.config import features_config, pipeline_config, regimes_config

EXPECTED_REGIME_LABELS = [
    "Pre-GFC Stability",
    "Global Financial Crisis",
    "Post-GFC Recovery",
    "Brexit Transition",
    "COVID-19 Shock",
    "Post-COVID Recovery",
]

EXPECTED_ENGINEERED_NAMES = {
    "gdp_lag_1",
    "gdp_lag_4",
    "gdp_rolling_mean_4q",
    "gdp_yoy",
    "cpi_yoy",
    "business_confidence_rolling_mean_4q",
    "yield_curve_slope",
}


# pipeline.yaml

def test_pipeline_config_loads_as_dict() -> None:
    assert isinstance(pipeline_config(), dict)


def test_pipeline_config_has_required_top_level_keys() -> None:
    cfg = pipeline_config()
    for key in ("project", "paths", "data_sources"):
        assert key in cfg, f"missing top-level key: {key}"


def test_pipeline_random_seed_is_42() -> None:
    assert pipeline_config()["project"]["random_state"] == 42


def test_pipeline_data_sources_are_the_four_expected() -> None:
    sources = set(pipeline_config()["data_sources"].keys())
    assert sources == {"ons", "boe", "fred"}


def test_pipeline_paths_include_final_dataset() -> None:
    paths = pipeline_config()["paths"]
    assert paths["final_dataset"] == "data/processed/final_dataset.parquet"

# features.yaml


def test_features_target_is_gdp_growth() -> None:
    assert features_config()["target"]["name"] == "gdp_growth"


def test_features_engineered_count_is_seven() -> None:
    assert len(features_config()["engineered"]) == 7


def test_features_engineered_names_match_spec() -> None:
    names = {item["name"] for item in features_config()["engineered"]}
    assert names == EXPECTED_ENGINEERED_NAMES


def test_features_intermediate_only_lists_both_gilts() -> None:
    intermediate = set(features_config()["intermediate_only_then_dropped"])
    assert intermediate == {"gilt_2y_yield", "gilt_10y_yield"}


def test_features_final_column_count_is_17_while_slope_deferred() -> None:
    assert features_config()["final_column_count"] == 17


def test_features_final_column_count_returns_to_18_when_slope_wired() -> None:
    assert features_config()["final_column_count_when_slope_wired"] == 18


def test_yield_curve_slope_is_deferred() -> None:
    engineered = features_config()["engineered"]
    slope = next(
        item for item in engineered if item["name"] == "yield_curve_slope")
    assert slope.get("deferred") is True


def test_active_engineered_count_is_six() -> None:
    engineered = features_config()["engineered"]
    active = [item for item in engineered if not item.get("deferred", False)]
    assert len(active) == 6


def test_gilt_2y_yield_is_deferred_in_pipeline_config() -> None:
    boe_series = pipeline_config()["data_sources"]["boe"]["series"]
    assert boe_series["gilt_2y_yield"].get("deferred") is True


def test_gilt_10y_yield_is_not_deferred() -> None:
    boe_or_fred = pipeline_config()["data_sources"]["fred"]["series"]
    assert boe_or_fred["gilt_10y_yield"].get("deferred", False) is False

# regimes.yaml


def test_regimes_count_is_six() -> None:
    assert len(regimes_config()["regimes"]) == 6


def test_regime_labels_match_canonical_order() -> None:
    labels = [r["label"] for r in regimes_config()["regimes"]]
    assert labels == EXPECTED_REGIME_LABELS


def test_regime_dates_parse_as_dates() -> None:
    for r in regimes_config()["regimes"]:
        assert isinstance(r["start"], dt.date)
        assert isinstance(r["end"], dt.date)


def test_regimes_are_ordered_and_non_overlapping() -> None:
    regimes = regimes_config()["regimes"]
    for current, nxt in zip(regimes, regimes[1:]):
        assert current["end"] < nxt["start"], (
            f"Regime {current['label']} ends {current['end']} but "
            f"{nxt['label']} starts {nxt['start']}"
        )


def test_regimes_cover_full_study_period() -> None:
    regimes = regimes_config()["regimes"]
    assert regimes[0]["start"] == dt.date(2000, 1, 1)
    assert regimes[-1]["end"] == dt.date(2025, 12, 31)


def test_regime_quarter_counts_sum_to_104() -> None:
    total = sum(r["quarters"] for r in regimes_config()["regimes"])
    assert total == 104
