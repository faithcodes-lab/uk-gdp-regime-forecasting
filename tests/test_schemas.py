"""Tests for pandera schemas in :mod:`data.schemas`."""

from __future__ import annotations

import pandas as pd
import pandera.errors as pa_errors
import pytest

from data.schemas.processed_quarterly import FINAL_DATASET_SCHEMA
from data.schemas.raw_boe_rate import BOE_RATE_RAW_SCHEMA
from data.schemas.raw_fred import RAW_FRED_SCHEMA
from data.schemas.raw_ons_gdp import ONS_GDP_RAW_SCHEMA


def test_ons_gdp_accepts_valid_frame_including_covid_magnitude() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"]),
            "gdp_growth": [-2.0, -19.4, 17.0],
        }
    )
    ONS_GDP_RAW_SCHEMA.validate(df)


def test_ons_gdp_rejects_value_below_lower_bound() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "gdp_growth": [-30.0],
        }
    )
    with pytest.raises(pa_errors.SchemaError):
        ONS_GDP_RAW_SCHEMA.validate(df)


def test_ons_gdp_rejects_value_above_upper_bound() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-09-30"]),
            "gdp_growth": [30.0],
        }
    )
    with pytest.raises(pa_errors.SchemaError):
        ONS_GDP_RAW_SCHEMA.validate(df)


# Raw BoE Bank Rate


def test_boe_rate_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "bank_rate": [0.1],
        }
    )
    BOE_RATE_RAW_SCHEMA.validate(df)


def test_boe_rate_rejects_negative_rate() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "bank_rate": [-0.5],
        }
    )
    with pytest.raises(pa_errors.SchemaError):
        BOE_RATE_RAW_SCHEMA.validate(df)


# Generic raw FRED


def test_fred_accepts_valid_frame() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "value": [50.0],
        }
    )
    RAW_FRED_SCHEMA.validate(df)


def test_fred_rejects_null_value() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "value": [float("nan")],
        }
    )
    with pytest.raises(pa_errors.SchemaError):
        RAW_FRED_SCHEMA.validate(df)


# Processed quarterly


def _synthetic_processed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30"]),
            "gdp_growth": [-2.0, -19.4],
            "unemployment_rate": [4.0, 4.5],
            "cpi_inflation": [1.5, 0.6],
            "trade_balance": [-5000.0, -4000.0],
            "gfcf_growth": [-1.0, -15.0],
            "govt_consumption_growth": [0.5, 6.5],
            "bank_rate": [0.75, 0.10],
            "gbp_usd_rate": [1.25, 1.24],
            "brent_oil": [50.0, 30.0],
            "business_confidence": [-4.5, -23.5],
            "consumer_confidence": [-12.0, -30.0],
            "gdp_lag_1": [0.5, -2.0],
            "gdp_lag_4": [0.6, 0.4],
            "gdp_rolling_mean_4q": [0.2, -5.0],
            "gdp_yoy": [1.5, -20.0],
            "business_confidence_rolling_mean_4q": [-3.0, -15.0],
            "yield_curve_slope": [0.5, 0.3],
        }
    )


def test_processed_quarterly_accepts_synthetic_17_column_frame() -> None:
    FINAL_DATASET_SCHEMA.validate(_synthetic_processed_frame())


def test_processed_quarterly_rejects_missing_target() -> None:
    df = _synthetic_processed_frame().drop(columns=["gdp_growth"])
    with pytest.raises(pa_errors.SchemaError):
        FINAL_DATASET_SCHEMA.validate(df)


def test_processed_quarterly_rejects_missing_raw_predictor() -> None:
    df = _synthetic_processed_frame().drop(columns=["bank_rate"])
    with pytest.raises(pa_errors.SchemaError):
        FINAL_DATASET_SCHEMA.validate(df)


def test_processed_quarterly_rejects_missing_yield_curve_slope() -> None:
    df = _synthetic_processed_frame().drop(columns=["yield_curve_slope"])
    with pytest.raises(pa_errors.SchemaError):
        FINAL_DATASET_SCHEMA.validate(df)


def test_processed_quarterly_rejects_out_of_range_target() -> None:
    df = _synthetic_processed_frame()
    df.loc[0, "gdp_growth"] = -50.0
    with pytest.raises(pa_errors.SchemaError):
        FINAL_DATASET_SCHEMA.validate(df)
