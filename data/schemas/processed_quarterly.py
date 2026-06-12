"""Validation rules for the final processed quarterly dataset.

This is the rulebook for the finished dataset, after every source has
been downloaded, merged, and turned into features. It checks a date
column plus 17 data columns: the GDP growth target, 10 raw predictors,
and 6 engineered features. An 18th data column, yield_curve_slope, is
optional, allowed but not required while it's deferred, and validated
once it's wired in.

The range limits on each column are deliberately wide. They aren't meant
to describe the data precisely, they're there to catch obvious
corruption (wrong units, a flipped sign, a column that's come through as
all zeros) and otherwise let the real values through.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

_RAW_KEPT_COLUMNS = {
    "unemployment_rate": Column(
        float,
        checks=Check.in_range(0.0, 25.0),
        nullable=False,
        description="UK unemployment rate (%).",
    ),
    "cpi_inflation": Column(
        float,
        checks=Check.in_range(-10.0, 30.0),
        nullable=False,
        description="UK CPI 12-month inflation rate (%).",
    ),
    "trade_balance": Column(
        float,
        nullable=False,
        description="UK trade balance (units per ONS series).",
    ),
    "gfcf_growth": Column(
        float,
        checks=Check.in_range(-50.0, 50.0),
        nullable=False,
        description="Gross fixed capital formation, quarterly growth (%).",
    ),
    "govt_consumption_growth": Column(
        float,
        checks=Check.in_range(-30.0, 30.0),
        nullable=False,
        description="General government final consumption expenditure, quarterly growth (%).",
    ),
    "bank_rate": Column(
        float,
        checks=Check.in_range(0.0, 25.0),
        nullable=False,
        description="Bank of England Bank Rate (%).",
    ),
    "gbp_usd_rate": Column(
        float,
        checks=Check.in_range(0.5, 3.0),
        nullable=False,
        description="GBP/USD spot exchange rate.",
    ),
    "brent_oil": Column(
        float,
        checks=Check.in_range(0.0, 300.0),
        nullable=False,
        description="Brent crude oil price (USD/barrel).",
    ),
    "business_confidence": Column(
        float,
        checks=Check.in_range(85.0, 115.0),
        nullable=False,
        description="OECD Business Confidence Indicator for the UK (PMI proxy).",
    ),
    "consumer_confidence": Column(
        float,
        checks=Check.in_range(85.0, 115.0),
        nullable=False,
        description="OECD Consumer Confidence Indicator for the UK.",
    ),
}

_ENGINEERED_COLUMNS = {
    "gdp_lag_1": Column(
        float,
        checks=Check.in_range(-25.0, 25.0),
        nullable=True,
        description="GDP growth lagged one quarter.",
    ),
    "gdp_lag_4": Column(
        float,
        checks=Check.in_range(-25.0, 25.0),
        nullable=True,
        description="GDP growth lagged four quarters.",
    ),
    "gdp_rolling_mean_4q": Column(
        float,
        checks=Check.in_range(-25.0, 25.0),
        nullable=True,
        description="Four-quarter rolling mean of GDP growth.",
    ),
    "gdp_yoy": Column(
        float,
        checks=Check.in_range(-30.0, 30.0),
        nullable=True,
        description="Year-on-year change in GDP.",
    ),
    "cpi_yoy": Column(
        float,
        checks=Check.in_range(-15.0, 35.0),
        nullable=True,
        description="Year-on-year change in CPI.",
    ),
    "business_confidence_rolling_mean_4q": Column(
        float,
        checks=Check.in_range(85.0, 115.0),
        nullable=True,
        description="Four-quarter rolling mean of business confidence.",
    ),
}

# Optional: validated only when present. Wired in once the slope definition
# is settled and the feature is added to the pipeline.
_OPTIONAL_COLUMNS = {
    "yield_curve_slope": Column(
        float,
        checks=Check.in_range(-10.0, 10.0),
        nullable=True,
        required=False,
        description="Yield curve slope (10-year gilt yield minus 2-year gilt yield).",
    ),
}

FINAL_DATASET_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            nullable=False,
            description="Quarter-end observation date.",
        ),
        "gdp_growth": Column(
            float,
            checks=Check.in_range(-25.0, 25.0),
            nullable=False,
            description=(
                "UK GDP quarter-on-quarter growth rate (target). "
                "Bound of -25 accommodates COVID Q2 2020."
            ),
        ),
        **_RAW_KEPT_COLUMNS,
        **_ENGINEERED_COLUMNS,
        **_OPTIONAL_COLUMNS,
    },
    strict=False,
    coerce=False,
)
