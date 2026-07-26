"""Tests for :mod:`src.data.quality`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import QualityCheckFailed, check_final_dataset


def _consistent_frame(
    *,
    covid_value: float = -19.4,
    include_covid_row: bool = True,
) -> pd.DataFrame:
    """A self-consistent synthetic dataset with the lag/rolling columns derived from gdp_growth.

    Args:
        covid_value: Value placed at the COVID Q2 2020 row. Default is the
            sharp drop that satisfies the COVID-shock check.
        include_covid_row: If False, the COVID Q2 2020 row is removed entirely
            (creates a quarter gap, useful for missing-row tests).
    """
    dates = pd.date_range("2018-03-31", "2021-12-31", freq="QE")
    n = len(dates)
    rng = np.random.default_rng(42)
    gdp = rng.normal(0.5, 1.0, n)
    covid_idx = dates.get_loc(pd.Timestamp("2020-06-30"))
    gdp[covid_idx] = covid_value

    df = pd.DataFrame({"date": dates, "gdp_growth": gdp})
    if not include_covid_row:
        df = df.loc[df["date"] != pd.Timestamp("2020-06-30")].reset_index(drop=True)

    df["gdp_lag_1"] = df["gdp_growth"].shift(1)
    df["gdp_lag_4"] = df["gdp_growth"].shift(4)
    df["gdp_rolling_mean_4q"] = df["gdp_growth"].rolling(window=4, min_periods=4).mean()
    return df


def _valid_synthetic_frame() -> pd.DataFrame:
    """A minimal synthetic final dataset that satisfies every quality check."""
    return _consistent_frame()


def test_accepts_valid_synthetic_frame() -> None:
    check_final_dataset(_valid_synthetic_frame())


def test_rejects_missing_date_column() -> None:
    df = _valid_synthetic_frame().drop(columns=["date"])
    with pytest.raises(QualityCheckFailed, match="date"):
        check_final_dataset(df)


def test_rejects_non_quarter_end_dates() -> None:
    df = _valid_synthetic_frame()
    df.loc[0, "date"] = pd.Timestamp("2018-03-15")
    with pytest.raises(QualityCheckFailed, match="March/June/September/December|month-end"):
        check_final_dataset(df)


def test_rejects_dates_in_non_quarter_months() -> None:
    df = _valid_synthetic_frame()
    df.loc[0, "date"] = pd.Timestamp("2018-04-30")
    with pytest.raises(QualityCheckFailed, match="March/June/September/December"):
        check_final_dataset(df)


def test_rejects_missing_quarters() -> None:
    df = _valid_synthetic_frame()
    df = df.drop(index=df.index[5]).reset_index(drop=True)
    with pytest.raises(QualityCheckFailed, match="Missing"):
        check_final_dataset(df)


def test_covid_outlier_detection_works() -> None:
    """A self-consistent frame with a benign COVID value triggers the COVID check."""
    df = _consistent_frame(covid_value=0.5)
    with pytest.raises(QualityCheckFailed, match="COVID Q2 2020"):
        check_final_dataset(df)


def test_covid_outlier_missing_row_is_caught() -> None:
    """Removing the COVID row creates a quarter gap; missing-quarter check fires first."""
    df = _consistent_frame(include_covid_row=False)
    with pytest.raises(QualityCheckFailed, match="Missing|COVID Q2 2020"):
        check_final_dataset(df)


def test_lag_invariant_violation_is_caught() -> None:
    df = _valid_synthetic_frame()
    df.loc[5, "gdp_lag_1"] = df.loc[5, "gdp_lag_1"] + 100.0
    with pytest.raises(QualityCheckFailed, match="gdp_lag_1"):
        check_final_dataset(df)


def test_lag4_invariant_violation_is_caught() -> None:
    df = _valid_synthetic_frame()
    df.loc[8, "gdp_lag_4"] = df.loc[8, "gdp_lag_4"] + 100.0
    with pytest.raises(QualityCheckFailed, match="gdp_lag_4"):
        check_final_dataset(df)


def test_rolling_invariant_violation_is_caught() -> None:
    df = _valid_synthetic_frame()
    df.loc[10, "gdp_rolling_mean_4q"] = 999.0
    with pytest.raises(QualityCheckFailed, match="gdp_rolling_mean_4q"):
        check_final_dataset(df)
