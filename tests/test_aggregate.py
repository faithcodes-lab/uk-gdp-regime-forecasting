"""Tests for :func:`src.data.aggregate.to_quarterly`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.aggregate import to_quarterly


def test_quarterly_mean_on_monthly_input() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "value": [10.0, 20.0, 30.0],
        }
    )
    result = to_quarterly(df, method="quarterly_mean")
    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2020-03-31")
    assert result["value"].iloc[0] == 20.0


def test_end_of_period_on_monthly_input() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "value": [10.0, 20.0, 30.0],
        }
    )
    result = to_quarterly(df, method="end_of_period")
    assert result["value"].iloc[0] == 30.0


def test_identity_passes_quarterly_input_through() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"]),
            "value": [1.0, 2.0, 3.0],
        }
    )
    result = to_quarterly(df, method="identity")
    assert len(result) == 3
    assert list(result["value"]) == [1.0, 2.0, 3.0]
    assert list(result["date"]) == list(df["date"])


def test_rejects_unknown_method() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-03-31"]), "value": [1.0]})
    with pytest.raises(ValueError, match="Unknown aggregation method"):
        to_quarterly(df, method="quartile_max")


def test_qe_alias_produces_quarter_end_dates() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2020-12-31", freq="MS"),
            "value": range(12),
        }
    )
    result = to_quarterly(df, method="quarterly_mean")
    months = [d.month for d in result["date"]]
    assert months == [3, 6, 9, 12]
    # 31-Mar / 30-Jun / 30-Sep / 31-Dec — all last day of month
    assert (result["date"] == result["date"] + pd.offsets.MonthEnd(0)).all()


def test_quarterly_mean_on_daily_input() -> None:
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    df = pd.DataFrame({"date": dates, "value": [1.0] * len(dates)})
    result = to_quarterly(df, method="quarterly_mean")
    assert result["value"].iloc[0] == 1.0
