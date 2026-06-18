"""Tests for src/regimes/assign.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.regimes.assign import assign_regimes


def _real_regimes() -> list[dict]:
    """The six project regimes, mirroring config/regimes.yaml exactly."""
    return [
        {
            "id": 1,
            "label": "Pre-GFC Stability",
            "start": pd.Timestamp("2000-01-01"),
            "end": pd.Timestamp("2007-12-31"),
        },
        {
            "id": 2,
            "label": "Global Financial Crisis",
            "start": pd.Timestamp("2008-01-01"),
            "end": pd.Timestamp("2009-12-31"),
        },
        {
            "id": 3,
            "label": "Post-GFC Recovery",
            "start": pd.Timestamp("2010-01-01"),
            "end": pd.Timestamp("2016-06-30"),
        },
        {
            "id": 4,
            "label": "Brexit Transition",
            "start": pd.Timestamp("2016-07-01"),
            "end": pd.Timestamp("2019-12-31"),
        },
        {
            "id": 5,
            "label": "COVID-19 Shock",
            "start": pd.Timestamp("2020-01-01"),
            "end": pd.Timestamp("2021-06-30"),
        },
        {
            "id": 6,
            "label": "Post-COVID Recovery",
            "start": pd.Timestamp("2021-07-01"),
            "end": pd.Timestamp("2025-12-31"),
        },
    ]


def _quarter_end_dates(start: str = "2000-01-01", periods: int = 104) -> pd.Series:
    """Quarter-end date series mirroring the project dataset's basis."""
    qe = pd.period_range(start=start, periods=periods,
                         freq="Q").to_timestamp(how="end").normalize()
    return pd.Series(qe, name="date")


def test_assigns_104_rows_into_six_regimes_with_correct_counts():
    """Full 104-quarter project window: 6 regimes with the counts in regimes.yaml."""
    dates = _quarter_end_dates(start="2000-01-01", periods=104)
    df = pd.DataFrame({"date": dates, "x": range(104)})
    out = assign_regimes(df, regimes=_real_regimes())
    assert out["regime"].nunique() == 6
    counts = out["regime"].value_counts()
    assert counts["Pre-GFC Stability"] == 32
    assert counts["Global Financial Crisis"] == 8
    assert counts["Post-GFC Recovery"] == 26
    assert counts["Brexit Transition"] == 14
    assert counts["COVID-19 Shock"] == 6
    assert counts["Post-COVID Recovery"] == 18


def test_regime_column_is_inserted_directly_after_date():
    """Output column order: date, regime, then everything else."""
    df = pd.DataFrame({"date": _quarter_end_dates(periods=4),
                      "x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    out = assign_regimes(df, regimes=_real_regimes())
    assert list(out.columns) == ["date", "regime", "x", "y"]


def test_boundary_quarter_routes_to_post_regime_for_gfc():
    """2007 Q4 stays Pre-GFC; 2008 Q1 jumps to GFC even with quarter-end input."""
    df = pd.DataFrame({"date": pd.to_datetime(["2007-12-31", "2008-03-31"])})
    out = assign_regimes(df, regimes=_real_regimes())
    assert list(out["regime"]) == [
        "Pre-GFC Stability", "Global Financial Crisis"]


def test_all_four_remaining_boundaries():
    """Spot-check each boundary quarter pair (Brexit, COVID, Post-COVID, Post-GFC)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2009-12-31",  # last GFC quarter
                    "2010-03-31",  # first Post-GFC
                    "2016-06-30",  # last Post-GFC
                    "2016-09-30",  # first Brexit
                    "2019-12-31",  # last Brexit
                    "2020-03-31",  # first COVID
                    "2021-06-30",  # last COVID
                    "2021-09-30",  # first Post-COVID
                ]
            )
        }
    )
    out = assign_regimes(df, regimes=_real_regimes())
    assert list(out["regime"]) == [
        "Global Financial Crisis",
        "Post-GFC Recovery",
        "Post-GFC Recovery",
        "Brexit Transition",
        "Brexit Transition",
        "COVID-19 Shock",
        "COVID-19 Shock",
        "Post-COVID Recovery",
    ]


def test_accepts_full_config_dict():
    """Passing the parsed regimes_config() dict works."""
    config = {"regimes": _real_regimes()}
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"])})
    out = assign_regimes(df, regimes=config)
    assert out["regime"].iloc[0] == "Post-GFC Recovery"


def test_accepts_none_loads_from_project_config():
    """Passing None loads via src.data.config.regimes_config()."""
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"])})
    out = assign_regimes(df, regimes=None)
    assert out["regime"].iloc[0] == "Post-GFC Recovery"


def test_input_dataframe_is_not_mutated():
    """The function must not modify the caller's DataFrame."""
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"]), "x": [42]})
    before_cols = list(df.columns)
    assign_regimes(df, regimes=_real_regimes())
    assert list(df.columns) == before_cols
    assert "regime" not in df.columns


def test_missing_date_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="date"):
        assign_regimes(df, regimes=_real_regimes())


def test_row_outside_regime_union_raises():
    """A row dated before the first regime should raise."""
    df = pd.DataFrame({"date": pd.to_datetime(["1995-03-31"])})
    with pytest.raises(ValueError, match="outside the regime union"):
        assign_regimes(df, regimes=_real_regimes())


def test_overlap_in_config_raises():
    bad = [
        {
            "label": "A",
            "start": pd.Timestamp("2000-01-01"),
            "end": pd.Timestamp("2005-12-31"),
        },
        {
            "label": "B",
            "start": pd.Timestamp("2005-10-01"),  # overlaps A's last quarter
            "end": pd.Timestamp("2010-12-31"),
        },
    ]
    df = pd.DataFrame({"date": pd.to_datetime(["2003-03-31"])})
    with pytest.raises(ValueError, match="overlap"):
        assign_regimes(df, regimes=bad)


def test_gap_in_config_raises():
    bad = [
        {
            "label": "A",
            "start": pd.Timestamp("2000-01-01"),
            "end": pd.Timestamp("2005-12-31"),
        },
        {
            "label": "B",
            "start": pd.Timestamp("2007-01-01"),  # skips all of 2006
            "end": pd.Timestamp("2010-12-31"),
        },
    ]
    df = pd.DataFrame({"date": pd.to_datetime(["2003-03-31"])})
    with pytest.raises(ValueError, match="Gap"):
        assign_regimes(df, regimes=bad)


def test_inverted_regime_raises():
    """A regime with end before start should raise."""
    bad = [
        {
            "label": "A",
            "start": pd.Timestamp("2005-01-01"),
            "end": pd.Timestamp("2000-12-31"),
        },
    ]
    df = pd.DataFrame({"date": pd.to_datetime(["2003-03-31"])})
    with pytest.raises(ValueError, match="end"):
        assign_regimes(df, regimes=bad)


def test_empty_regimes_list_raises():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"])})
    with pytest.raises(ValueError, match="empty"):
        assign_regimes(df, regimes=[])


def test_dict_without_regimes_key_raises():
    df = pd.DataFrame({"date": pd.to_datetime(["2010-03-31"])})
    with pytest.raises(ValueError, match="regimes"):
        assign_regimes(df, regimes={"other_key": []})
