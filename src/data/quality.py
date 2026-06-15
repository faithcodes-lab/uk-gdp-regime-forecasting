"""Quality checks on the merged final dataset.

Run at the end of the pipeline (after merge plus engineering, before save)
to catch look ahead leakage and dataset shape problems that would
otherwise propagate silently to model training. Each check raises
``:class:QualityCheckFailed`` on failure; the caller is expected to let the
exception propagate so the pipeline fails loudly rather than producing a
corrupted dataset.
"""

from __future__ import annotations

import pandas as pd

_QUARTER_END_MONTHS = {3, 6, 9, 12}
_COVID_Q2_2020 = pd.Timestamp("2020-06-30")

# COVID Q2 2020 GDP growth is roughly -19.4%. -10 sits comfortably above that
# while still being well below any plausible non-shock quarter, so a value
# ≥ -10 at that row is a strong signal of corruption (wrong sign, wrong units,
# zero-fill) rather than a genuine economy reading.

_COVID_SHOCK_CEILING = -10.0


class QualityCheckFailed(Exception):
    """Raised when a quality check on the final dataset fails."""


def check_final_dataset(df: pd.DataFrame) -> None:
    """Run all quality checks on the final quarterly dataset.

    Args:
        df: Final merged plus engineered quarterly frame.

    Raises:
        QualityCheckFailed: With a message identifying which check failed.
    """
    # Ordering matters: the lag/rolling/COVID checks all assume the dataset
    # is well-formed (sorted quarter-end dates, no gaps), so it can verify the
    # shape first and only run the look-ahead invariants once those pass.
    _check_date_column_present(df)
    _check_quarter_end_dates(df)
    _check_no_missing_quarters(df)
    _check_lag_invariants(df)
    _check_rolling_invariants(df)
    _check_covid_outlier_present(df)


def _check_date_column_present(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise QualityCheckFailed("Dataset has no 'date' column.")


def _check_quarter_end_dates(df: pd.DataFrame) -> None:
    dates = pd.to_datetime(df["date"])
    bad_month = ~dates.dt.month.isin(_QUARTER_END_MONTHS)
    if bad_month.any():
        n = int(bad_month.sum())
        raise QualityCheckFailed(f"{n} rows have dates not in March/June/September/December.")
    last_day = dates + pd.offsets.MonthEnd(0)
    if not (dates == last_day).all():
        n = int((dates != last_day).sum())
        raise QualityCheckFailed(f"{n} rows have non-month-end dates.")


def _check_no_missing_quarters(df: pd.DataFrame) -> None:
    dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)
    expected = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq="QE")
    missing = expected.difference(dates)
    if len(missing) > 0:
        raise QualityCheckFailed(
            f"Missing {len(missing)} quarter(s) between "
            f"{dates.iloc[0].date()} and {dates.iloc[-1].date()}: "
            f"first missing = {missing[0].date()}."
        )


def _check_lag_invariants(df: pd.DataFrame) -> None:
    """gdp_lag_k[t] must equal gdp_growth[t-k] at every valid row."""
    if "gdp_growth" not in df.columns:
        return
    df_sorted = df.sort_values("date").reset_index(drop=True)
    growth = df_sorted["gdp_growth"]
    for lag in (1, 4):
        col = f"gdp_lag_{lag}"
        if col not in df_sorted.columns:
            continue
        expected = growth.shift(lag)
        actual = df_sorted[col]
        mask = ~(expected.isna() | actual.isna())
        if not mask.any():
            continue
        if not (expected[mask].to_numpy() == actual[mask].to_numpy()).all():
            raise QualityCheckFailed(
                f"{col} does not equal gdp_growth.shift({lag}) at all valid rows."
            )


def _check_rolling_invariants(df: pd.DataFrame) -> None:
    """gdp_rolling_mean_4q[t] must equal mean(gdp_growth[t-3..t])."""
    if "gdp_growth" not in df.columns or "gdp_rolling_mean_4q" not in df.columns:
        return
    df_sorted = df.sort_values("date").reset_index(drop=True)
    expected = df_sorted["gdp_growth"].rolling(window=4, min_periods=4).mean()
    actual = df_sorted["gdp_rolling_mean_4q"]
    mask = ~(expected.isna() | actual.isna())
    if not mask.any():
        return
    if not (expected[mask].round(10).to_numpy() == actual[mask].round(10).to_numpy()).all():
        raise QualityCheckFailed(
            "gdp_rolling_mean_4q does not equal "
            "gdp_growth.rolling(4, min_periods=4).mean() at all valid rows."
        )


def _check_covid_outlier_present(df: pd.DataFrame) -> None:
    """COVID Q2 2020 GDP growth must be present and below -10%."""
    if "gdp_growth" not in df.columns or "date" not in df.columns:
        return
    dates = pd.to_datetime(df["date"])
    covid = df.loc[dates == _COVID_Q2_2020]
    if len(covid) == 0:
        raise QualityCheckFailed("COVID Q2 2020 row (2020-06-30) missing from dataset.")
    value = float(covid["gdp_growth"].iloc[0])
    if not value < _COVID_SHOCK_CEILING:
        raise QualityCheckFailed(
            f"COVID Q2 2020 gdp_growth = {value} is not < "
            f"{_COVID_SHOCK_CEILING}; expected a sharp negative shock, "
            "data may be corrupted."
        )
