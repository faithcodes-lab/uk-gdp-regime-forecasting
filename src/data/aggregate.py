"""Aggregate raw series to quarterly (QE) frequency.

Each downloader produces a (date, value) frame at the source's native
frequency. :func:to_quarterly reduces these to quarter-end observations,
applying the aggregation method declared in config/pipeline.yaml:

identity: input is already quarterly; aligns dates to the nearest QE.
quarterly_mean: average within each quarter.
end_of_period: last observed value within each quarter.
qoq_pct_change: quarter-on-quarter percent change of a quarterly level
    series (e.g. ONS chained-volume £m levels for GFCF and government
    consumption). First quarter is NaN.

Uses the modern "QE" resampling alias rather than the deprecated "Q"
"""

from __future__ import annotations

import pandas as pd

_VALID_METHODS = ("identity", "quarterly_mean", "end_of_period", "qoq_pct_change")


def to_quarterly(
    df: pd.DataFrame,
    *,
    method: str,
    date_col: str = "date",
    value_col: str = "value",
) -> pd.DataFrame:
    """Aggregate a (date, value) frame to quarter-end frequency.

    Args:
        df: Input frame with date and value columns.
        method: Aggregation method ("identity", "quarterly_mean",
            or "end_of_period").
        date_col: Name of the date column.
        value_col: Name of the value column.

    Returns:
        Two-column (date, value) frame indexed by quarter-end dates.

    Raises:
        ValueError: If method is not one of the three valid options.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown aggregation method: {method!r}. " f"Must be one of {list(_VALID_METHODS)}."
        )

    series = df.copy()
    series[date_col] = pd.to_datetime(series[date_col])
    indexed = series.set_index(date_col)[value_col]

    if method in {"identity", "end_of_period"}:
        result = indexed.resample("QE").last()
    elif method == "quarterly_mean":
        result = indexed.resample("QE").mean()
    else:  # qoq_pct_change
        # ONS level series (e.g. NPQT, NMRY) to QoQ % change.
        result = indexed.resample("QE").last().pct_change() * 100.0

    out = result.reset_index()
    out.columns = ["date", "value"]
    return out
