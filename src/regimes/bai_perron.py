"""Multi-breakpoint detection via the ``ruptures`` library.

Two algorithms are exposed:

:func:`detect_breaks_pelt` runs the PELT algorithm with a penalty
  parameter; the number of breaks is selected by the algorithm.
:func:`detect_breaks_binseg` runs Binary Segmentation for a known
  number of breaks (useful as a top-down counterpart to PELT).

A third helper, :func:`tune_penalty`, runs PELT across a grid of penalty
values and returns a single-table summary of the breakpoints detected at
each value.

Conventions:

- Input is a single :class:`pandas.Series` with a
  :class:`pandas.DatetimeIndex`; the dataset's ``date`` column should
  become the index before calling these functions.
- The ``ruptures`` ``predict()`` API returns a list of breakpoint indices
  that always ends with ``len(signal)`` (the "end of signal" sentinel).
  The sentinel is dropped; each remaining index ``i`` is mapped to
  ``series.index[i]``, the first observation of the new segment. This
  matches the Chow-test convention: a break date marks the start of the
  post-period regime.
- Rows where the value is NaN are dropped before fitting; the loguru
  logger records the drop count when any rows are removed.
- If, after NaN handling, the cleaned series has fewer than four points,
  the function returns an empty list with a warning. The PELT algorithm
  needs a few points either side of any candidate break, and a sweep
  over such a short series produces noise rather than signal.
"""

from __future__ import annotations

import pandas as pd
import ruptures as rpt
from loguru import logger

# Default penalty grid for tune_penalty. Chosen per the task-file
# guidance (penalties of 5, 10, 15, 20, 30 typically span "many noisy
# breaks" through "one or two strong breaks" on quarterly macro data).
# CP4 may widen this grid if every value here yields the same number of
# breaks on the real series.
_DEFAULT_PENALTIES: list[float] = [5.0, 10.0, 15.0, 20.0, 30.0]

# Minimum length of the cleaned series before the algorithms are
# considered meaningful. Two segments of two points each is the smallest
# arrangement PELT can credibly partition.
_MIN_POINTS = 4


def _prepare_signal(series: pd.Series) -> pd.Series:
    """Validate the input and return a NaN-free copy.

    Raises:
        ValueError: If ``series.index`` is not a ``DatetimeIndex``.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex; got " f"{type(series.index).__name__}")
    n_original = len(series)
    clean = series.dropna()
    n_dropped = n_original - len(clean)
    if n_dropped > 0:
        logger.warning(
            "bai_perron: dropped {} NaN values from a series of {}",
            n_dropped,
            n_original,
        )
    return clean


def _indices_to_dates(
    raw_breakpoints: list[int],
    clean: pd.Series,
) -> list[pd.Timestamp]:
    """Drop the end-of-signal sentinel and map each index to a date."""
    return [pd.Timestamp(clean.index[i]) for i in raw_breakpoints if i < len(clean)]


def detect_breaks_pelt(
    series: pd.Series,
    penalty: float,
    model: str = "rbf",
) -> list[pd.Timestamp]:
    """Detect breakpoint dates in ``series`` via the PELT algorithm.

    Args:
        series: Univariate time series with a ``DatetimeIndex``.
        penalty: PELT penalty value. Higher values produce fewer breaks.
        model: Cost model passed to ``ruptures.Pelt``. Defaults to
            ``"rbf"`` (radial basis function kernel), which is sensitive
            to changes in distribution rather than only the mean.

    Returns:
        Sorted list of break dates. Empty list if no breaks are
        detected, the cleaned series is too short, or every value is NaN.
    """
    clean = _prepare_signal(series)
    if len(clean) < _MIN_POINTS:
        logger.warning(
            "bai_perron.detect_breaks_pelt: cleaned series has {} points "
            "(need >= {}); returning no breaks",
            len(clean),
            _MIN_POINTS,
        )
        return []

    signal = clean.to_numpy(dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(signal)
    raw_breakpoints = algo.predict(pen=penalty)
    return _indices_to_dates(raw_breakpoints, clean)


def detect_breaks_binseg(
    series: pd.Series,
    n_breaks: int,
    model: str = "rbf",
) -> list[pd.Timestamp]:
    """Detect ``n_breaks`` breakpoint dates via Binary Segmentation.

    Args:
        series: Univariate time series with a ``DatetimeIndex``.
        n_breaks: Number of breakpoints to return. Must be non-negative.
        model: Cost model passed to ``ruptures.Binseg``. Defaults to
            ``"rbf"``.

    Returns:
        Sorted list of exactly ``n_breaks`` break dates (or empty list
        if ``n_breaks`` is zero or the series is too short).
    """
    if n_breaks < 0:
        raise ValueError(f"n_breaks must be non-negative; got {n_breaks}")

    clean = _prepare_signal(series)
    if len(clean) < _MIN_POINTS or n_breaks == 0:
        if len(clean) < _MIN_POINTS:
            logger.warning(
                "bai_perron.detect_breaks_binseg: cleaned series has {} points "
                "(need >= {}); returning no breaks",
                len(clean),
                _MIN_POINTS,
            )
        return []

    signal = clean.to_numpy(dtype=float).reshape(-1, 1)
    algo = rpt.Binseg(model=model).fit(signal)
    raw_breakpoints = algo.predict(n_bkps=n_breaks)
    return _indices_to_dates(raw_breakpoints, clean)


def tune_penalty(
    series: pd.Series,
    penalties: list[float] | None = None,
    model: str = "rbf",
) -> pd.DataFrame:
    """Run PELT at multiple penalty values; return a sensitivity table.

    Args:
        series: Univariate time series with a ``DatetimeIndex``.
        penalties: Penalty values to try. Defaults to
            ``[5, 10, 15, 20, 30]``. CP4 may widen this grid on the real
            series if every value yields the same number of breaks.
        model: Cost model passed to ``ruptures.Pelt``.

    Returns:
        DataFrame with one row per penalty and the columns:

        ``penalty`` (float): the penalty value used.
        ``n_breaks`` (int): number of breaks PELT reported.
        ``breakpoints`` (str): comma-separated ISO-formatted dates;
          empty string when no breaks were detected.
    """
    if penalties is None:
        penalties = list(_DEFAULT_PENALTIES)

    rows = []
    for penalty in penalties:
        breaks = detect_breaks_pelt(series, penalty=float(penalty), model=model)
        rows.append(
            {
                "penalty": float(penalty),
                "n_breaks": len(breaks),
                "breakpoints": ", ".join(b.strftime("%Y-%m-%d") for b in breaks),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "detect_breaks_binseg",
    "detect_breaks_pelt",
    "tune_penalty",
]
