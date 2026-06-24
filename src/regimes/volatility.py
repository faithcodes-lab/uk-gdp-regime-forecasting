"""Inclan-Tiao (1994) ICSS variance change-point test.

Detects multiple points at which the variance of a univariate series
shifts. The algorithm is CUSUM-of-squares applied recursively: locate
the variance break with the largest D statistic, split, recurse on each
sub-sample, then run a final iterative refinement pass that re-tests
each detected break against the segment bounded by its two neighbours.

ICSS is the standard multi-break variance change-point test in
econometrics and finance (Inclan and Tiao 1994, JASA). It complements
the mean-based Chow test and the distributional rbf-PELT (Bai-Perron)
sweep used elsewhere in this project by targeting the second moment
specifically.

Conventions match :mod:`src.regimes.bai_perron`: the input series must
have a ``DatetimeIndex``; rows with NaN are dropped with a loguru
warning; a break at index ``i`` is mapped to ``series.index[i]`` (the
first observation of the new variance regime).

Asymptotic critical values are from Inclan and Tiao 1994 Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

_CRITICAL_VALUES: dict[float, float] = {
    0.01: 1.628,
    0.05: 1.358,
    0.10: 1.224,
}

# ICSS is asymptotic; below this length the statistic gives noise rather
# than signal.
_MIN_POINTS = 20

# Each recursive sub-segment needs a few points either side of any
# candidate break for the variance estimate to mean anything.
_MIN_SEGMENT_POINTS = 10

_METHOD_NAME = "ICSS (Inclan-Tiao 1994)"


@dataclass(frozen=True)
class ICSSResult:
    """Result of an ICSS variance change-point test on one series."""

    method: str
    series_name: str
    n_observations: int
    significance_level: float
    critical_value: float
    breakpoint_dates: list[pd.Timestamp]
    d_statistics: list[float]
    n_breaks: int


def _max_d_statistic(centered: np.ndarray) -> tuple[float, int]:
    """Return the ICSS test statistic and the index that maximises it.

    For the centred series ``y_t``:

    * ``C_k = sum_{t=0..k-1} y_t^2``
    * ``D_k = C_k / C_T - k / T``  for ``k = 1, ..., T-1``
    * test statistic ``= sqrt(T / 2) * max_k |D_k|``

    The returned index is in 0-based coordinates of the input array and
    points to the first observation of the new variance regime.
    """
    n_points = len(centered)
    if n_points < 2:
        return 0.0, 0
    squared = centered**2
    cum = np.cumsum(squared)
    total = cum[-1]
    if total == 0.0:
        # Degenerate (all zeros after centring); no variance to break.
        return 0.0, 0
    ks = np.arange(1, n_points)
    d_values = cum[:-1] / total - ks / n_points
    abs_d = np.abs(d_values)
    local_idx = int(np.argmax(abs_d))
    statistic = float(np.sqrt(n_points / 2.0) * abs_d[local_idx])
    break_index = int(ks[local_idx])
    return statistic, break_index


def _recurse(
    centered: np.ndarray,
    start: int,
    end: int,
    critical_value: float,
    found: list[tuple[int, float]],
    max_breaks: int,
) -> None:
    """Recursive ICSS: append ``(absolute_index, statistic)`` to ``found``."""
    if len(found) >= max_breaks:
        return
    if end - start < _MIN_SEGMENT_POINTS:
        return
    segment = centered[start:end]
    statistic, local_idx = _max_d_statistic(segment)
    if statistic <= critical_value:
        return
    absolute_idx = start + local_idx
    found.append((absolute_idx, statistic))
    _recurse(centered, start, absolute_idx, critical_value, found, max_breaks)
    _recurse(centered, absolute_idx, end, critical_value, found, max_breaks)


def _refine_breaks(
    centered: np.ndarray,
    candidates: list[int],
    critical_value: float,
) -> list[tuple[int, float]]:
    """Iterative refinement: re-test each break against its neighbours.

    A break only survives if, in the segment bounded by the immediately
    preceding and following breaks (or the sample endpoints), the test
    statistic at that break index still exceeds the critical value. The
    refinement repeats until the surviving set is stable.
    """
    current = sorted(set(candidates))
    n_points = len(centered)
    while True:
        boundaries = [0] + current + [n_points]
        survivors: list[tuple[int, float]] = []
        for i, bp in enumerate(current):
            left = boundaries[i]
            right = boundaries[i + 2]
            segment = centered[left:right]
            local_bp = bp - left
            if len(segment) < 2 or local_bp <= 0 or local_bp >= len(segment):
                continue
            cum = np.cumsum(segment**2)
            total = cum[-1]
            if total == 0.0:
                continue
            d_at = cum[local_bp - 1] / total - local_bp / len(segment)
            statistic = float(np.sqrt(len(segment) / 2.0) * abs(d_at))
            if statistic > critical_value:
                survivors.append((bp, statistic))
        new_breaks = [s[0] for s in survivors]
        if new_breaks == current:
            return survivors
        current = new_breaks


def icss_test(
    series: pd.Series,
    *,
    alpha: float = 0.05,
    max_breakpoints: int = 20,
) -> ICSSResult:
    """Run the Inclan-Tiao ICSS variance change-point test.

    Args:
        series: Univariate time series with a ``DatetimeIndex``. The
            mean of the series is removed internally before the
            CUSUM-of-squares is computed; do not centre the input
            yourself.
        alpha: Significance level. Must be one of ``0.01``, ``0.05``,
            ``0.10`` (the levels for which Inclan and Tiao 1994 Table 1
            provides asymptotic critical values).
        max_breakpoints: Safety bound on the recursion depth. Default 20.

    Returns:
        An :class:`ICSSResult` with the detected break dates sorted
        ascending, their D statistics, and the metadata needed for
        downstream reporting.

    Raises:
        ValueError: If ``series.index`` is not a ``DatetimeIndex``, or
            ``alpha`` is not one of the supported critical-value levels.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex; got " f"{type(series.index).__name__}")
    if alpha not in _CRITICAL_VALUES:
        raise ValueError(f"alpha must be one of {sorted(_CRITICAL_VALUES)}; got {alpha}")

    critical_value = _CRITICAL_VALUES[alpha]

    n_original = len(series)
    clean = series.dropna()
    n_dropped = n_original - len(clean)
    if n_dropped > 0:
        logger.warning(
            "icss_test: dropped {} NaN values from a series of {}",
            n_dropped,
            n_original,
        )

    series_name = str(series.name) if series.name is not None else ""

    if len(clean) < _MIN_POINTS:
        logger.warning(
            "icss_test: cleaned series has {} points (need >= {}); " "returning no breaks",
            len(clean),
            _MIN_POINTS,
        )
        return ICSSResult(
            method=_METHOD_NAME,
            series_name=series_name,
            n_observations=int(len(clean)),
            significance_level=alpha,
            critical_value=critical_value,
            breakpoint_dates=[],
            d_statistics=[],
            n_breaks=0,
        )

    centered = (clean - clean.mean()).to_numpy(dtype=float)

    found: list[tuple[int, float]] = []
    _recurse(centered, 0, len(centered), critical_value, found, max_breakpoints)
    refined = _refine_breaks(centered, [b[0] for b in found], critical_value)
    refined.sort(key=lambda x: x[0])

    breakpoint_dates = [pd.Timestamp(clean.index[bp]) for bp, _ in refined]
    d_statistics = [stat for _, stat in refined]

    return ICSSResult(
        method=_METHOD_NAME,
        series_name=series_name,
        n_observations=int(len(clean)),
        significance_level=alpha,
        critical_value=critical_value,
        breakpoint_dates=breakpoint_dates,
        d_statistics=d_statistics,
        n_breaks=len(breakpoint_dates),
    )


__all__ = ["ICSSResult", "icss_test"]
