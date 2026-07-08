"""Diebold-Mariano test for comparing forecast accuracy of two models.

Implements the Diebold and Mariano 1995 statistic with the Harvey, Leybourne
and Newbold 1997 small-sample correction; refers the corrected statistic to a
t-distribution with n-1 degrees of freedom. Reports both the raw and the
Bonferroni-corrected p-value so a family of pairwise comparisons can carry
the correction visibly. DM is deterministic given fixed inputs, so no random
state is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DMTestResult:
    """Result of a Diebold-Mariano test with HLN small-sample correction."""

    statistic: float
    p_value: float
    p_value_bonferroni: float
    n_observations: int
    n_comparisons: int
    loss_power: int
    horizon: int
    raw_dm_statistic: float


def diebold_mariano_test(
    errors_a,
    errors_b,
    h: int = 1,
    power: int = 2,
    n_comparisons: int = 1,
) -> DMTestResult:
    """Runs the Diebold-Mariano test comparing the forecast accuracy of two models.

    Applies the HLN small-sample correction and refers the corrected statistic
    to a t-distribution with n-1 degrees of freedom. With n_comparisons > 1 the
    returned dataclass carries both the raw p-value and the Bonferroni-corrected
    p-value for the named family size.
    """
    if power not in (1, 2):
        raise ValueError(f"power must be 1 or 2, got {power}")

    errors_a_arr = np.asarray(errors_a, dtype=float)
    errors_b_arr = np.asarray(errors_b, dtype=float)
    _validate_pair(errors_a_arr, errors_b_arr)

    # identical forecasts: no difference to test, p = 1.0 by definition
    if np.array_equal(errors_a_arr, errors_b_arr):
        return DMTestResult(
            statistic=0.0,
            p_value=1.0,
            p_value_bonferroni=1.0,
            n_observations=int(errors_a_arr.size),
            n_comparisons=n_comparisons,
            loss_power=power,
            horizon=h,
            raw_dm_statistic=0.0,
        )

    if power == 1:
        d = np.abs(errors_a_arr) - np.abs(errors_b_arr)
    else:
        d = errors_a_arr**2 - errors_b_arr**2

    n = int(d.size)
    d_bar = float(np.mean(d))

    # Newey-West long-run variance with truncation at lag h-1; for h=1 the loop
    # is empty and the variance is just the sample variance of d
    gamma_0 = float(np.mean((d - d_bar) ** 2))
    long_run_var = gamma_0
    for lag in range(1, h):
        gamma_k = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        long_run_var += 2 * gamma_k

    if long_run_var <= 0:
        raise ValueError(
            "non-positive long-run variance; cannot compute DM statistic "
            "(loss differentials may be constant non-zero)"
        )

    raw_dm_stat = d_bar / float(np.sqrt(long_run_var / n))

    # HLN factor; for h=1 simplifies to sqrt((n-1)/n)
    hln_factor = float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    hln_stat = raw_dm_stat * hln_factor

    p_value = float(2 * (1 - stats.t.cdf(abs(hln_stat), df=n - 1)))
    p_value_bonferroni = float(min(p_value * n_comparisons, 1.0))

    return DMTestResult(
        statistic=hln_stat,
        p_value=p_value,
        p_value_bonferroni=p_value_bonferroni,
        n_observations=n,
        n_comparisons=n_comparisons,
        loss_power=power,
        horizon=h,
        raw_dm_statistic=raw_dm_stat,
    )


def _validate_pair(errors_a: np.ndarray, errors_b: np.ndarray) -> None:
    """Raises ValueError if the arrays are empty, mismatched in length, or contain NaN."""
    if errors_a.size == 0 or errors_b.size == 0:
        raise ValueError("errors_a and errors_b must be non-empty")
    if errors_a.size != errors_b.size:
        raise ValueError(
            f"length mismatch: len(errors_a)={errors_a.size}, len(errors_b)={errors_b.size}"
        )
    if np.isnan(errors_a).any() or np.isnan(errors_b).any():
        raise ValueError("errors_a and errors_b must contain no NaN values")
