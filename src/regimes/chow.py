"""Chow test for a single known structural breakpoint.

Implements the Chow (1960) F-test, which checks whether the regression of
``y`` on ``X`` differs between the pre- and post-breakpoint sub-samples.
The test fits three OLS regressions (pooled, pre, post) and computes

    F = ((RSS_pooled - (RSS_pre + RSS_post)) / k_params) /
        ((RSS_pre + RSS_post) / (n_total - 2 * k_params))

where ``k_params = features + 1`` because an intercept is always added via
``sm.add_constant``.


Both the input ``dates`` and ``breakpoint_date`` are normalised to the
quarter-start of the quarter they fall in before comparison. The project
dataset stores quarter-end dates (e.g. 2008-03-31 for Q1 2008) and the
regime boundaries in ``config/regimes.yaml`` are quarter-start (e.g.
2008-01-01 for the GFC onset). After normalisation, the comparison
``dates_q_start < breakpoint_q_start`` is unambiguous and routes the
boundary quarter (Q1 2008 in the example) to the post-period, as
intended.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats


class InsufficientObservationsError(ValueError):
    """Raised when a sub-sample is too small to fit the regression.

    Each sub-sample needs more rows than parameters so the residual
    variance has positive degrees of freedom. The caller should catch this and record the skip rather
    than dropping it silently.
    """


@dataclass(frozen=True)
class ChowTestResult:
    """Result of a single Chow test at a known breakpoint."""

    breakpoint_date: pd.Timestamp
    f_statistic: float
    p_value: float
    rss_pooled: float
    rss_pre: float
    rss_post: float
    n_total: int
    n_pre: int
    n_post: int
    k_params: int
    df_numerator: int
    df_denominator: int


def chow_test(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    breakpoint_date: str | pd.Timestamp,
) -> ChowTestResult:
    """Run a Chow test for a single known breakpoint at ``breakpoint_date``.

    Args:
        X: Predictor matrix, ``n`` rows by ``k`` features. An intercept
            is added internally; do not pre-pend a constant column.
        y: Target series, length ``n``.
        dates: Date series, length ``n``. Drives the pre/post split.
            Any datetime basis (quarter-start, quarter-end, month-end,
            etc.) is accepted; dates are normalised to quarter-start
            internally so the boundary quarter routes to post.
        breakpoint_date: Candidate break date. Strings are parsed via
            ``pd.Timestamp``. The quarter containing this date is the
            first quarter of the post-period.

    Returns:
        A ``ChowTestResult`` with the F statistic, p-value, RSS values,
        and sample sizes.

    Raises:
        InsufficientObservationsError: If either sub-sample has at most
            ``k_params`` observations, leaving no positive residual
            degrees of freedom for the variance estimate.
    """
    dates = pd.to_datetime(dates)
    breakpoint = pd.Timestamp(breakpoint_date)

    # Drop any row where X, y, or dates carries a NaN, before splitting.
    combined = pd.concat(
        [X, y.rename("__y__"), dates.rename("__date__")],
        axis=1,
    )
    n_original = len(combined)
    combined = combined.dropna()
    n_dropped = n_original - len(combined)
    if n_dropped > 0:
        logger.warning(
            "chow_test: dropped {} rows containing NaN from a sample of {}",
            n_dropped,
            n_original,
        )

    feature_cols = [c for c in combined.columns if c not in ("__y__", "__date__")]
    X_clean = combined[feature_cols]
    y_clean = combined["__y__"].rename(y.name)
    dates_clean = combined["__date__"]

    # Normalise both sides of the comparison to quarter-start so the
    # boundary quarter (a quarter-end date in the dataset, e.g. 2008-03-31)
    # lands in the post-period when its quarter-start (2008-01-01) is not
    # strictly less than the breakpoint's quarter-start.
    sample_q_start = dates_clean.dt.to_period("Q").dt.to_timestamp()
    break_q_start = breakpoint.to_period("Q").to_timestamp()

    pre_mask = (sample_q_start < break_q_start).to_numpy()
    post_mask = ~pre_mask

    # Add an intercept column. k_params then counts features + intercept.
    X_with_const = sm.add_constant(X_clean, has_constant="add")
    n_total = len(X_with_const)
    k_params = X_with_const.shape[1]
    n_pre = int(pre_mask.sum())
    n_post = int(post_mask.sum())

    if n_pre <= k_params or n_post <= k_params:
        raise InsufficientObservationsError(
            f"Chow test at {breakpoint.date()} needs more than {k_params} "
            f"observations per sub-sample; got n_pre={n_pre}, n_post={n_post}."
        )

    rss_pooled = float(sm.OLS(y_clean, X_with_const).fit().ssr)
    rss_pre = float(sm.OLS(y_clean[pre_mask], X_with_const[pre_mask]).fit().ssr)
    rss_post = float(sm.OLS(y_clean[post_mask], X_with_const[post_mask]).fit().ssr)

    df_num = k_params
    df_den = n_total - 2 * k_params
    f_statistic = ((rss_pooled - (rss_pre + rss_post)) / df_num) / ((rss_pre + rss_post) / df_den)
    p_value = float(stats.f.sf(f_statistic, df_num, df_den))

    return ChowTestResult(
        breakpoint_date=breakpoint,
        f_statistic=float(f_statistic),
        p_value=p_value,
        rss_pooled=rss_pooled,
        rss_pre=rss_pre,
        rss_post=rss_post,
        n_total=n_total,
        n_pre=n_pre,
        n_post=n_post,
        k_params=k_params,
        df_numerator=df_num,
        df_denominator=df_den,
    )
