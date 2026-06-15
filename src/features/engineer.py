"""Apply engineered features from ``config/features.yaml``.

Routes each entry by transformation: lag (shift), rolling_mean (right-closed
window), year_over_year (4-quarter compounded growth), subtract (one column
minus another, for the yield-curve slope). Columns under
intermediate_only_then_dropped (the two gilt yields) feed transformations,
then are removed.

Look-ahead discipline: only shift, right-closed rolling, and
contemporaneous arithmetic: no future row leaks into a feature at time ``t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.config import features_config


def engineer_features(
    df: pd.DataFrame,
    *,
    features_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply all engineered features and drop intermediate-only columns.

    Args:
        df: Merged quarterly frame with raw source columns as columns.
        features_cfg: Override for testing. Defaults to :func:`features_config`.

    Returns:
        A new frame with engineered features added and intermediate-only
        columns removed.

    Raises:
        ValueError: If an entry's ``transformation`` is unknown.
    """
    cfg = features_cfg if features_cfg is not None else features_config()
    out = df.copy()

    for entry in cfg["engineered"]:
        name = entry["name"]
        transformation = entry["transformation"]

        if transformation == "lag":
            out[name] = _lag(out[entry["source"]], periods=int(entry["periods"]))
        elif transformation == "rolling_mean":
            out[name] = _rolling_mean(out[entry["source"]], window=int(entry["window"]))
        elif transformation == "year_over_year":
            out[name] = _cumulative_compounded_growth(
                out[entry["source"]], periods=int(entry["periods"])
            )
        elif transformation == "subtract":
            sources = entry["sources"]
            out[name] = out[sources[0]] - out[sources[1]]
        else:
            raise ValueError(f"Unknown engineered transformation: {transformation!r}")

    intermediate = cfg.get("intermediate_only_then_dropped", [])
    cols_to_drop = [c for c in intermediate if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    return out


def _lag(series: pd.Series, *, periods: int) -> pd.Series:
    """Lag a quarterly series by ``periods`` quarters. Strictly past values."""
    return series.shift(periods)


def _rolling_mean(series: pd.Series, *, window: int) -> pd.Series:
    """Right-closed rolling mean.

    The window at time ``t`` covers ``[t-window+1, t]`` — past and present
    only. The first ``window-1`` observations are NaN (insufficient history).
    """
    return series.rolling(window=window, min_periods=window).mean()


def _cumulative_compounded_growth(growth_pct: pd.Series, *, periods: int) -> pd.Series:
    """Compounded growth over the trailing ``periods`` quarters, in percent.

    Input is a QoQ growth rate in percent (e.g. -19.4 for -19.4%). Compounds
    (1 + g) over a right-closed window [t-periods+1, t], past and present
    only, so periods=4 gives year-over-year growth. The first
    ``periods - 1`` rows are NaN (insufficient history).
    """
    g_dec = growth_pct / 100.0
    one_plus = 1.0 + g_dec
    rolling_prod = one_plus.rolling(window=periods, min_periods=periods).apply(np.prod, raw=True)
    return (rolling_prod - 1.0) * 100.0
