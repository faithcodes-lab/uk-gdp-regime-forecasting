"""Assign each row of the final dataset to one of the six regimes.

The assignment is quarter-based: a row's date is converted to its
pandas ``Period[Q]`` and matched against the ``[start, end]`` quarter
range of each regime in ``config/regimes.yaml``. This avoids the
quarter-end vs quarter-start basis mismatch that appears in CP1 and
CP2 — the comparison operates on quarters, not timestamps, so the
boundary quarter of each regime lands in the new regime as expected.

The module performs two layers of validation:

* **Config-level**: the regimes must be contiguous over their union
  (no quarter belongs to two regimes; no quarter inside the union is
  missing from every regime).
* **Row-level**: every row of the input DataFrame must fall inside the
  regime union. A row outside the union raises rather than producing a
  silent NaN.
"""

from __future__ import annotations

import pandas as pd


def _resolve_regimes(
    regimes: list[dict] | dict | None,
) -> list[dict]:
    """Accept any of the supported forms and return a list of regime dicts."""
    if regimes is None:
        # Local import keeps src.regimes free of a hard dependency on src.data
        # in the common test path where the caller supplies the regime list.
        from src.data.config import regimes_config

        return list(regimes_config()["regimes"])
    if isinstance(regimes, dict):
        if "regimes" not in regimes:
            raise ValueError(
                "regimes dict must contain a 'regimes' key; got keys " f"{list(regimes.keys())}"
            )
        return list(regimes["regimes"])
    return list(regimes)


def _validate_regime_config(regimes_list: list[dict]) -> None:
    """Check that the regime list is contiguous: no overlaps, no gaps.

    Raises:
        ValueError: If the list is empty, any regime has end before
            start, two regimes overlap, or a quarter between two
            regimes is missing.
    """
    if not regimes_list:
        raise ValueError("regimes list is empty")

    sorted_regimes = sorted(
        regimes_list,
        key=lambda r: pd.Timestamp(r["start"]).to_period("Q"),
    )
    for i, r in enumerate(sorted_regimes):
        start_q = pd.Timestamp(r["start"]).to_period("Q")
        end_q = pd.Timestamp(r["end"]).to_period("Q")
        if end_q < start_q:
            raise ValueError(f"Regime '{r['label']}' has end ({end_q}) before start ({start_q})")
        if i == 0:
            continue
        prev = sorted_regimes[i - 1]
        prev_end_q = pd.Timestamp(prev["end"]).to_period("Q")
        if start_q <= prev_end_q:
            raise ValueError(
                f"Regimes '{prev['label']}' and '{r['label']}' overlap: "
                f"previous ends {prev_end_q}, next starts {start_q}"
            )
        if start_q != prev_end_q + 1:
            raise ValueError(
                f"Gap between '{prev['label']}' (ends {prev_end_q}) and "
                f"'{r['label']}' (starts {start_q})"
            )


def assign_regimes(
    df: pd.DataFrame,
    regimes: list[dict] | dict | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``regime`` label column inserted after ``date``.

    Args:
        df: Input DataFrame; must contain a ``date`` column. The input
            is never mutated.
        regimes: Regime specification. Accepted forms:

            * ``None`` (default): load via ``src.data.config.regimes_config()``.
            * ``dict`` with a ``"regimes"`` key (the full parsed config).
            * ``list[dict]``: the bare list of regime dicts.

            Each regime dict must expose ``label``, ``start``, and
            ``end`` entries (start/end may be ``date``, ``datetime``,
            or ``pd.Timestamp``).

    Returns:
        A new DataFrame with the ``regime`` column inserted directly
        after ``date``. Values are the string ``label`` from the regime
        config.

    Raises:
        ValueError: If ``df`` lacks a ``date`` column, the regime
            configuration has gaps or overlaps, or any row falls
            outside the regime union.
    """
    if "date" not in df.columns:
        raise ValueError("df must contain a 'date' column; got columns " f"{list(df.columns)}")

    regimes_list = _resolve_regimes(regimes)
    _validate_regime_config(regimes_list)

    # Build a per-quarter lookup so the row assignment is a single map().
    quarter_to_label: dict[pd.Period, str] = {}
    for r in regimes_list:
        start_q = pd.Timestamp(r["start"]).to_period("Q")
        end_q = pd.Timestamp(r["end"]).to_period("Q")
        for q in pd.period_range(start_q, end_q, freq="Q"):
            quarter_to_label[q] = r["label"]

    row_quarters = pd.to_datetime(df["date"]).dt.to_period("Q")
    labels = row_quarters.map(quarter_to_label)

    if labels.isna().any():
        unassigned_dates = df.loc[labels.isna(), "date"].tolist()
        raise ValueError(
            f"{len(unassigned_dates)} row(s) fall outside the regime union; "
            f"first unassigned date: {unassigned_dates[0]}"
        )

    result = df.copy()
    date_position = result.columns.get_loc("date")
    result.insert(loc=date_position + 1, column="regime", value=labels.to_numpy())
    return result


__all__ = ["assign_regimes"]
