"""Tests for scripts/rolling_window_check.py.

Only the fold-comparability property is unit tested here since it is
fast and deterministic; the full CV run is covered by the script's own
main(), which reports the actual RMSE gap between schemes rather than
assuming a match.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.cv import expanding_window_splits, rolling_window_splits


def _dummy_X(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f1": np.arange(n, dtype=float)})


def test_rolling_and_expanding_share_test_folds():
    """The comparison in rolling_window_check.py is only valid if both schemes test on the same rows."""
    X = _dummy_X(104)
    expanding = expanding_window_splits(X, n_splits=8, test_size=4)
    rolling = rolling_window_splits(X, n_splits=8, test_size=4)
    for (_, exp_test), (_, roll_test) in zip(expanding, rolling):
        np.testing.assert_array_equal(exp_test, roll_test)


def test_rolling_window_default_size_matches_first_expanding_fold():
    """The default rolling window_size equals expanding_window_splits' first-fold training size."""
    X = _dummy_X(104)
    expanding = expanding_window_splits(X, n_splits=8, test_size=4)
    rolling = rolling_window_splits(X, n_splits=8, test_size=4)
    assert len(rolling[0][0]) == len(expanding[0][0])
