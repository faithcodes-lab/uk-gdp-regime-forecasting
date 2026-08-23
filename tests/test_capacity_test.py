"""Tests for scripts/capacity_test.py.

Only the active-feature-counting logic is unit tested here since it is
fast and deterministic; the full CV run is covered by the reproduction
check in the script's own main() (it logs each variant's active feature
count and RMSE against the recorded values from dissertation Table 4.7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.capacity_test import _VARIANTS, _active_feature_count


def test_variants_cumulative_not_isolated():
    """Each later variant carries forward the previous step's changes rather than resetting to baseline."""
    variants = list(_VARIANTS.values())
    assert variants[1]["n_estimators"] == 500
    assert variants[2]["n_estimators"] == 500  # carried forward, not reset to 50
    assert variants[2]["learning_rate"] == 0.1
    assert variants[3]["n_estimators"] == 500  # carried forward
    assert variants[3]["learning_rate"] == 0.1  # carried forward
    assert variants[3]["max_depth"] == 6


def test_active_feature_count_ignores_unused_feature():
    """A feature with no predictive relationship to y is not counted as active."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "informative": rng.normal(size=60),
            "pure_noise": rng.normal(size=60),
        }
    )
    y = pd.Series(3.0 * X["informative"] + rng.normal(scale=0.05, size=60))
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}
    n_active = _active_feature_count(X, y, params)
    assert n_active == 1


def test_active_feature_count_is_at_most_n_columns():
    """The active feature count never exceeds the number of columns in X."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(size=60))
    params = {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}
    n_active = _active_feature_count(X, y, params)
    assert 0 <= n_active <= 5
