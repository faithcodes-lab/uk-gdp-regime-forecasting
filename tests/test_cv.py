"""Tests for ``src/models/cv.py`` and ``src/models/visualise_cv.py``.

The three named leakage tests prove the splitters never let a fold see future data. The rest check
structural and guard behaviour.
"""

from __future__ import annotations
from src.models.visualise_cv import plot_cv_splits
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

import matplotlib

matplotlib.use("Agg")  # headless backend for tests


from src.models.cv import expanding_window_splits, regime_aligned_splits  # noqa: E402

# helpers


def _dummy_X(n: int) -> pd.DataFrame:
    """Builds a synthetic feature DataFrame with n rows and two numeric columns."""
    return pd.DataFrame(
        {
            "f1": np.arange(n, dtype=float),
            "f2": np.arange(n, dtype=float) ** 2,
        }
    )


def _project_regime_X() -> pd.DataFrame:
    """Builds a 104-row DataFrame mirroring the project's six-regime structure (counts 33, 6, 27, 14, 6, 18)."""
    counts = [33, 6, 27, 14, 6, 18]
    labels = [
        "Pre-GFC Stability",
        "Global Financial Crisis",
        "Post-GFC Recovery",
        "Brexit Transition",
        "COVID-19 Shock",
        "Post-COVID Recovery",
    ]
    regime_col: list[str] = []
    for label, count in zip(labels, counts):
        regime_col.extend([label] * count)
    n = sum(counts)
    return pd.DataFrame(
        {
            "f1": np.arange(n, dtype=float),
            "regime": regime_col,
        }
    )


#  the leakage trio


def test_no_train_test_overlap():
    """No row index appears in both train and test for any fold of either scheme."""
    X = _project_regime_X()

    for train_idx, test_idx in expanding_window_splits(X, n_splits=8, test_size=4):
        assert set(train_idx).isdisjoint(set(test_idx))

    for train_idx, test_idx in regime_aligned_splits(X):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_train_temporally_before_test():
    """Every fold's max train index is strictly less than its min test index, in either scheme.

    This is the no future data in training check: any leakage of a
    later row into the training set would push max(train_idx) up to or
    past min(test_idx) and fail this assertion.
    """
    X = _project_regime_X()

    # asserted with a message so a regression points straight to the offending fold
    for fold_idx, (train_idx, test_idx) in enumerate(
        expanding_window_splits(X, n_splits=8, test_size=4)
    ):
        assert train_idx.max() < test_idx.min(), (
            f"expanding window fold {fold_idx + 1}: max train idx "
            f"{train_idx.max()} is not strictly less than min test idx {test_idx.min()}"
        )

    for fold_idx, (train_idx, test_idx) in enumerate(regime_aligned_splits(X)):
        assert train_idx.max() < test_idx.min(), (
            f"regime aligned fold {fold_idx + 1}: max train idx "
            f"{train_idx.max()} is not strictly less than min test idx {test_idx.min()}"
        )


def test_regime_aligned_train_contains_only_assigned_regimes():
    """For fold k the regimes in train_idx are exactly the first k by first-appearance order, and the test regimes are exactly the rest."""
    X = _project_regime_X()
    # dict.fromkeys preserves first-appearance order, matching what
    # regime_aligned_splits does internally.
    regimes_in_order = list(dict.fromkeys(X["regime"]))

    splits = regime_aligned_splits(X)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        k = fold_idx + 1
        expected_train = set(regimes_in_order[:k])
        expected_test = set(regimes_in_order[k:])
        actual_train = set(X["regime"].iloc[train_idx])
        actual_test = set(X["regime"].iloc[test_idx])
        assert (
            actual_train == expected_train
        ), f"fold {k}: train regimes {actual_train} != expected {expected_train}"
        assert (
            actual_test == expected_test
        ), f"fold {k}: test regimes {actual_test} != expected {expected_test}"


# expanding window structural tests


def test_expanding_window_yields_n_splits_folds():
    """n_splits=8 produces exactly 8 folds."""
    X = _dummy_X(104)
    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    assert len(splits) == 8


def test_expanding_window_test_size_constant():
    """Every fold's test set has exactly test_size rows."""
    X = _dummy_X(104)
    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    for _train_idx, test_idx in splits:
        assert len(test_idx) == 4


def test_expanding_window_expanding_train_size():
    """Training set grows by test_size each fold, monotonically."""
    X = _dummy_X(104)
    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    sizes = [len(train_idx) for train_idx, _ in splits]
    # consecutive differences should all equal test_size
    diffs = [sizes[i + 1] - sizes[i] for i in range(len(sizes) - 1)]
    assert all(
        d == 4 for d in diffs), f"train sizes not monotonically +4: {sizes}"


# regime aligned structural tests


def test_regime_aligned_yields_five_folds_for_six_regimes():
    """Six regimes produce five folds (r minus one)."""
    X = _project_regime_X()
    splits = regime_aligned_splits(X)
    assert len(splits) == 5


def test_regime_aligned_first_fold_trains_on_regime_1_only():
    """Fold 1 trains on the first regime alone."""
    X = _project_regime_X()
    splits = regime_aligned_splits(X)
    train_idx, _ = splits[0]
    train_regimes = set(X["regime"].iloc[train_idx])
    assert train_regimes == {"Pre-GFC Stability"}


def test_regime_aligned_last_fold_tests_on_regime_6_only():
    """The last fold tests on the sixth regime alone."""
    X = _project_regime_X()
    splits = regime_aligned_splits(X)
    _, test_idx = splits[-1]
    test_regimes = set(X["regime"].iloc[test_idx])
    assert test_regimes == {"Post-COVID Recovery"}


#  guard tests


def test_expanding_window_raises_when_sample_too_small():
    """Calling on too few rows raises a ValueError matching 'too small'."""
    # 30 rows fails because n_splits=8, test_size=4 requires 8 * 4 + 20 = 52 rows
    X = _dummy_X(30)
    with pytest.raises(ValueError, match="too small"):
        expanding_window_splits(X, n_splits=8, test_size=4)


def test_regime_aligned_raises_when_regime_column_missing():
    """Missing regime column raises a ValueError naming the column."""
    X = pd.DataFrame({"f1": [1, 2, 3]})
    with pytest.raises(ValueError, match="regime_column"):
        regime_aligned_splits(X)


def test_regime_aligned_raises_when_fewer_than_two_regimes():
    """A single regime cannot be split into folds and raises a ValueError."""
    X = pd.DataFrame({"regime": ["A"] * 10, "f1": range(10)})
    with pytest.raises(ValueError, match="at least 2"):
        regime_aligned_splits(X)


# visualise smoke test


def test_plot_cv_splits_returns_figure():
    """plot_cv_splits returns a matplotlib Figure object."""
    X = _project_regime_X()
    splits = expanding_window_splits(X, n_splits=8, test_size=4)
    fig = plot_cv_splits(splits, n_samples=len(X), title="Test")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
