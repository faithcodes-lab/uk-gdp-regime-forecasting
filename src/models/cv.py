"""Splitters that cut the data into time-aware folds for cross-validation.

Two ways to split are provided. Both guarantee the test set sits
entirely after the training set, so the model never sees the future
during training.

- ``expanding_window_splits``: each fold trains on more quarters than
  the last and tests on the next few quarters. Primary scheme.

- ``regime_aligned_splits``: each fold trains on the regimes seen so
  far and tests on the regimes that come after, showing how a model
  handles a regime it has not learned from. Secondary scheme.

Both return row indices only and never modify the input data.

Three things the caller must still get right:

1. Fit any data-learning step (scaler, encoder, imputer) inside each
   fold on the training portion only. Wrap it in a sklearn Pipeline so
   it fits per fold automatically. Otherwise scaling parameters leak
   the test-set statistics into training. (Rule 1.)

2. Row ``i`` in X must already represent features at quarter t that
   predict y at t+1. The splitters do not shift X or y; that is the
   caller's job before calling them. (Rule 5.)

3. ``regime_aligned_splits`` reads ``X[regime_column]`` for grouping
   but never returns it as a feature. Drop the regime column from the
   feature matrix before fitting any model. (Rule 6.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRAIN_SIZE = 20
"""The smallest training set the first fold is allowed to have.

Twenty quarters is five years. Less than that and the first model has
too little history to learn anything stable. ``expanding_window_splits``
raises if X is too short to satisfy this floor.
"""


def expanding_window_splits(
    X: pd.DataFrame,
    n_splits: int = 8,
    test_size: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splits the data into training and test sets that grow over time, so each test set is always the quarters immediately after what the model trained on.

    Test size is the same in every fold; the training set grows by
    ``test_size`` each fold. The first fold trains on the smallest
    history that still satisfies ``MIN_TRAIN_SIZE``, and the last
    fold's test window ends exactly at the end of the data.

    Parameters
    X : pd.DataFrame
        Feature matrix. Must be sorted by date. Only ``len(X)`` is used.
    n_splits : int, default 8
        Number of folds.
    test_size : int, default 4
        Number of rows in each fold's test set. 4 quarters is one year.

    Returns:
    list of (train_idx, test_idx) tuples
        Each tuple is two numpy int arrays of row positions in X.

    Raises
    ValueError
        If ``len(X)`` is too small to give the first fold at least
        ``MIN_TRAIN_SIZE`` training rows.
    """
    n = len(X)
    # The smallest dataset we accept is one where the first fold still
    # has MIN_TRAIN_SIZE training rows after reserving n_splits *
    # test_size rows for all the test sets together.
    min_required = n_splits * test_size + MIN_TRAIN_SIZE
    if n < min_required:
        raise ValueError(
            f"len(X)={n} is too small for n_splits={n_splits}, "
            f"test_size={test_size}; need at least {min_required} rows "
            f"(MIN_TRAIN_SIZE={MIN_TRAIN_SIZE})."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    # The first fold trains on rows 0..first_train_end. After that the
    # training set grows by test_size each fold and the test window
    # slides forward by the same amount, so the last fold's test window ends exactly at row n.
    first_train_end = n - n_splits * test_size
    for k in range(n_splits):
        train_end = first_train_end + k * test_size
        test_end = train_end + test_size
        train_idx = np.arange(0, train_end, dtype=int)
        test_idx = np.arange(train_end, test_end, dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def regime_aligned_splits(
    X: pd.DataFrame,
    regime_column: str = "regime",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Splits the data so each fold trains on the regimes seen so far and tests on the regimes that come after.

    With ``r`` distinct regimes this gives ``r - 1`` folds. Fold 1
    trains on the first regime alone and tests on every regime that
    follows; fold 2 trains on the first two regimes and tests on the
    rest; and so on. The point is to show how a model performs on a
    regime it has never seen.

    Parameters:
    X : pd.DataFrame
        Feature matrix containing the regime column. Must be sorted by
        date so the order of regimes matches the order they happened.
    regime_column : str, default 'regime'
        Name of the column with the regime label for each row.

    Returns:
    list of (train_idx, test_idx) tuples
        Each tuple is two numpy int arrays of row positions in X.

    Raises:
    ValueError
        If ``regime_column`` is missing from X, or if fewer than two
        distinct regimes are present (no fold to make).
    """
    if regime_column not in X.columns:
        raise ValueError(
            f"regime_column={regime_column!r} not found in X.columns " f"({list(X.columns)})"
        )

    # Build the regime order from the first time each label appears in
    # X. This works because X is sorted by date, so first-appearance
    # order is the same as time order.
    regimes_in_order: list = []
    for label in X[regime_column]:
        if label not in regimes_in_order:
            regimes_in_order.append(label)

    if len(regimes_in_order) < 2:
        raise ValueError(
            f"regime_aligned_splits requires at least 2 distinct regimes; "
            f"found {len(regimes_in_order)} in X[{regime_column!r}]."
        )

    regime_values = X[regime_column].to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Start at k = 1 because fold 1 trains on the first regime only; k
    # = 0 would mean training on no regimes. Stop before
    # len(regimes_in_order) so the last fold has at least one regime
    # left to test on.
    for k in range(1, len(regimes_in_order)):
        train_regimes_list = regimes_in_order[:k]
        test_regimes_list = regimes_in_order[k:]
        train_idx = np.where(np.isin(regime_values, train_regimes_list))[0]
        test_idx = np.where(np.isin(regime_values, test_regimes_list))[0]
        splits.append((train_idx, test_idx))
    return splits
