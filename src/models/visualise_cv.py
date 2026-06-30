"""Draws Gantt-style charts of the cross-validation folds.

Used by the methodology figure: one chart per CV scheme showing which
rows are train and which are test in each fold. Returns the
figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_cv_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_samples: int,
    title: str = "Cross-validation splits",
) -> Figure:
    """Draws a chart showing which rows are used for training and which for testing in each fold.

    One horizontal bar per fold: the training span in one colour, the
    test span in another, plotted against the row index along the x
    axis. Fold 1 appears at the top.

    Assumes the train and test index arrays for each fold are
    contiguous ranges, which is true for both splitters in this module.

    Parameters:
    splits : list of (train_idx, test_idx) tuples
        Output of ``expanding_window_splits`` or ``regime_aligned_splits``.
    n_samples : int
        Total number of rows in the dataset, used to fix the x-axis range.
    title : str, default 'Cross-validation splits'
        Figure title.

    Returns
    matplotlib.figure.Figure
        The figure object. Not saved; the caller decides where (if
        anywhere) to write it.
    """
    n_folds = len(splits)
    # Figure height grows with fold count so each fold's bar stays readable.
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.4 * n_folds))

    # Label only the first fold's bars so the legend has one Train and
    # one Test entry, not n_folds of each.
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        if len(train_idx) > 0:
            ax.barh(
                fold_num,
                len(train_idx),
                left=int(train_idx.min()),
                height=0.7,
                color="tab:blue",
                label="Train" if fold_idx == 0 else None,
            )
        if len(test_idx) > 0:
            ax.barh(
                fold_num,
                len(test_idx),
                left=int(test_idx.min()),
                height=0.7,
                color="tab:orange",
                label="Test" if fold_idx == 0 else None,
            )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Fold")
    ax.set_yticks(range(1, n_folds + 1))
    # Without this, fold 1 plots at the bottom because matplotlib y
    # increases upwards by default. We want fold 1 at the top so the
    # reader sees folds in time order.
    ax.invert_yaxis()
    ax.set_xlim(0, n_samples)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
