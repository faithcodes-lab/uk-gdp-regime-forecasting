"""Renders Figure 3.1: pipeline architecture as a six-box flow diagram.

Recreates the existing results/figures/pipeline_architecture.png with the
make-target labels sitting directly on each arrow shaft (a white halo
behind the text keeps it legible over the line) rather than floating
above or between the boxes.

Run with
    PYTHONPATH=. python scripts/pipeline_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

_OUT_PNG = Path("results/figures/pipeline_architecture.png")
_OUT_PDF = Path("results/figures/pipeline_architecture.pdf")

_BOX_W, _BOX_H = 3.2, 2.6
_COLORS = {
    "raw": "#7CA9C9",
    "frozen": "#F5A85A",
    "regimes": "#8FC08A",
    "tuning": "#D98C8C",
    "eval": "#B7A0D6",
    "shap": "#B79A93",
}

_BOXES = [
    # (key, col, row, title, body_lines)
    ("raw", 0, 1, "Raw sources", ["ONS", "Bank of England", "FRED"]),
    ("frozen", 1, 1, "Frozen dataset", ["data/processed/", "final_dataset.parquet", "(104 x 17)"]),
    ("regimes", 2, 1, "Regimes + break tests", ["six regimes", "Chow, Bai-Perron, ICSS"]),
    ("tuning", 2, 0, "Tuning and training", ["ARIMA, Ridge,", "XGBoost, LightGBM"]),
    ("eval", 1, 0, "Evaluation", ["RMSE, MAE, MASE, R2", "Diebold-Mariano", "per-regime, tables, figures"]),
    ("shap", 0, 0, "SHAP analysis", ["per-regime attributions", "+ Spearman rank stability"]),
]

_COL_X = {0: 0.5, 1: 5.5, 2: 10.5}
_ROW_Y = {0: 0.5, 1: 5.0}

_ARROWS = [
    # (from_key, to_key, label, direction) direction: 'h' horizontal, 'v' vertical
    ("raw", "frozen", "make data", "h"),
    ("frozen", "regimes", "make regimes /\nmake break-tests", "h"),
    ("regimes", "tuning", "make tune /\nmake train", "v"),
    ("tuning", "eval", "make evaluate", "h_rev"),
    ("eval", "shap", "make shap", "h_rev"),
]


def _box_center(key: str) -> tuple[float, float]:
    for k, col, row, *_ in _BOXES:
        if k == key:
            return _COL_X[col] + _BOX_W / 2, _ROW_Y[row] + _BOX_H / 2
    raise KeyError(key)


def render_pipeline_diagram() -> plt.Figure:
    """Builds and returns the pipeline architecture figure."""
    fig, ax = plt.subplots(figsize=(14, 7.3))

    for key, col, row, title, body_lines in _BOXES:
        x, y = _COL_X[col], _ROW_Y[row]
        box = FancyBboxPatch(
            (x, y),
            _BOX_W,
            _BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#4d4d4d",
            facecolor=_COLORS[key],
        )
        ax.add_patch(box)
        cx = x + _BOX_W / 2
        ax.text(cx, y + _BOX_H - 0.45, title, ha="center", va="center", fontsize=13, fontweight="bold")
        n = len(body_lines)
        for i, line in enumerate(body_lines):
            ly = y + _BOX_H - 0.95 - i * 0.5
            ax.text(cx, ly, line, ha="center", va="center", fontsize=10.5)

    for from_key, to_key, label, direction in _ARROWS:
        fx, fy = _box_center(from_key)
        tx, ty = _box_center(to_key)

        if direction == "h":
            start = (fx + _BOX_W / 2, fy)
            end = (tx - _BOX_W / 2, ty)
        elif direction == "h_rev":
            start = (fx - _BOX_W / 2, fy)
            end = (tx + _BOX_W / 2, ty)
        else:  # vertical
            start = (fx, fy - _BOX_H / 2)
            end = (tx, ty + _BOX_H / 2)

        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=18),
        )

        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        is_vertical = abs(start[0] - end[0]) < 0.01
        if is_vertical:
            # label sits just to the right of the line, not touching it
            ax.text(mx + 0.25, my, label, ha="left", va="center", fontsize=10, fontstyle="italic")
        else:
            # label floats just above the line, not touching it
            ax.text(mx, my + 0.18, label, ha="center", va="bottom", fontsize=10, fontstyle="italic")

    ax.set_xlim(-0.3, 14.2)
    ax.set_ylim(-0.3, 8.1)
    ax.set_title("Figure 3.1: Pipeline architecture", fontsize=16, pad=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    fig = render_pipeline_diagram()
    _OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(_OUT_PDF, bbox_inches="tight")
    print(f"wrote {_OUT_PNG}")
    print(f"wrote {_OUT_PDF}")


if __name__ == "__main__":
    main()
