"""Publication-quality figure of UK GDP growth with the six regime bands.

This is the methodology-chapter figure (a 300 DPI PNG plus a vector PDF
for LaTeX). It is a polished sibling of the exploratory CP0 figure
``results/figures/eda/gdp_growth_timeseries.png`` (150 DPI) — both
co-exist deliberately, the EDA one for the exploratory narrative and
this one for the dissertation.

The legend lists each regime with its date range, derived from
``config/regimes.yaml`` rather than hardcoded, so updates to the regime
config flow through automatically.

The figure intentionally contains no annotations, subtitle, or
interpretation text. The reader is given the data, the axes, and the
regime ranges; the LaTeX caption around the figure does the prose work.

Run with
--------
    make figure-regimes
or  PYTHONPATH=. python -m src.regimes.visualise
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from src.data.config import pipeline_config, regimes_config
from src.logging_setup import configure_logging

# ColorBrewer "Set2" — the same colour-blind-safe palette as the CP0
# EDA figures, so the two figures look visually consistent.
_REGIME_PALETTE = [
    "#66c2a5",  # Pre-GFC Stability
    "#fc8d62",  # Global Financial Crisis
    "#8da0cb",  # Post-GFC Recovery
    "#e78ac3",  # Brexit Transition
    "#a6d854",  # COVID-19 Shock
    "#ffd92f",  # Post-COVID Recovery
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dataset() -> pd.DataFrame:
    parquet_path = _repo_root() / pipeline_config()["paths"]["final_dataset"]
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _resolve_regimes(regimes: list[dict] | None) -> list[dict]:
    if regimes is None:
        return list(regimes_config()["regimes"])
    return list(regimes)


def _quarter_label(date: pd.Timestamp) -> str:
    """Format a date as e.g. '2008 Q1'."""
    period = pd.Timestamp(date).to_period("Q")
    return f"{period.year} Q{period.quarter}"


def _legend_label(regime: dict) -> str:
    """Format a regime as 'Label (YYYY QN - YYYY QN)'."""
    start_q = _quarter_label(regime["start"])
    end_q = _quarter_label(regime["end"])
    return f"{regime['label']} ({start_q} - {end_q})"


def plot_gdp_with_regimes(
    df: pd.DataFrame,
    regimes: list[dict] | None = None,
    *,
    png_path: Path | None = None,
    pdf_path: Path | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (12, 5),
) -> matplotlib.figure.Figure:
    """Plot quarterly GDP growth with the six regimes shaded behind the line.

    Args:
        df: Dataset; must contain ``date`` and ``gdp_growth`` columns.
            Not mutated.
        regimes: Regime list. ``None`` (default) loads from
            ``regimes_config()``. The legend lists each regime with its
            date range (e.g. "Pre-GFC Stability (2000 Q1 - 2007 Q4)").
        png_path: If set, save the figure as PNG at this path.
        pdf_path: If set, save the figure as a vector PDF.
        dpi: Resolution for the PNG. Defaults to 300 (publication).
        figsize: Width, height in inches. Defaults to (12, 5).

    Returns:
        The matplotlib Figure object, so callers and tests can inspect
        axes, the legend, and patches.

    Raises:
        ValueError: If ``df`` is missing ``date`` or ``gdp_growth``.
    """
    for required_col in ("date", "gdp_growth"):
        if required_col not in df.columns:
            raise ValueError(
                f"df must contain a {required_col!r} column; got " f"{list(df.columns)}"
            )

    regimes_list = _resolve_regimes(regimes)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df["date"], df["gdp_growth"], color="black", linewidth=1.3)

    for i, r in enumerate(regimes_list):
        ax.axvspan(
            pd.Timestamp(r["start"]),
            pd.Timestamp(r["end"]),
            color=_REGIME_PALETTE[i % len(_REGIME_PALETTE)],
            alpha=0.35,
            label=_legend_label(r),
        )

    ax.set_xlabel("Quarter")
    ax.set_ylabel("GDP growth, % quarter on quarter")
    ax.set_title("UK quarterly GDP growth and the six economic regimes, 2000-2025")
    ax.grid(True, alpha=0.3)

    # End the x-axis at the final quarter so the line finishes flush at
    # the right edge, matching the CP0 EDA convention.
    ax.set_xlim(right=df["date"].max())
    # Ticks every 5 years ending on 2025 so the final year is labelled.
    tick_years = (2000, 2005, 2010, 2015, 2020, 2025)
    ax.set_xticks(
        [pd.Timestamp(f"{y}-01-01") for y in tick_years],
        labels=[str(y) for y in tick_years],
    )
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()

    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=dpi, format="png")
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format="pdf")

    return fig


def main() -> None:
    configure_logging()
    logger.info("Loading dataset and regime configuration")
    df = _load_dataset()

    out_dir = _repo_root() / "results" / "figures"
    png_path = out_dir / "regime_visualisation.png"
    pdf_path = out_dir / "regime_visualisation.pdf"

    logger.info("Rendering publication figure (300 DPI PNG + vector PDF)")
    fig = plot_gdp_with_regimes(df, png_path=png_path, pdf_path=pdf_path)
    plt.close(fig)

    logger.info("Wrote {}", png_path.relative_to(_repo_root()))
    logger.info("Wrote {}", pdf_path.relative_to(_repo_root()))


if __name__ == "__main__":
    main()
