"""Exploratory data analysis for the UK GDP regime forecasting project.

What this script does?

- Reads the final modelling dataset (104 quarters of UK macro variables).
- Reads the six regime periods from the project config.
- Saves four figures in ``results/figures/eda/`` and one short Markdown
  summary in ``results/eda-summary.md``.

Run with
    make eda
or  python scripts/eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from statsmodels.tsa.stattools import adfuller

from src.data.config import pipeline_config, regimes_config
from src.logging_setup import configure_logging

# Fix the random seed in case any plotting call uses randomness; nothing in
# this script should actually need it, but this keeps re-runs identical.
np.random.seed(42)

# A fixed palette so all four figures share the same regime colours.
# Colours come from ColorBrewer "Set2", a palette that is readable for people
# with colour-blindness.
_REGIME_PALETTE = [
    "#66c2a5",  # Pre-GFC Stability
    "#fc8d62",  # Global Financial Crisis
    "#8da0cb",  # Post-GFC Recovery
    "#e78ac3",  # Brexit Transition
    "#a6d854",  # COVID-19 Shock
    "#ffd92f",  # Post-COVID Recovery
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dataset() -> pd.DataFrame:
    """Read the final dataset parquet file and return it as a DataFrame."""
    parquet_path = _repo_root() / pipeline_config()["paths"]["final_dataset"]
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_regimes() -> list[dict]:
    """Read the six regime periods from the project config."""
    regimes = regimes_config()["regimes"]
    # The config file stores each regime's start and end as a plain date.
    # Convert to pandas timestamps so they can be compared to df["date"].
    for r in regimes:
        r["start"] = pd.Timestamp(r["start"])
        r["end"] = pd.Timestamp(r["end"])
    return regimes


def _regime_labels(df: pd.DataFrame, regimes: list[dict]) -> pd.Series:
    """Return a series naming the regime each row belongs to."""
    labels = pd.Series(index=df.index, dtype="object")
    for r in regimes:
        mask = (df["date"] >= r["start"]) & (df["date"] <= r["end"])
        labels[mask] = r["label"]
    return labels


def plot_gdp_timeseries(df: pd.DataFrame, regimes: list[dict], out_path: Path) -> None:
    """Plot GDP growth across the whole sample with the six regimes shaded."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], df["gdp_growth"], color="black", linewidth=1.2)
    for i, r in enumerate(regimes):
        ax.axvspan(
            r["start"],
            r["end"],
            color=_REGIME_PALETTE[i],
            alpha=0.35,
            label=r["label"],
        )
    ax.set_xlabel("Quarter")
    ax.set_ylabel("GDP growth, % quarter on quarter")
    ax.set_title(
        "UK quarterly GDP growth with the six hypothesised regimes (2000-2025)")
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    # End the x-axis at the final quarter in the data so the line finishes flush at the right edge.
    ax.set_xlim(right=df["date"].max())
    # Place ticks every 5 years ending on 2025 so the final year is labelled.
    tick_years = (2000, 2005, 2010, 2015, 2020, 2025)
    ax.set_xticks(
        [pd.Timestamp(f"{y}-01-01") for y in tick_years],
        labels=[str(y) for y in tick_years],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gdp_by_regime(
    df: pd.DataFrame,
    regime_labels: pd.Series,
    regimes: list[dict],
    out_path: Path,
) -> None:
    """Box plot of GDP growth values inside each regime."""
    order = [r["label"] for r in regimes]
    palette = {label: _REGIME_PALETTE[i] for i, label in enumerate(order)}
    plot_df = pd.DataFrame(
        {"regime": regime_labels.values, "gdp_growth": df["gdp_growth"].values})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=plot_df,
        x="regime",
        y="gdp_growth",
        order=order,
        hue="regime",
        hue_order=order,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Regime")
    ax.set_ylabel("GDP growth, % quarter on quarter")
    ax.set_title("Distribution of UK quarterly GDP growth by regime")
    # Rotate the regime names so they fit on the axis without overlapping.
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gdp_rolling(df: pd.DataFrame, out_path: Path) -> None:
    """Plot the rolling 4-quarter mean of GDP growth from the stored column."""
    # Use the engineered column directly; it is populated from 2000 Q1 because
    # the pipeline computes the rolling mean before trimming to the project window.
    rolling_mean = df["gdp_rolling_mean_4q"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["date"], rolling_mean, color="#1f77b4")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Rolling 4q mean (% QoQ)")
    ax.set_title("Rolling 4-quarter mean of UK GDP growth")
    ax.grid(True, alpha=0.3)
    # End the x-axis at the final quarter in the data so the line finishes flush at the right edge.
    ax.set_xlim(right=df["date"].max())
    # Place ticks every 5 years ending on 2025 so the final year is labelled.
    tick_years = (2000, 2005, 2010, 2015, 2020, 2025)
    ax.set_xticks(
        [pd.Timestamp(f"{y}-01-01") for y in tick_years],
        labels=[str(y) for y in tick_years],
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Annotated Pearson correlation heatmap for every numeric column."""
    numeric = df.drop(columns=["date"])
    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(
        "Pearson correlation across all features (including the target)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_stats_by_regime(
    df: pd.DataFrame, regime_labels: pd.Series, regimes: list[dict]
) -> pd.DataFrame:
    """Per-regime summary of gdp_growth with count, mean, std, min, max, skew, kurtosis."""
    order = [r["label"] for r in regimes]
    rows = []
    for label in order:
        series = df.loc[regime_labels == label, "gdp_growth"].dropna()
        rows.append(
            {
                "regime": label,
                "count": int(series.count()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "skewness": series.skew(),
                "kurtosis": series.kurt(),
            }
        )
    return pd.DataFrame(rows)


def overall_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Pandas describe() transposed so each row is one column from the dataset."""
    return df.drop(columns=["date"]).describe().T.round(3)


def adf_table(df: pd.DataFrame) -> pd.DataFrame:
    """Run an ADF test on every numeric column; return statistic, p-value, pass/fail."""
    rows = []
    for col in df.columns:
        if col == "date":
            continue
        series = df[col].dropna()
        # ADF needs at least a handful of points; skip very short series safely.
        if len(series) < 12:
            rows.append(
                {
                    "column": col,
                    "adf_stat": float("nan"),
                    "p_value": float("nan"),
                    "verdict_at_5pct": "insufficient observations",
                }
            )
            continue
        stat, p_value, *_ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "column": col,
                "adf_stat": stat,
                "p_value": p_value,
                "verdict_at_5pct": "pass" if p_value < 0.05 else "fail",
            }
        )
    return pd.DataFrame(rows)


def missing_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column that has at least one missing value."""
    counts = df.drop(columns=["date"]).isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return (
        counts.rename("missing_count").to_frame(
        ).reset_index().rename(columns={"index": "column"})
    )


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table, without needing `tabulate`."""
    headers = [str(c) for c in df.columns]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "n/a" if pd.isna(v) else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


_PREAMBLE = """\
# Exploratory data analysis: UK quarterly GDP, 2000-2025

This page describes what the final modelling dataset looks like before any
formal structural-break tests are run. The point of the EDA is to see the
data clearly and to flag candidate points where the level or volatility of
GDP growth appears to change.

It does **not** claim that any regime boundary is statistically confirmed.
The Chow tests and the Bai-Perron sensitivity sweep that come next in the
sprint are what will formally accept or reject each boundary.

Dataset: `data/processed/final_dataset.parquet` (104 quarters, 18 columns,
2000 Q1 - 2025 Q4). Target: `gdp_growth` (% quarter on quarter). Regime
periods used for shading and grouping below come from `config/regimes.yaml`.
"""


def _build_summary_markdown(
    *,
    summary_by_regime: pd.DataFrame,
    describe_all: pd.DataFrame,
    adf: pd.DataFrame,
    missing: pd.DataFrame,
) -> str:
    parts = [_PREAMBLE]

    parts.append(
        "## Figure 1: GDP growth over time, with the six regime bands\n")
    parts.append(
        "![GDP growth time series](figures/eda/gdp_growth_timeseries.png)\n")
    parts.append(
        "Quarterly UK GDP growth over the sample period is shown with six "
        "hypothesised regimes indicated by shaded bands. Growth remains "
        "within a relatively narrow range for most of the sample, typically "
        "between -1% and +1.5% quarter-on-quarter. Two episodes deviate "
        "substantially from this pattern. During the global financial crisis, "
        "GDP contracted steadily, reaching approximately -2% in 2008–2009. "
        "The COVID-19 shock generated a much larger disruption, with GDP "
        "declining by approximately -20% in 2020 Q2, followed by an increase "
        "of approximately +17% in the subsequent quarter. The differing "
        "dynamics of these episodes (a prolonged contraction during the "
        "global financial crisis and a sharp contraction followed by rapid "
        "growth during COVID-19) suggest the presence of distinct economic "
        "regimes. The shaded intervals represent economically motivated "
        "breakpoints associated with the global financial crisis, the 2016 "
        "EU referendum, and the COVID-19 pandemic. These classifications "
        "are evaluated formally using structural break tests in subsequent "
        "analysis.\n"
    )

    parts.append("## Figure 2: Distribution of GDP growth by regime\n")
    parts.append(
        "![GDP growth by regime](figures/eda/gdp_growth_by_regime.png)\n")
    parts.append(
        "Figure 2 presents the distribution of quarterly GDP growth across "
        "regimes. The pre-global financial crisis, post-global financial "
        "crisis, Brexit, and post-COVID regimes exhibit similar "
        "distributions, characterised by relatively low dispersion and "
        "median growth rates slightly above zero. In contrast, the crisis "
        "regimes display distinct characteristics. The global financial "
        "crisis regime is centred marginally below zero with moderate "
        "dispersion, whereas the COVID-19 regime exhibits substantially "
        "greater variability, reflecting the extreme contraction and "
        "subsequent recovery observed in 2020. The evidence suggests that "
        "differences across regimes arise from both changes in average "
        "growth and changes in volatility, with the COVID-19 period "
        "primarily distinguished by elevated variance. Interpretation of "
        "the crisis-period distributions should recognise the limited "
        "number of observations available within these regimes.\n"
    )

    parts.append("## Figure 3: Rolling 4-quarter mean of GDP growth\n")
    parts.append(
        "![Rolling 4-quarter mean](figures/eda/gdp_growth_rolling.png)\n")
    parts.append(
        "Figure 3 reports the four-quarter rolling mean of GDP growth. The "
        "smoothed series highlights three broad phases: a sustained decline "
        "during 2008–2009, a relatively stable period centred around 0.5% "
        "growth between 2010 and 2019, and a pronounced decline followed by "
        "recovery during the COVID-19 period. The series is derived from "
        "the engineered gdp_rolling_mean_4q variable and spans the full "
        "sample from 2000 Q1. These movements are consistent with potential "
        "shifts in the underlying growth process and provide descriptive "
        "evidence for subsequent structural break testing.\n"
    )

    parts.append("## Figure 4: Pearson correlation heatmap\n")
    parts.append(
        "![Pearson correlation heatmap](figures/eda/correlation_heatmap.png)\n")
    parts.append(
        "Figure 4 presents the Pearson correlation matrix for all variables. "
        "The strongest correlations with GDP growth are observed for "
        "government consumption growth (0.90) and gross fixed capital "
        "formation growth (0.73). Given that both variables are expenditure "
        "components of GDP, these relationships largely reflect accounting "
        "identities rather than independent predictive content. Most "
        "macroeconomic and financial indicators, including unemployment, "
        "inflation, the policy interest rate, the exchange rate, oil "
        "prices, and confidence measures, exhibit weak to moderate "
        "correlations with GDP growth. Engineered GDP features display "
        "moderate correlations by construction (gdp_rolling_mean_4q = 0.53; "
        "gdp_yoy = 0.46), while lagged GDP terms are comparatively weakly "
        "associated with the target. The matrix also reveals substantial "
        "collinearity among several predictors, including "
        "gdp_rolling_mean_4q and gdp_yoy (0.99), business confidence and "
        "its rolling mean (0.81), and the yield-curve slope with both the "
        "policy rate (-0.71) and unemployment (0.75). These relationships "
        "are relevant for subsequent model estimation and interpretation.\n"
    )

    parts.append("## Summary statistics for `gdp_growth` by regime\n")
    parts.append(_df_to_markdown(summary_by_regime))
    parts.append("\n")
    parts.append(
        "Means and spreads vary visibly across regimes. The small GFC and "
        "COVID samples make any moment estimate noisy; the project uses "
        "bootstrap intervals for per-regime claims in the SHAP analysis later.\n"
    )

    describe_view = describe_all.reset_index().rename(
        columns={"index": "column"})
    parts.append("## Overall `describe()` for every column\n")
    parts.append(_df_to_markdown(describe_view))
    parts.append("\n")
    parts.append(
        "The describe table makes the very different feature scales visible "
        "(`bank_rate` in percent, `brent_oil` in dollars per barrel, "
        "`trade_balance` in pounds). Models that are sensitive to scale "
        "(Ridge in particular) will need standardisation in the modelling step.\n"
    )

    parts.append("## ADF stationarity tests at the 5% level\n")
    parts.append(_df_to_markdown(adf))
    parts.append("\n")
    parts.append(
        "An ADF test asks whether each series wanders without returning to a "
        "stable level. A small p-value (below 0.05) returns a **pass** verdict "
        "here, meaning the data look stationary in level. Series that come "
        "back **fail** (non-stationary) will need attention in the modelling "
        "step, typically by working with first differences.\n"
    )

    parts.append("## Missing-value counts\n")
    if missing.empty:
        parts.append(
            "No missing values in any column. This is by design: the data "
            "pipeline engineers the lag and rolling features **before** "
            "trimming the dataset to the 2000-2025 window, so the "
            "engineered features at the start of the sample draw on pre-2000 "
            "history instead of producing NaNs. Nothing is filled or imputed "
            "here.\n"
        )
    else:
        parts.append(_df_to_markdown(missing))
        parts.append("\n")
        parts.append(
            "These NaNs are the leading rows of the lag and rolling features. "
            "They are listed here and left as-is; the models in the next "
            "sprint will skip or shift the affected rows rather than impute "
            "any values.\n"
        )

    return "\n".join(parts)


def main() -> None:
    configure_logging()
    logger.info("Loading dataset and regime configuration")
    df = _load_dataset()
    regimes = _load_regimes()
    regime_labels = _regime_labels(df, regimes)

    fig_dir = _repo_root() / "results" / "figures" / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_path = _repo_root() / "results" / "eda-summary.md"

    logger.info("Plotting figure 1: GDP growth time series with regime bands")
    plot_gdp_timeseries(df, regimes, fig_dir / "gdp_growth_timeseries.png")

    logger.info(
        "Plotting figure 2: GDP growth distribution by regime (box plot)")
    plot_gdp_by_regime(df, regime_labels, regimes,
                       fig_dir / "gdp_growth_by_regime.png")

    logger.info("Plotting figure 3: rolling 4-quarter mean (stored column)")
    plot_gdp_rolling(df, fig_dir / "gdp_growth_rolling.png")

    logger.info("Plotting figure 4: Pearson correlation heatmap")
    plot_correlation_heatmap(df, fig_dir / "correlation_heatmap.png")

    logger.info("Computing summary tables")
    summary_by_regime = summary_stats_by_regime(df, regime_labels, regimes)
    describe_all = overall_describe(df)
    adf = adf_table(df)
    missing = missing_value_counts(df)

    logger.info("Writing summary markdown to {}", summary_path)
    summary_md = _build_summary_markdown(
        summary_by_regime=summary_by_regime,
        describe_all=describe_all,
        adf=adf,
        missing=missing,
    )
    summary_path.write_text(summary_md)
    logger.info("EDA complete")


if __name__ == "__main__":
    main()
