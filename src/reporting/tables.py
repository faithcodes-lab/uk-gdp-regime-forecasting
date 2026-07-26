"""Publication tables for the evaluation: overall, per-regime, and Diebold-Mariano.

Each public function returns a dict with "markdown" and "latex" keys for
the same table. Markdown is built by hand (no tabulate dependency); LaTeX
is built by hand with booktabs rules (no jinja2 dependency) for symmetry.
Captions use hedged language consistent with the project's writing
discipline. Numeric metrics are rounded to 3 decimal places; p-values
below 0.001 render as "< 0.001". Small-sample regimes (n < 10 quarters)
are flagged with an asterisk and a footnote.
"""

from __future__ import annotations

import pandas as pd

_THREE_DP = "{:.3f}"
_SMALL_SAMPLE_MARK = "*"
_SMALL_SAMPLE_FOOTNOTE = (
    "* Small sample (n < 10 quarters); bootstrap confidence intervals " "reported separately."
)


def _format_metric(value: float) -> str:
    return _THREE_DP.format(value)


def _format_p_value(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return _THREE_DP.format(p)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Returns a GFM-style markdown table for a DataFrame whose cells are already strings."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep, *rows])


def _df_to_latex(df: pd.DataFrame, caption: str) -> str:
    """Returns a hand-built LaTeX table for an already-formatted DataFrame.

    Uses booktabs rules (toprule, midrule, bottomrule); first column
    left-aligned, remaining columns right-aligned. Cell values and the
    caption are not escaped for LaTeX specials because they are entirely
    author-controlled (no user input flows into a published table).
    """
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)
    header = " & ".join(df.columns) + " \\\\"
    rows = [" & ".join(str(v) for v in row) + " \\\\" for _, row in df.iterrows()]
    lines = [
        "\\begin{table}",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def make_overall_performance_table(aggregated_results: pd.DataFrame, scheme: str) -> dict[str, str]:
    """Builds the overall performance table for one CV scheme.

    Rows are models; columns show mean for each metric with the standard
    deviation across folds in parentheses. Lower RMSE, MAE, and MASE
    indicate better forecast accuracy; higher R2 indicates better fit.
    """
    filtered = aggregated_results[aggregated_results["scheme"] == scheme].copy()

    formatted = pd.DataFrame({"Model": filtered["model"].values})
    for metric, label in [("rmse", "RMSE"), ("mae", "MAE"), ("mase", "MASE"), ("r2", "R2")]:
        mean_col = filtered[f"mean_{metric}"]
        std_col = filtered[f"std_{metric}"]
        formatted[label] = [
            (f"{_format_metric(m)} ({_format_metric(s)})" if pd.notna(s) else _format_metric(m))
            for m, s in zip(mean_col, std_col)
        ]

    caption = (
        f"Overall forecast accuracy ({scheme.replace('_', '-')} CV). "
        "Lower values indicate better performance for RMSE, MAE, and MASE; "
        "higher R2 indicates better fit. Mean across folds with standard "
        "deviation in parentheses."
    )
    md = "\n".join([f"**{caption}**", "", _df_to_markdown(formatted)])
    latex = _df_to_latex(formatted, caption=caption)
    return {"markdown": md, "latex": latex}


def make_per_regime_table(per_regime_results: pd.DataFrame, scheme: str) -> dict[str, str]:
    """Builds the per-regime performance table for one CV scheme.

    Rows are (model, regime); columns are the four metrics and the
    observation count. Small-sample regimes carry an asterisk and the
    footnote appears below the table.
    """
    filtered = per_regime_results[per_regime_results["scheme"] == scheme].copy()

    regime_label = filtered.apply(
        lambda r: r["regime"] + (_SMALL_SAMPLE_MARK if r["small_sample"] else ""),
        axis=1,
    )
    formatted = pd.DataFrame(
        {
            "Model": filtered["model"].values,
            "Regime": regime_label.values,
            "n": filtered["n_observations"].values,
            "RMSE": [_format_metric(v) for v in filtered["rmse"]],
            "MAE": [_format_metric(v) for v in filtered["mae"]],
            "MASE": [_format_metric(v) for v in filtered["mase"]],
            "R2": [_format_metric(v) for v in filtered["r2"]],
        }
    )

    caption = (
        f"Per-regime forecast accuracy ({scheme.replace('_', '-')} CV). "
        "Lower values indicate better performance for RMSE, MAE, and MASE; "
        "higher R2 indicates better fit. Asterisk marks small-sample regimes."
    )

    has_small_sample = bool(filtered["small_sample"].any())
    md_parts = [f"**{caption}**", "", _df_to_markdown(formatted)]
    if has_small_sample:
        md_parts.extend(["", _SMALL_SAMPLE_FOOTNOTE])
    latex = _df_to_latex(formatted, caption=caption)
    if has_small_sample:
        latex = latex + "\n\n" + _SMALL_SAMPLE_FOOTNOTE

    return {"markdown": "\n".join(md_parts), "latex": latex}


def make_dm_test_table(dm_results_df: pd.DataFrame, scheme: str) -> dict[str, str]:
    """Builds the Diebold-Mariano pairwise comparison table for one CV scheme.

    Each row is a (model_a, model_b) comparison; columns show the DM
    statistic, the raw p-value, and the Bonferroni-corrected p-value
    (against the six-comparison family). p-values format to 3 decimals,
    or "< 0.001" below that threshold.
    """
    filtered = dm_results_df[dm_results_df["scheme"] == scheme].copy()

    formatted = pd.DataFrame(
        {
            "Model A": filtered["model_a"].values,
            "Model B": filtered["model_b"].values,
            "n": filtered["n_observations"].values,
            "Statistic": [_format_metric(v) for v in filtered["statistic"]],
            "p-value": [_format_p_value(v) for v in filtered["p_value"]],
            "p-value (Bonferroni)": [_format_p_value(v) for v in filtered["p_value_bonferroni"]],
        }
    )

    caption = (
        f"Diebold-Mariano pairwise comparisons ({scheme.replace('_', '-')} CV). "
        "Harvey-Leybourne-Newbold small-sample correction applied; reference "
        "t-distribution with n minus 1 degrees of freedom. Bonferroni "
        "correction applied across the six-comparison family. Where the "
        "Bonferroni-corrected p-value is below 0.05, the test indicates a "
        "significant difference in forecast accuracy."
    )

    md = "\n".join([f"**{caption}**", "", _df_to_markdown(formatted)])
    latex = _df_to_latex(formatted, caption=caption)
    return {"markdown": md, "latex": latex}
