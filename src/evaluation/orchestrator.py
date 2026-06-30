"""End-to-end orchestration of the Sprint 4 evaluation pipeline.

Reads the processed dataset, generates per-fold predictions for all four
models on both CV schemes, runs the four downstream stages (per-fold and
across-fold aggregation, per-regime breakdown, Diebold-Mariano pairwise
tests, publication tables and figures), and writes every artifact to
disk under the chosen output directory. The whole pipeline is
deterministic given the seed and the input parquet.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.aggregation import aggregate_cv_results, compute_per_fold_metrics
from src.evaluation.diebold_mariano import diebold_mariano_test
from src.evaluation.predictions import generate_predictions
from src.evaluation.regime_evaluation import bootstrap_regime_metrics, evaluate_per_regime
from src.reporting.figures import (
    plot_cv_fold_variance,
    plot_model_comparison_bar,
    plot_regime_performance_heatmap,
)
from src.reporting.tables import (
    make_dm_test_table,
    make_overall_performance_table,
    make_per_regime_table,
)

# canonical model order: ARIMA first (baseline), then linear, then trees
_MODEL_ORDER = ("arima", "ridge", "xgboost", "lightgbm")
_DEFAULT_SCHEMES = ("expanding_window", "regime_aligned")


def run_evaluation(
    dataset_path: Path | str,
    output_dir: Path | str,
    models: Sequence[str] = _MODEL_ORDER,
    schemes: Sequence[str] = _DEFAULT_SCHEMES,
    metrics_for_figures: Sequence[str] = ("rmse",),
    seed: int = 42,
    quiet: bool = False,
) -> dict[str, Path]:
    """Runs the full Sprint 4 evaluation and writes every artifact to disk.

    Returns a dict mapping artifact-category names to their on-disk paths
    (the directory for category outputs, or the parquet path for the
    cached predictions). The seed is forwarded only to the bootstrap step;
    sklearn and ARIMA training are seeded inside CP2 via the cached
    tuning results.
    """
    start = time.perf_counter()
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for d in (predictions_dir, metrics_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    target_df = df.iloc[1:].reset_index(drop=True)
    y_train = target_df["gdp_growth"]

    predictions_path = predictions_dir / "predictions.parquet"
    predictions_df = generate_predictions(df)
    predictions_df = predictions_df[predictions_df["model"].isin(models)]
    predictions_df = predictions_df[predictions_df["scheme"].isin(schemes)]
    predictions_df.to_parquet(predictions_path)

    per_fold = compute_per_fold_metrics(predictions_df, y_train)
    aggregated = aggregate_cv_results(per_fold)
    per_regime = evaluate_per_regime(predictions_df, y_train)
    small_sample_cis = _bootstrap_flagged_regimes(predictions_df, per_regime, y_train, seed)
    dm_results = _build_dm_table(predictions_df, models, schemes)

    per_fold.to_csv(metrics_dir / "per_fold.csv", index=False)
    aggregated.to_csv(metrics_dir / "aggregated.csv", index=False)
    per_regime.to_csv(metrics_dir / "per_regime.csv", index=False)
    small_sample_cis.to_csv(metrics_dir / "small_sample_cis.csv", index=False)
    dm_results.to_csv(metrics_dir / "dm_test.csv", index=False)

    _write_tables(aggregated, per_regime, dm_results, schemes, tables_dir)
    _write_figures(aggregated, per_regime, per_fold, schemes, metrics_for_figures, figures_dir)

    elapsed = time.perf_counter() - start
    if not quiet:
        _print_summary(
            models=list(models),
            schemes=list(schemes),
            n_predictions=len(predictions_df),
            output_dir=output_dir,
            elapsed=elapsed,
        )

    return {
        "predictions": predictions_path,
        "metrics": metrics_dir,
        "tables": tables_dir,
        "figures": figures_dir,
    }


def _bootstrap_flagged_regimes(
    predictions_df: pd.DataFrame,
    per_regime: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Runs the IID bootstrap on every (model, scheme, regime) flagged small_sample.

    Returns a long DataFrame with lower / upper bounds per metric.
    """
    rows: list[dict] = []
    flagged = per_regime[per_regime["small_sample"]]
    for _, row in flagged.iterrows():
        subset = predictions_df[
            (predictions_df["model"] == row["model"])
            & (predictions_df["scheme"] == row["scheme"])
            & (predictions_df["regime"] == row["regime"])
        ]
        if len(subset) < 2:
            continue
        cis = bootstrap_regime_metrics(subset, y_train, random_state=seed)
        entry = {
            "model": row["model"],
            "scheme": row["scheme"],
            "regime": row["regime"],
            "n_observations": int(row["n_observations"]),
        }
        for metric_name, (lower, upper) in cis.items():
            entry[f"{metric_name}_lower"] = lower
            entry[f"{metric_name}_upper"] = upper
        rows.append(entry)
    return pd.DataFrame(rows)


def _build_dm_table(
    predictions_df: pd.DataFrame,
    models: Sequence[str],
    schemes: Sequence[str],
) -> pd.DataFrame:
    """Runs the six pairwise Diebold-Mariano tests per scheme, returns a flat DataFrame.

    Models are ordered by canonical _MODEL_ORDER. n_comparisons=6 sets the
    Bonferroni denominator visible on every row.
    """
    ordered = [m for m in _MODEL_ORDER if m in models]
    rows: list[dict] = []
    for scheme in schemes:
        for model_a, model_b in combinations(ordered, 2):
            errors_a = _errors_for(predictions_df, model_a, scheme)
            errors_b = _errors_for(predictions_df, model_b, scheme)
            result = diebold_mariano_test(errors_a, errors_b, n_comparisons=6)
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "scheme": scheme,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "p_value_bonferroni": result.p_value_bonferroni,
                    "n_observations": result.n_observations,
                }
            )
    return pd.DataFrame(rows)


def _errors_for(predictions_df: pd.DataFrame, model: str, scheme: str) -> pd.Series:
    """Returns the forecast errors (y_true - y_pred) for one (model, scheme), sorted by quarter."""
    subset = predictions_df[
        (predictions_df["model"] == model) & (predictions_df["scheme"] == scheme)
    ].sort_values("quarter")
    return subset["y_true"].to_numpy() - subset["y_pred"].to_numpy()


def _write_tables(
    aggregated: pd.DataFrame,
    per_regime: pd.DataFrame,
    dm_results: pd.DataFrame,
    schemes: Sequence[str],
    tables_dir: Path,
) -> None:
    """Writes the three tables, both formats, for each scheme."""
    for scheme in schemes:
        for stem, table in [
            ("overall_performance", make_overall_performance_table(aggregated, scheme)),
            ("per_regime", make_per_regime_table(per_regime, scheme)),
            ("dm_test", make_dm_test_table(dm_results, scheme)),
        ]:
            (tables_dir / f"{stem}_{scheme}.md").write_text(table["markdown"])
            (tables_dir / f"{stem}_{scheme}.tex").write_text(table["latex"])


def _write_figures(
    aggregated: pd.DataFrame,
    per_regime: pd.DataFrame,
    per_fold: pd.DataFrame,
    schemes: Sequence[str],
    metrics_for_figures: Iterable[str],
    figures_dir: Path,
) -> None:
    """Writes each figure as PNG (dpi 300) and PDF, for each (scheme, metric)."""
    for scheme in schemes:
        for metric in metrics_for_figures:
            for stem, fig in [
                (
                    "model_comparison",
                    plot_model_comparison_bar(aggregated, scheme, metric=metric),
                ),
                (
                    "regime_heatmap",
                    plot_regime_performance_heatmap(per_regime, scheme, metric=metric),
                ),
                (
                    "cv_fold_variance",
                    plot_cv_fold_variance(per_fold, scheme, metric=metric),
                ),
            ]:
                base = figures_dir / f"{stem}_{scheme}_{metric}"
                fig.savefig(base.with_suffix(".png"), dpi=300)
                fig.savefig(base.with_suffix(".pdf"))
                plt.close(fig)


def _print_summary(
    models: list[str],
    schemes: list[str],
    n_predictions: int,
    output_dir: Path,
    elapsed: float,
) -> None:
    """Prints a one-screen summary of what the orchestrator produced."""
    print()
    print("Sprint 4 evaluation complete")
    print(f"  models:        {', '.join(models)}")
    print(f"  schemes:       {', '.join(schemes)}")
    print(f"  predictions:   {n_predictions} rows")
    print(f"  output_dir:    {output_dir}")
    print(f"  predictions:   {output_dir / 'predictions' / 'predictions.parquet'}")
    print(f"  metrics:       {output_dir / 'metrics'}/*.csv")
    print(f"  tables:        {output_dir / 'tables'}/*.{{md,tex}}")
    print(f"  figures:       {output_dir / 'figures'}/*.{{png,pdf}}")
    print(f"  elapsed:       {elapsed:.1f}s")
    print()
