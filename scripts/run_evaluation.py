"""CLI wrapper around src.evaluation.orchestrator.run_evaluation.

Runs the full Sprint 4 evaluation on the processed dataset and writes
every artifact under results/. Defaults match the dissertation setup
(all four models, both CV schemes, RMSE figures only, seed 42).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.orchestrator import run_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Sprint 4 evaluation pipeline end-to-end.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/final_dataset.parquet"),
        help="Path to the processed dataset parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory under which predictions, metrics, tables, and figures will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["arima", "ridge", "xgboost", "lightgbm"],
        choices=["arima", "ridge", "xgboost", "lightgbm"],
        help="Which models to evaluate.",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["expanding_window", "regime_aligned"],
        choices=["expanding_window", "regime_aligned"],
        help="Which CV schemes to evaluate.",
    )
    parser.add_argument(
        "--metrics-for-figures",
        nargs="+",
        default=["rmse"],
        choices=["rmse", "mae", "mase", "r2"],
        help="Which metrics receive a figure trio per scheme.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the small-sample bootstrap CIs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the stdout summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        models=args.models,
        schemes=args.schemes,
        metrics_for_figures=args.metrics_for_figures,
        seed=args.seed,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
