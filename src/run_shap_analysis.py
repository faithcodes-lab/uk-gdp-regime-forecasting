"""Master script for the regime-aware SHAP analysis (make shap).

Runs the whole pipeline end to end on the reserved full-history XGBoost
model and persists every output so the analysis is reproducible: global
SHAP importance, per-regime importance and rankings, the pairwise
Spearman stability matrix with the Akoglu bands, and bootstrap confidence
intervals for the two small regimes. TreeSHAP is exact and deterministic
for tree models and the bootstrap is seeded, so re-running reproduces
identical numbers.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.explainability.regime_shap import (
    compute_per_regime_rankings,
    compute_per_regime_shap,
)
from src.explainability.shap_compute import (
    compute_shap_values,
    load_best_model,
    load_global_regimes,
    load_global_X,
)
from src.explainability.stability import (
    bootstrap_rankings,
    bootstrap_spearman_ci,
    classify_stability,
    pairwise_spearman_matrix,
)
from src.explainability.visualisations import (
    plot_global_importance,
    plot_per_regime_importance,
    plot_stability_heatmap,
)
from src.logging_setup import configure_logging

_SHAP_DIR = Path("results/shap")
_FIG_DIR = Path("results/figures")

# Chronological order so every table and figure reads left to right in time.
_REGIME_ORDER = [
    "Pre-GFC Stability",
    "Global Financial Crisis",
    "Post-GFC Recovery",
    "Brexit Transition",
    "COVID-19 Shock",
    "Post-COVID Recovery",
]


def _global_importance(explanation) -> pd.Series:
    """Mean absolute SHAP per feature across all rows."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    return pd.Series(mean_abs, index=list(explanation.feature_names), name="mean_abs_shap")


def _per_regime_importance(per_regime_shap: dict) -> pd.DataFrame:
    """Feature-by-regime matrix of mean absolute SHAP."""
    cols = {
        regime: pd.Series(np.abs(exp.values).mean(axis=0), index=list(exp.feature_names))
        for regime, exp in per_regime_shap.items()
    }
    return pd.DataFrame(cols)


def _ordered(present) -> list[str]:
    return [r for r in _REGIME_ORDER if r in set(present)]


def run_shap_analysis(
    output_dir: Path | str = _SHAP_DIR,
    figures_dir: Path | str = _FIG_DIR,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Runs the full SHAP analysis, writes CSVs and figures, and returns the key tables."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, name = load_best_model()
    logger.info("reserved gradient boosting model: {}", name)
    X = load_global_X()
    regimes = load_global_regimes()
    logger.info("feature matrix {}, regimes {}", X.shape, regimes.nunique())

    logger.info("computing global SHAP")
    global_exp = compute_shap_values(model, X)
    global_imp = _global_importance(global_exp).sort_values(ascending=False)
    global_df = global_imp.reset_index().rename(columns={"index": "feature"})
    global_df["rank"] = global_df["mean_abs_shap"].rank(ascending=False, method="average")
    global_df.to_csv(output_dir / "global_importance.csv", index=False)

    logger.info("computing per-regime SHAP and rankings")
    per_regime = compute_per_regime_shap(model, X, regimes)
    rankings, metadata = compute_per_regime_rankings(per_regime)
    order = _ordered(rankings.columns)
    rankings = rankings[order]
    importance = _per_regime_importance(per_regime)[order]
    importance.to_csv(output_dir / "per_regime_importance.csv")
    rankings.to_csv(output_dir / "per_regime_rankings.csv")

    logger.info("computing pairwise Spearman stability matrix")
    matrix = pairwise_spearman_matrix(rankings).loc[order, order]
    matrix.to_csv(output_dir / "stability_matrix.csv")

    small_regimes = {r for r, m in metadata.items() if m["small_sample"]}

    pair_rows = [
        {
            "regime_a": a,
            "regime_b": b,
            "spearman_rho": float(matrix.loc[a, b]),
            "band": classify_stability(float(matrix.loc[a, b])),
            "involves_small_regime": a in small_regimes or b in small_regimes,
        }
        for a, b in combinations(order, 2)
    ]
    pd.DataFrame(pair_rows).to_csv(output_dir / "stability_pairs.csv", index=False)

    logger.info(
        "bootstrapping confidence intervals for small-regime pairs (n_bootstrap={})", n_bootstrap
    )
    boot_cache: dict[str, np.ndarray] = {}

    def _boot(regime: str) -> np.ndarray:
        if regime not in boot_cache:
            x_regime = X.loc[(regimes == regime).to_numpy()]
            boot_cache[regime] = bootstrap_rankings(
                model, x_regime, n_bootstrap=n_bootstrap, random_state=seed
            )
        return boot_cache[regime]

    ci_rows = []
    for a, b in combinations(order, 2):
        if a in small_regimes or b in small_regimes:
            lower, upper = bootstrap_spearman_ci(_boot(a), _boot(b), n_bootstrap=n_bootstrap)
            ci_rows.append(
                {
                    "regime_a": a,
                    "regime_b": b,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "effective_n": min(
                        metadata[a]["n_observations"], metadata[b]["n_observations"]
                    ),
                }
            )
    pd.DataFrame(ci_rows).to_csv(output_dir / "bootstrap_cis.csv", index=False)

    logger.info("rendering figures")
    for stem, fig in [
        ("shap_global_importance", plot_global_importance(global_imp)),
        ("shap_per_regime_importance", plot_per_regime_importance(importance)),
        ("shap_stability_heatmap", plot_stability_heatmap(matrix, small_regimes)),
    ]:
        base = figures_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=300)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)

    logger.success("SHAP analysis complete: {} and {}", output_dir, figures_dir)
    return {
        "global": global_df,
        "rankings": rankings,
        "importance": importance,
        "matrix": matrix,
        "pairs": pd.DataFrame(pair_rows),
        "cis": pd.DataFrame(ci_rows),
        "small_regimes": small_regimes,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the regime-aware SHAP analysis end to end.")
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="Bootstrap iterations for small-regime CIs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed.")
    args = parser.parse_args(argv)
    configure_logging()
    run_shap_analysis(n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
