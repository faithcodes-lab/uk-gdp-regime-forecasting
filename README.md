# UK GDP Regime Forecasting

> **MSc dissertation research repository.** Forecasting UK GDP Growth Using Interpretable Gradient Boosting: Regime-Aware SHAP Analysis Across the Global Financial Crisis (GFC), Brexit and COVID-19 Structural Breaks (2000–2025).

[![CI](https://github.com/faithcodes-lab/uk-gdp-regime-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/faithcodes-lab/uk-gdp-regime-forecasting/actions/workflows/ci.yml)
[![Coverage: 82%](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](https://github.com/faithcodes-lab/uk-gdp-regime-forecasting/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Faith Olan-George
**Programme:** MSc Data Science, University of the West of England (UWE Bristol)
**Module:** UFCF9Y-60-M (CSCT Masters Project)

## Overview

Four models (ARIMA, Ridge, XGBoost, LightGBM) are trained on 104 quarters of UK GDP growth across six economic regimes spanning the Global Financial Crisis, Brexit and COVID-19. The trained models are the basis for this project's actual question: whether a model's SHAP feature-importance explanations stay stable as the economy moves through structural breaks, because policymakers need explanations that hold up across changing conditions, not just calm ones.

## Key findings

- Across expanding-window and regime-aligned cross-validation, no model reliably beats the others; pairwise Diebold-Mariano differences do not survive Bonferroni correction.
- Every model's error concentrates in the crisis regimes: XGBoost's RMSE in the COVID-19 Shock regime (~11.2) is roughly 20–30x its RMSE in calmer regimes.
- SHAP attribution on the trained XGBoost model collapses onto two features, lagged GDP growth (~82%) and four-quarter-lagged GDP growth (~18%); the ten macroeconomic predictors receive exactly zero attribution.
- Regime-pair SHAP rankings score a perfect Spearman stability of 1.0 across all fifteen comparisons, but this is a side effect of the model relying on only two features, not evidence of genuine robustness. This is confirmed independently four ways: feature ablation, a LASSO comparison, a model-capacity sweep, and a rolling-window check.
- The trained model was used to forecast 2026 Q1 and Q2 UK GDP growth out-of-sample, validated against the actual ONS releases as they became available.

Full methodology, results, and discussion are in the dissertation report (submitted separately, not included in this repository).

## Related work

The regime-aware SHAP stability methodology developed here has been extracted into a standalone, general-purpose Python package:

- **[regime-shap](https://github.com/faithcodes-lab/regime-shap)** - `pip install regime-shap`. [PyPI](https://pypi.org/project/regime-shap/) · [Docs](https://faithcodes-lab.github.io/regime-shap/)

An interactive Plotly dashboard covering model comparison, regime evaluation, SHAP explanations and the stability matrix for this project's GDP results is also live: **[uk-gdp-results-dashboard](https://huggingface.co/spaces/FaithCodes/uk-gdp-results-dashboard)** on Hugging Face Spaces.

## Setup

```bash
# Clone
git clone https://github.com/faithcodes-lab/uk-gdp-regime-forecasting.git
cd uk-gdp-regime-forecasting

# Set up Python environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Edit .env with your actual values
```

## Repository Structure

<pre>
uk-gdp-regime-forecasting/
├── <a href="config/">config/</a>                   # Configuration files (data sources, models)
├── <a href="data/">data/</a>                     # Data lifecycle (raw, interim, processed)
├── <a href="src/">src/</a>                      # Source code (data, models, evaluation, explainability)
├── <a href="tests/">tests/</a>                    # Pytest test suite
├── <a href="scripts/">scripts/</a>                  # Standalone scripts (master pipelines, utilities)
├── <a href="results/">results/</a>                  # Pipeline outputs (models, figures, tables, SHAP)
├── <a href="dashboard/">dashboard/</a>                # Streamlit + Plotly results dashboard
└── <a href=".github/">.github/</a>                 # CI workflows
</pre>

## Quick Start

```bash
make setup    # create environment and install dependencies
make data     # run the data pipeline
make train    # train all models
make evaluate # produce results tables and figures
```

## Tests

```bash
make test
```

CI runs the full test suite, linting, and format checks on every push (Python 3.11).

## Licence

Code: MIT (see `LICENSE`).
Dissertation text and figures: Copyright Faith Olan-George 2026, all rights reserved.

## Citation

If you use this work, please cite:

```
Olan-George, F. (2026). Forecasting UK GDP Growth Using Interpretable Gradient
Boosting: Regime-Aware SHAP Analysis Across the Global Financial Crisis (GFC),
Brexit and COVID-19 Structural Breaks. MSc Dissertation, University of the
West of England, Bristol.
```
