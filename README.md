# UK GDP Regime Forecasting

> **MSc dissertation research repository.** Forecasting UK GDP Growth Using Interpretable Gradient Boosting: Regime-Aware SHAP Analysis Across Brexit and COVID-19 Structural Breaks (2000–2025).

**Author:** Faith Olan-George
**Programme:** MSc Data Science, University of the West of England (UWE Bristol)
**Module:** UFCF9Y-60-M (CSCT Masters Project)
**Student ID:** 25047901
**Submission:** 3 September 2026

## Repository Status

This repository is under active development.

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

```
uk-gdp-regime-forecasting/
├── config/                   # Configuration files (data sources, models)
├── data/                     # Data lifecycle (raw → interim → processed)
├── src/                      # Source code (data, models, evaluation, explainability)
├── tests/                    # Pytest test suite
├── notebooks/                # Exploratory notebooks
├── scripts/                  # Standalone scripts (master pipelines, utilities)
├── results/                  # Pipeline outputs (models, figures, tables, SHAP)
├── report/                   # Dissertation report (chapters, appendices, bibliography)
├── paper/                    # Research paper (LaTeX, figures)
└── .github/                  # CI workflows
```

## Quick Start

```bash
  make setup    # create environment and install dependencies
  make data     # run the data pipeline
  make train    # train all models
  make evaluate # produce results tables and figures
```

## Licence

Code: MIT (see `LICENSE`).
Dissertation text and figures: Copyright Faith Olan-George 2026, all rights reserved.

## Citation

If you use this work, please cite:

```
Olan-George, F. (2026). Forecasting UK GDP Growth Using Interpretable Gradient
Boosting: Regime-Aware SHAP Analysis Across Brexit and COVID-19 Structural
Breaks. MSc Dissertation, University of the West of England, Bristol.
```