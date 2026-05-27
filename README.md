# Forecasting UK GDP Growth Using Interpretable Gradient Boosting and Traditional Econometric Methods

**Regime-Aware SHAP Stability Analysis Across the GFC, Brexit, and COVID-19 Structural Breaks (2000–2025)**

MSc Data Science — University of the West of England (UWE Bristol), 2026

---

## About

This project investigates whether machine learning model explanations can be trusted across economic structural breaks. Four forecasting models (ARIMA, Ridge Regression, XGBoost, LightGBM) are trained on UK quarterly GDP data spanning 2000–2025, covering six distinct economic regimes. SHAP feature importance is computed separately for each regime and explanation stability is quantified using Spearman rank correlation.

The key question: if SHAP attributes GDP growth to different variables depending on the economic regime, can those explanations be trusted for policy guidance during a crisis?

## Research Questions

- **RQ1:** How do four forecasting approaches compare in predicting UK quarterly GDP growth, and to what extent do accuracy gains stem from additional features versus non-linear modelling?
- **RQ2:** How does forecast accuracy vary across six economic regimes, and which models demonstrate greatest robustness during structural breaks?
- **RQ3:** Which macroeconomic features drive GDP predictions in each regime, and how do SHAP feature importance rankings shift between periods?
- **RQ4:** Can the temporal stability of SHAP explanations be quantified using rank correlation metrics, and what thresholds distinguish stable from unreliable attributions?
- **RQ5:** Can the best-performing model produce reliable GDP forecasts for 2026 when validated against actual ONS releases?

## Installation

```bash
git clone https://github.com/faithcodes-lab/uk-gdp-regime-forecasting.git
cd uk-gdp-regime-forecasting
python -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

```bash
make data       # Download and build the dataset
make train      # Train all four models
make evaluate   # Evaluate performance overall and per-regime
make shap       # Run SHAP stability analysis
make all        # Run the full pipeline
make test       # Run tests
make lint       # Run linting
```

## Project Structure

```
uk-gdp-regime-forecasting/
├── data/
│   ├── raw/                   # Downloaded source data (not tracked)
│   ├── processed/             # Analysis-ready dataset (built by pipeline)
│   └── scripts/               # Download and build scripts
├── src/
│   ├── features/              # Feature engineering and structural break tests
│   ├── models/                # ARIMA, Ridge, XGBoost, LightGBM
│   ├── evaluation/            # Metrics, per-regime evaluation, DM test
│   └── explainability/        # SHAP analysis, regime-aware SHAP, stability metrics
├── notebooks/                 # EDA and analysis notebooks
├── configs/                   # Regime dates, model parameters
├── results/
│   ├── tables/                # Model comparison tables
│   └── figures/               # Publication-quality plots
├── dashboard/                 # Interactive Plotly dashboard
├── tests/                     # pytest test suite
├── report/                    # Dissertation chapter drafts
├── requirements.txt
├── Makefile
├── pyproject.toml
├── CITATION.cff
└── .github/workflows/ci.yml   # Automated testing on every push
```

## Economic Regimes

| Regime | Period | Event |
|--------|--------|-------|
| Pre-GFC stability | 2000 Q1 – 2007 Q4 | Great Moderation |
| Global Financial Crisis | 2008 Q1 – 2009 Q4 | Credit crunch, banking crisis |
| Pre-Brexit recovery | 2010 Q1 – 2016 Q2 | Austerity, QE |
| Brexit transition | 2016 Q3 – 2019 Q4 | EU referendum, uncertainty |
| COVID-19 shock | 2020 Q1 – 2021 Q2 | Pandemic collapse |
| Post-COVID recovery | 2021 Q3 – 2025 Q4 | Inflation, cost-of-living |

## Data Sources

- **ONS** — GDP quarterly growth, CPI, unemployment, house prices
- **Bank of England** — Bank Rate, gilt yields, M4, exchange rates
- **OECD** — US and Eurozone GDP
- **S&P Global** — PMI (composite, manufacturing, services)
- **GfK** — Consumer confidence
- **Yahoo Finance** — FTSE 100
- **FRED** — Brent crude oil price

## Related: regime-shap

The SHAP stability analysis code from this project is being extracted into a standalone, pip-installable package: [`regime-shap`](https://github.com/faithcodes-lab/regime-shap) *(coming soon)*

## Licence

MIT — see [LICENSE](LICENSE)

## Citation

If you use this work, please cite:

```
Olan-George, F. (2026). Forecasting UK GDP Growth Using Interpretable Gradient Boosting
and Traditional Econometric Methods: Regime-Aware SHAP Stability Analysis Across the GFC,
Brexit, and COVID-19 Structural Breaks (2000–2025). MSc Dissertation, UWE Bristol.
```
