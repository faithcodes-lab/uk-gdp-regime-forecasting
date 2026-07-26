---
title: UK GDP Regime Forecasting Results
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Model, regime, and SHAP stability results for UK GDP
---

# UK GDP regime forecasting: results dashboard

An interactive Streamlit and Plotly dashboard for the UK GDP regime-forecasting dissertation. It
presents four things across the six economic regimes:

1. Model comparison (four models, two cross-validation schemes, Diebold-Mariano tests).
2. Regime evaluation (per-regime performance and the regime timeline).
3. SHAP explanations (global and per-regime feature importance).
4. The SHAP stability matrix.

Every figure is read from precomputed result files bundled under `data/`. No model is run here and
the frozen raw dataset is not loaded.

The SHAP and stability results are shown honestly. The model concentrates on two features
(`gdp_growth` and `gdp_lag_4`), so the stability matrix is 1.000 everywhere. This is a genuine
finding at this sample size, not a display artefact, and it is reported as such.
