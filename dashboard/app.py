"""GDP results dashboard (Streamlit + Plotly).

Surfaces the UK GDP regime-forecasting results: model comparison, regime
evaluation, SHAP explanations, and the SHAP stability matrix. Everything is read
from the precomputed result files under ``data/``; no model runs here and the
frozen raw dataset is not loaded. The SHAP and stability results are shown as
they are, including the two-feature concentration and the near-1.000 stability.
"""

from __future__ import annotations

import streamlit as st

import charts as c

st.set_page_config(page_title="UK GDP regime forecasting: results", layout="wide")

st.title("UK GDP regime forecasting: results")
st.caption(
    "Model comparison, regime evaluation, and SHAP stability for a quarterly UK GDP "
    "growth forecast across six economic regimes. All figures are read from precomputed "
    "results; nothing is refit here."
)

with st.sidebar:
    st.header("Controls")
    scheme = st.radio(
        "Cross-validation scheme",
        ["expanding_window", "regime_aligned"],
        help="Expanding window is the primary scheme; regime aligned tests one regime at a time.",
    )
    metric_label = st.selectbox("Metric", list(c.METRICS), index=0)
    st.markdown(
        "Source: [github.com/faithcodes-lab/uk-gdp-regime-forecasting]"
        "(https://github.com/faithcodes-lab/uk-gdp-regime-forecasting)"
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Model comparison", "Regime evaluation", "SHAP explanations", "Stability matrix"]
)

with tab1:
    st.subheader("Model comparison")
    col1, col2 = st.columns(2)
    col1.plotly_chart(c.fig_aggregated(scheme, metric_label), width="stretch")
    col2.plotly_chart(c.fig_perfold_box(scheme, metric_label), width="stretch")
    st.plotly_chart(c.fig_dm_matrix(scheme), width="stretch")
    st.caption(
        "The Diebold-Mariano test asks whether two models' forecast errors differ "
        "significantly. After the Bonferroni correction for multiple comparisons, the "
        "differences are not significant, which is consistent with a small quarterly "
        "sample where models are hard to separate."
    )
    models = st.multiselect(
        "Models to overlay", c.MODEL_ORDER, default=c.MODEL_ORDER
    )
    st.plotly_chart(c.fig_predictions(scheme, models), width="stretch")

with tab2:
    st.subheader("Regime evaluation")
    st.plotly_chart(c.fig_regime_timeline(), width="stretch")
    st.plotly_chart(c.fig_per_regime(scheme, metric_label), width="stretch")
    st.caption(
        "Three regimes are short (the Global Financial Crisis and COVID-19 Shock have six "
        "quarters each, Brexit has fourteen), so their per-regime scores are noisier and "
        "should be read with that in mind. The COVID-19 Shock, with its extreme values, "
        "dominates the error in every model."
    )

with tab3:
    st.subheader("SHAP explanations")
    st.markdown(
        "**Interpretation.** The model's importance concentrates almost entirely on two "
        "features, gdp_growth and gdp_lag_4, and the macroeconomic predictors receive "
        "near-zero SHAP importance. This reflects the limited predictive signal in the "
        "macro block at a quarterly sample of this size. Standard dimension-reduction and "
        "shrinkage methods did not recover out-of-sample macro signal, so it is treated as "
        "a finding and developed in the limitations."
    )
    st.plotly_chart(c.fig_global_importance(), width="stretch")
    st.plotly_chart(c.fig_per_regime_importance(), width="stretch")

with tab4:
    st.subheader("Stability matrix")
    st.markdown(
        "**Interpretation.** The stability matrix is 1.000 across every regime pair. This "
        "reflects the two-feature concentration shown above: the same two features drive the "
        "model in every regime, so the importance rankings are identical and their Spearman "
        "correlations are perfect. Rather than indicating rich cross-regime stability, this "
        "perfect stability reveals the model's reliance on a minimal feature set, which is "
        "itself the key finding: a model can appear perfectly stable while depending on very "
        "few features."
    )
    st.plotly_chart(c.fig_stability_heatmap(), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regime pairs and stability bands**")
        st.dataframe(c.load("stability_pairs.csv"), width="stretch", hide_index=True)
    with col2:
        st.markdown("**Bootstrap confidence intervals (small-regime pairs)**")
        st.dataframe(c.load("bootstrap_cis.csv"), width="stretch", hide_index=True)
