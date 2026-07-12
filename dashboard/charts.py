"""Data loading and Plotly figures for the GDP results dashboard.

Pure functions only, no Streamlit, so they can be tested on their own. Every
figure reads the precomputed result files bundled under ``data/``; nothing here
runs a model or touches the frozen raw dataset.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

DATA = Path(__file__).parent / "data"

MODEL_ORDER = ["arima", "ridge", "lightgbm", "xgboost"]
MODEL_COLOURS = {
    "arima": "#ff7f0e",
    "ridge": "#1f77b4",
    "lightgbm": "#2ca02c",
    "xgboost": "#d62728",
}
REGIME_ORDER = [
    "Pre-GFC Stability",
    "Global Financial Crisis",
    "Post-GFC Recovery",
    "Brexit Transition",
    "COVID-19 Shock",
    "Post-COVID Recovery",
]
METRICS = {"RMSE": "rmse", "MAE": "mae", "MASE": "mase", "R2": "r2"}
# Akoglu (2018) band colours, reused for the stability heatmap.
BAND_COLOURS = {"unstable": "#d73027", "moderately stable": "#fdae61", "stable": "#1a9850"}


@functools.lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    """Load a bundled result file (csv or parquet) by file name."""
    path = DATA / name
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


@functools.lru_cache(maxsize=1)
def regimes() -> list[dict]:
    """The six regime definitions (label, start, end) from the bundled config."""
    with (DATA / "regimes.yaml").open() as handle:
        return yaml.safe_load(handle)["regimes"]


# ---- section 1: model comparison ----
def fig_aggregated(scheme: str, metric_label: str) -> go.Figure:
    metric = METRICS[metric_label]
    df = load("aggregated.csv")
    df = df[df["scheme"] == scheme].set_index("model").reindex(MODEL_ORDER).reset_index()
    fig = go.Figure(
        go.Bar(
            x=df["model"],
            y=df[f"mean_{metric}"],
            error_y=dict(type="data", array=df[f"std_{metric}"]),
            marker_color=[MODEL_COLOURS[m] for m in df["model"]],
        )
    )
    fig.update_layout(
        title=f"Mean {metric_label} by model (error bars are 1 SD across folds)",
        yaxis_title=metric_label,
        xaxis_title=None,
    )
    return fig


def fig_perfold_box(scheme: str, metric_label: str) -> go.Figure:
    metric = METRICS[metric_label]
    df = load("per_fold.csv")
    df = df[df["scheme"] == scheme]
    fig = px.box(
        df, x="model", y=metric, color="model",
        category_orders={"model": MODEL_ORDER}, color_discrete_map=MODEL_COLOURS, points="all",
    )
    fig.update_layout(
        title=f"{metric_label} spread across folds", yaxis_title=metric_label,
        xaxis_title=None, showlegend=False,
    )
    return fig


def fig_dm_matrix(scheme: str) -> go.Figure:
    df = load("dm_test.csv")
    df = df[df["scheme"] == scheme]
    mat = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for _, r in df.iterrows():
        mat.loc[r["model_a"], r["model_b"]] = r["p_value_bonferroni"]
        mat.loc[r["model_b"], r["model_a"]] = r["p_value_bonferroni"]
    fig = px.imshow(
        mat.astype(float), text_auto=".2f", color_continuous_scale="Blues_r",
        zmin=0, zmax=1, aspect="auto",
    )
    fig.update_layout(
        title="Diebold-Mariano test, Bonferroni-adjusted p-values (lower means a "
        "more significant difference)",
        coloraxis_colorbar_title="p",
    )
    return fig


def fig_predictions(scheme: str, models: list[str]) -> go.Figure:
    df = load("predictions.parquet")
    df = df[df["scheme"] == scheme].sort_values("quarter")
    fig = go.Figure()
    actual = df.drop_duplicates("quarter")
    fig.add_trace(
        go.Scatter(x=actual["quarter"], y=actual["y_true"], name="actual",
                   mode="lines+markers", line=dict(color="black", width=2.5),
                   marker=dict(size=5))
    )
    for m in models:
        d = df[df["model"] == m]
        fig.add_trace(
            go.Scatter(x=d["quarter"], y=d["y_pred"], name=m, mode="lines",
                       line=dict(color=MODEL_COLOURS.get(m)))
        )
    fig.update_layout(
        title="Predicted vs actual GDP growth over the test quarters",
        yaxis_title="GDP growth (%)", xaxis_title=None,
    )
    return fig


# ---- section 2: regime evaluation ----
def fig_regime_timeline() -> go.Figure:
    rows = [
        {"regime": r["label"], "start": r["start"], "end": r["end"], "quarters": r["quarters"]}
        for r in regimes()
    ]
    tl = pd.DataFrame(rows)
    fig = px.timeline(
        tl, x_start="start", x_end="end", y="regime", color="regime",
        category_orders={"regime": REGIME_ORDER[::-1]}, hover_data=["quarters"],
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(title="The six economic regimes over time", showlegend=False)
    return fig


def fig_per_regime(scheme: str, metric_label: str) -> go.Figure:
    metric = METRICS[metric_label]
    df = load("per_regime.csv")
    df = df[df["scheme"] == scheme]
    fig = px.bar(
        df, x="regime", y=metric, color="model", barmode="group",
        category_orders={"regime": REGIME_ORDER, "model": MODEL_ORDER},
        color_discrete_map=MODEL_COLOURS,
    )
    fig.update_layout(
        title=f"{metric_label} within each regime, by model", yaxis_title=metric_label,
        xaxis_title=None,
    )
    fig.update_xaxes(tickangle=25)
    return fig


# ---- section 3: SHAP explanations ----
def fig_global_importance() -> go.Figure:
    df = load("global_importance.csv").sort_values("mean_abs_shap")
    fig = go.Figure(
        go.Bar(x=df["mean_abs_shap"], y=df["feature"], orientation="h", marker_color="#1f4e79")
    )
    fig.update_layout(
        title="Global SHAP importance (mean absolute SHAP per feature)",
        xaxis_title="mean |SHAP|", yaxis_title=None, height=520,
    )
    return fig


def fig_per_regime_importance() -> go.Figure:
    df = load("per_regime_importance.csv").rename(columns={"Unnamed: 0": "feature"})
    df = df.set_index("feature")[REGIME_ORDER]
    fig = px.imshow(df, aspect="auto", color_continuous_scale="Viridis", text_auto=".2f")
    fig.update_layout(
        title="Per-regime SHAP importance (mean |SHAP| by feature and regime)",
        coloraxis_colorbar_title="mean |SHAP|", height=560,
    )
    return fig


# ---- section 4: stability matrix ----
def fig_stability_heatmap() -> go.Figure:
    df = load("stability_matrix.csv").rename(columns={"Unnamed: 0": "regime"})
    df = df.set_index("regime").reindex(index=REGIME_ORDER, columns=REGIME_ORDER)
    fig = px.imshow(
        df.astype(float), text_auto=".2f", zmin=0, zmax=1, aspect="auto",
        color_continuous_scale=[(0.0, BAND_COLOURS["unstable"]),
                                (0.3, BAND_COLOURS["unstable"]),
                                (0.3, BAND_COLOURS["moderately stable"]),
                                (0.6, BAND_COLOURS["moderately stable"]),
                                (0.6, BAND_COLOURS["stable"]),
                                (1.0, BAND_COLOURS["stable"])],
    )
    fig.update_layout(
        title="Cross-regime SHAP stability (Spearman rho of importance rankings)",
        coloraxis_colorbar_title="rho", height=560,
    )
    fig.update_xaxes(tickangle=25)
    return fig
