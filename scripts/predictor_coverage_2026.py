"""Per-predictor coverage report for 2026 Q1 and Q2.

Checks how many raw observations fall inside each 2026 quarter against what
the series's frequency expects, since a naive quarterly mean would silently
average an incomplete quarter into a plausible-looking number. Read-only,
never touches data/processed/final_dataset.parquet.

Run with
    PYTHONPATH=. python scripts/predictor_coverage_2026.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.build_dataset import load_raw_from_disk
from src.data.config import pipeline_config
from src.logging_setup import configure_logging

_EXPECTED_OBS_PER_QUARTER = {
    "quarterly": 1,
    "monthly": 3,
    "daily": None,  # variable count; not graded against a fixed expectation
}

_QUARTERS = {
    "2026 Q1": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
    "2026 Q2": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30")),
}


def _status(n_obs: int, expected: int | None) -> str:
    if n_obs == 0:
        return "MISSING"
    if expected is None:
        return f"present (n={n_obs}, daily series)"
    if n_obs >= expected:
        return "complete"
    return f"PARTIAL ({n_obs}/{expected} expected obs)"


def build_coverage_rows() -> list[dict]:
    raw = load_raw_from_disk()
    sources_cfg = pipeline_config()["data_sources"]
    rows: list[dict] = []
    for source, series_dict in raw.items():
        src_cfg = sources_cfg[source]["series"]
        for series_name, df in series_dict.items():
            freq = src_cfg[series_name]["frequency"]
            expected = _EXPECTED_OBS_PER_QUARTER.get(freq)
            for quarter_label, (start, end) in _QUARTERS.items():
                mask = (df["date"] >= start) & (df["date"] <= end)
                n_obs = int(mask.sum())
                last_date = df.loc[mask, "date"].max() if n_obs else pd.NaT
                last_value = df.loc[mask, "value"].iloc[-1] if n_obs else None
                rows.append(
                    {
                        "source": source,
                        "predictor": series_name,
                        "frequency": freq,
                        "quarter": quarter_label,
                        "n_obs": n_obs,
                        "last_obs_date": last_date,
                        "last_value": last_value,
                        "status": _status(n_obs, expected),
                    }
                )
    return rows


def _yield_curve_slope_note(raw: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Reports yield_curve_slope (10Y minus 2Y) since it is a derived predictor, not a raw_predictors_kept column."""
    gilt10 = raw["boe_yc"]["gilt_10y_yield"]
    gilt2 = raw["boe_yc"]["gilt_2y_yield"]
    lines = []
    for quarter_label, (start, end) in _QUARTERS.items():
        m10 = (gilt10["date"] >= start) & (gilt10["date"] <= end)
        m2 = (gilt2["date"] >= start) & (gilt2["date"] <= end)
        if m10.sum() and m2.sum():
            v10 = gilt10.loc[m10, "value"].iloc[-1]
            v2 = gilt2.loc[m2, "value"].iloc[-1]
            lines.append(
                f"- **{quarter_label}**: yield_curve_slope = {v10 - v2:.3f} "
                f"(10Y={v10:.3f}, 2Y={v2:.3f}, both present)"
            )
        else:
            lines.append(
                f"- **{quarter_label}**: yield_curve_slope MISSING (10Y or 2Y not present)"
            )
    return lines


def _gdp_vintage_drift_note() -> list[str]:
    """Flags any ONS revisions between the frozen training vintage and the freshly re-downloaded raw series."""
    repo_root = Path(__file__).resolve().parents[1]
    frozen = pd.read_parquet(repo_root / "data" / "processed" / "final_dataset.parquet")
    raw = pd.read_csv(repo_root / "data" / "raw" / "ons" / "gdp_growth.csv", parse_dates=["date"])
    merged = frozen[["date", "gdp_growth"]].merge(
        raw.rename(columns={"value": "gdp_growth_raw_now"}), on="date", how="inner"
    )
    drift = merged.loc[merged["gdp_growth"] != merged["gdp_growth_raw_now"]]
    lines = ["### GDP vintage drift (frozen training set vs. freshly re-downloaded raw)", ""]
    if drift.empty:
        lines.append(
            "No drift detected between the frozen training vintage and the current raw download."
        )
    else:
        lines.append(
            "ONS has revised the following quarters since the frozen 104-quarter training "
            "set was built. data/processed/final_dataset.parquet is untouched and still "
            "reflects the original vintage, which is correct, the training set stays frozen. "
            "Flagging this because any lag feature built for the 2026 forecast (gdp_lag_1, "
            "gdp_lag_4) must be sourced from the frozen vintage below, not the revised raw "
            "CSV, to stay consistent with what XGBoost was trained on."
        )
        lines.append("")
        lines.append("| Quarter | Frozen (training vintage) | Current raw (revised) |")
        lines.append("|---|---|---|")
        for _, r in drift.iterrows():
            lines.append(f"| {r['date'].date()} | {r['gdp_growth']} | {r['gdp_growth_raw_now']} |")
    lines.append("")
    latest_raw = raw.loc[raw["date"] == pd.Timestamp("2026-03-31"), "value"]
    if not latest_raw.empty:
        lines.append(
            f"2026 Q1 actual now present in the refreshed raw series: {latest_raw.iloc[0]}% "
            "(IHYQ / QNA vintage, real, CVM, SA, QoQ, see checkpoint 2 for which release this is)."
        )
    else:
        lines.append("2026 Q1 actual not yet present in the raw series.")
    return lines


def render_markdown(rows: list[dict], raw: dict[str, dict[str, pd.DataFrame]]) -> str:
    lines = [
        "# 2026 predictor coverage report",
        "",
        "Checkpoint 1 of the 2026 forecast validation epic. Generated from raw CSVs "
        "refreshed via make download after fixing the hardcoded BoE Dateto=31/Dec/2025 "
        "cutoff in config/pipeline.yaml. Read-only: data/processed/final_dataset.parquet "
        "(the frozen 2000-2025, 104-quarter training set) is untouched by this report.",
        "",
        "## Raw predictor coverage by quarter",
        "",
        "| Predictor | Source | Frequency | Quarter | Obs count | Last obs date | Last value | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        last_date = r["last_obs_date"].date() if pd.notna(r["last_obs_date"]) else "-"
        last_value = f"{r['last_value']:.4g}" if r["last_value"] is not None else "-"
        lines.append(
            f"| {r['predictor']} | {r['source']} | {r['frequency']} | {r['quarter']} | "
            f"{r['n_obs']} | {last_date} | {last_value} | {r['status']} |"
        )

    lines += ["", "## Derived: yield curve slope (10Y minus 2Y)", ""]
    lines += _yield_curve_slope_note(raw)

    lines += ["", "## GDP target vintage check", ""]
    lines += _gdp_vintage_drift_note()

    lines += ["", "## Summary", ""]
    q1_rows = [r for r in rows if r["quarter"] == "2026 Q1"]
    q2_rows = [r for r in rows if r["quarter"] == "2026 Q2"]
    q1_missing = [r["predictor"] for r in q1_rows if r["status"] == "MISSING"]
    q1_partial = [r["predictor"] for r in q1_rows if r["status"].startswith("PARTIAL")]
    q2_missing = [r["predictor"] for r in q2_rows if r["status"] == "MISSING"]
    q2_partial = [r["predictor"] for r in q2_rows if r["status"].startswith("PARTIAL")]
    lines.append(
        f"- **2026 Q1**: {len(q1_rows) - len(q1_missing) - len(q1_partial)}/{len(q1_rows)} "
        f"predictors complete."
        + (f" Missing: {', '.join(q1_missing)}." if q1_missing else "")
        + (f" Partial: {', '.join(q1_partial)}." if q1_partial else "")
    )
    lines.append(
        f"- **2026 Q2**: {len(q2_rows) - len(q2_missing) - len(q2_partial)}/{len(q2_rows)} "
        f"predictors complete."
        + (f" Missing: {', '.join(q2_missing)}." if q2_missing else "")
        + (f" Partial: {', '.join(q2_partial)}." if q2_partial else "")
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_logging()
    raw = load_raw_from_disk()
    rows = build_coverage_rows()
    report = render_markdown(rows, raw)
    out_path = Path(__file__).resolve().parents[1] / "results" / "2026-predictor-coverage.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("Wrote coverage report: {}", out_path)


if __name__ == "__main__":
    main()
