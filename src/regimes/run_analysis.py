"""Run Chow tests + Bai-Perron sweep on the live dataset; persist results.

What this script does
---------------------
- Loads ``data/processed/final_dataset.parquet``.
- For each of the five regime boundaries (start dates of regimes 2..6
  from ``config/regimes.yaml``), runs a Chow test with the feature set
  ``[gdp_lag_1, unemployment_rate, cpi_inflation, bank_rate]``.
  :class:`InsufficientObservationsError` is caught and recorded as a
  skip with the underlying error message rather than crashing the run.
- Runs the PELT Bai-Perron sensitivity sweep on ``gdp_growth`` over the
  default penalty grid ``[5, 10, 15, 20, 30]``. If every penalty in the
  default grid produces the same number of breaks (the grid is
  uninformative), the script logs a warning and re-runs with the wider
  grid ``[1, 3, 5, 10, 15, 20, 30, 50, 100]``; the wider grid's results
  are then the ones written to disk.
- Runs the ICSS variance break test (Inclan-Tiao 1994) on
  ``gdp_growth`` to look for variance shifts. The result is annotated
  with a GFC-validation block: any detected break whose date falls
  inside the GFC regime window (2008 Q2 to 2009 Q3 inclusive) is
  reported with its distance, in quarters, from the GFC start.
- Writes four artefacts into ``results/regimes/``:

    * ``chow_test_results.json`` (one entry per boundary)
    * ``bai_perron_sweep.json``  (full per-penalty break dates)
    * ``bai_perron_sensitivity.csv`` (direct dump of ``tune_penalty``)
    * ``icss_results.json`` (variance break test + GFC evidence)

- Prints a short human-readable summary via loguru.

Run with
--------
    make break-tests
or  PYTHONPATH=. python -m src.regimes.run_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data.config import pipeline_config, regimes_config
from src.logging_setup import configure_logging
from src.regimes.bai_perron import detect_breaks_pelt, tune_penalty
from src.regimes.chow import InsufficientObservationsError, chow_test
from src.regimes.volatility import icss_test

_CHOW_FEATURES = ["gdp_lag_1", "unemployment_rate", "cpi_inflation", "bank_rate"]
_TARGET = "gdp_growth"
_DEFAULT_PENALTIES = [5.0, 10.0, 15.0, 20.0, 30.0]
_WIDE_PENALTIES = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]
_SUMMARY_PENALTY = 10.0
_CONVERGENCE_TOLERANCE_QUARTERS = 2
_ICSS_ALPHA = 0.05
_GFC_REGIME_LABEL = "Global Financial Crisis"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dataset() -> pd.DataFrame:
    parquet_path = _repo_root() / pipeline_config()["paths"]["final_dataset"]
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _boundaries() -> list[dict]:
    """Five Chow-test boundaries: the start of each regime after the first."""
    regimes = regimes_config()["regimes"]
    return [{"label": r["label"], "date": pd.Timestamp(r["start"])} for r in regimes[1:]]


def _run_chow_tests(df: pd.DataFrame) -> list[dict]:
    """Run the Chow test at each boundary and return a serialisable list."""
    X = df[_CHOW_FEATURES]
    y = df[_TARGET]
    dates = df["date"]

    results: list[dict] = []
    for boundary in _boundaries():
        label = boundary["label"]
        date = boundary["date"]
        try:
            result = chow_test(X, y, dates, breakpoint_date=date)
            results.append(
                {
                    "boundary_label": label,
                    "breakpoint_date": date.strftime("%Y-%m-%d"),
                    "status": "ok",
                    "f_statistic": result.f_statistic,
                    "p_value": result.p_value,
                    "rss_pooled": result.rss_pooled,
                    "rss_pre": result.rss_pre,
                    "rss_post": result.rss_post,
                    "n_total": result.n_total,
                    "n_pre": result.n_pre,
                    "n_post": result.n_post,
                    "k_params": result.k_params,
                    "df_numerator": result.df_numerator,
                    "df_denominator": result.df_denominator,
                    "significant_at_5pct": bool(result.p_value < 0.05),
                }
            )
        except InsufficientObservationsError as exc:
            logger.warning("Skipping Chow at {} ({}): {}", date.date(), label, exc)
            results.append(
                {
                    "boundary_label": label,
                    "breakpoint_date": date.strftime("%Y-%m-%d"),
                    "status": "skipped",
                    "skip_reason": str(exc),
                }
            )
    return results


def _run_bai_perron_sweep(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, list[float]]:
    """Run the sensitivity sweep, auto-widening if the default grid is flat.

    Returns:
        ``(sensitivity_df, sweep_payload, penalties_used)``. The
        sensitivity DataFrame and the JSON payload reflect whichever
        grid was actually used.
    """
    series = df.set_index("date")[_TARGET]

    sensitivity_df = tune_penalty(series, penalties=list(_DEFAULT_PENALTIES))
    penalties_used = list(_DEFAULT_PENALTIES)
    if sensitivity_df["n_breaks"].nunique() == 1:
        logger.warning(
            "Default penalty grid {} gave a flat sweep ({} breaks at every "
            "penalty); re-running with wider grid {}",
            _DEFAULT_PENALTIES,
            int(sensitivity_df["n_breaks"].iloc[0]),
            _WIDE_PENALTIES,
        )
        sensitivity_df = tune_penalty(series, penalties=list(_WIDE_PENALTIES))
        penalties_used = list(_WIDE_PENALTIES)

    results_by_penalty = []
    for p in penalties_used:
        breaks = detect_breaks_pelt(series, penalty=p)
        results_by_penalty.append(
            {
                "penalty": float(p),
                "n_breaks": len(breaks),
                "breakpoint_dates": [b.strftime("%Y-%m-%d") for b in breaks],
            }
        )

    sweep_payload = {
        "series": _TARGET,
        "model": "rbf",
        "penalty_grid": penalties_used,
        "results_by_penalty": results_by_penalty,
        "n_total_observations": len(series),
    }
    return sensitivity_df, sweep_payload, penalties_used


def _gfc_regime_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps of the GFC regime from regimes.yaml."""
    gfc = next(r for r in regimes_config()["regimes"] if r["label"] == _GFC_REGIME_LABEL)
    return pd.Timestamp(gfc["start"]), pd.Timestamp(gfc["end"])


def _run_icss(df: pd.DataFrame) -> dict:
    """Run the ICSS variance break test on gdp_growth; build the JSON payload.

    The payload includes a ``gfc_evidence`` block that lists every
    detected break falling inside the GFC regime window (2008 Q2 to
    2009 Q3 inclusive) and reports each break's distance, in quarters,
    from the GFC start. ``evidence_inside_window`` is True when at least one
    break falls inside the window.
    """
    series = df.set_index("date")[_TARGET]
    result = icss_test(series, alpha=_ICSS_ALPHA)

    gfc_start, gfc_end = _gfc_regime_window()
    gfc_start_q = gfc_start.to_period("Q")
    gfc_end_q = gfc_end.to_period("Q")

    breaks_inside: list[dict] = []
    for date, stat in zip(result.breakpoint_dates, result.d_statistics):
        date_q = date.to_period("Q")
        if gfc_start_q <= date_q <= gfc_end_q:
            breaks_inside.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "d_statistic": float(stat),
                    "quarters_from_gfc_start": int(date_q.ordinal - gfc_start_q.ordinal),
                }
            )

    return {
        "method": result.method,
        "series": result.series_name,
        "significance_level": result.significance_level,
        "critical_value": result.critical_value,
        "n_observations": result.n_observations,
        "n_breaks": result.n_breaks,
        "breakpoints": [
            {"date": d.strftime("%Y-%m-%d"), "d_statistic": float(s)}
            for d, s in zip(result.breakpoint_dates, result.d_statistics)
        ],
        "gfc_evidence": {
            "expected_boundary": gfc_start.strftime("%Y-%m-%d"),
            "gfc_regime_window": [
                gfc_start.strftime("%Y-%m-%d"),
                gfc_end.strftime("%Y-%m-%d"),
            ],
            "breaks_inside_window": breaks_inside,
            "evidence_inside_window": bool(breaks_inside),
        },
    }


def _quarters_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    """Absolute number of quarters between two dates."""
    return abs(d1.to_period("Q").ordinal - d2.to_period("Q").ordinal)


def _print_summary(
    chow_results: list[dict],
    sweep_payload: dict,
    icss_payload: dict,
) -> None:
    """Print headline numbers, convergence, ICSS, and GFC validation."""
    n_ok = sum(1 for r in chow_results if r["status"] == "ok")
    n_significant = sum(1 for r in chow_results if r["status"] == "ok" and r["significant_at_5pct"])
    n_skipped = sum(1 for r in chow_results if r["status"] == "skipped")
    logger.info(
        "Chow tests: {} of {} boundaries significant at the 5% level " "({} run ok, {} skipped)",
        n_significant,
        len(chow_results),
        n_ok,
        n_skipped,
    )

    summary_row = next(
        (r for r in sweep_payload["results_by_penalty"] if r["penalty"] == _SUMMARY_PENALTY),
        None,
    )
    if summary_row is None:
        logger.info(
            "Bai-Perron: penalty={} not in the grid actually used; "
            "see results/regimes/bai_perron_sensitivity.csv for the full sweep.",
            _SUMMARY_PENALTY,
        )
        return

    if summary_row["n_breaks"] == 0:
        logger.info("Bai-Perron at penalty={}: 0 breaks detected.", _SUMMARY_PENALTY)
    else:
        logger.info(
            "Bai-Perron at penalty={}: {} break(s) detected: {}",
            _SUMMARY_PENALTY,
            summary_row["n_breaks"],
            ", ".join(summary_row["breakpoint_dates"]),
        )

    chow_dates = [pd.Timestamp(r["breakpoint_date"]) for r in chow_results if r["status"] == "ok"]
    bp_dates = [pd.Timestamp(d) for d in summary_row["breakpoint_dates"]]
    if not bp_dates:
        return

    agreements: list[tuple[str, str]] = []
    for chow_date in chow_dates:
        nearest = min(bp_dates, key=lambda d: _quarters_between(d, chow_date))
        if _quarters_between(nearest, chow_date) <= _CONVERGENCE_TOLERANCE_QUARTERS:
            agreements.append((chow_date.strftime("%Y-%m-%d"), nearest.strftime("%Y-%m-%d")))

    if agreements:
        logger.info(
            "Convergence (literature boundary within +/- {} quarters of a "
            "Bai-Perron break at penalty={}):",
            _CONVERGENCE_TOLERANCE_QUARTERS,
            _SUMMARY_PENALTY,
        )
        for chow_date, bp_date in agreements:
            logger.info("  literature {} <-> bai-perron {}", chow_date, bp_date)
    else:
        logger.info(
            "No literature boundary within +/- {} quarters of a Bai-Perron " "break at penalty={}.",
            _CONVERGENCE_TOLERANCE_QUARTERS,
            _SUMMARY_PENALTY,
        )

    logger.info(
        "ICSS variance break test: {} break(s) at alpha={}",
        icss_payload["n_breaks"],
        icss_payload["significance_level"],
    )
    for bp in icss_payload["breakpoints"]:
        logger.info("  {}  D={:.3f}", bp["date"], bp["d_statistic"])

    gfc = icss_payload["gfc_evidence"]
    window = gfc["gfc_regime_window"]
    if gfc["evidence_inside_window"]:
        logger.info(
            "GFC evidence: variance break(s) found inside the GFC window ({} to {}):",
            window[0],
            window[1],
        )
        for b in gfc["breaks_inside_window"]:
            logger.info(
                "  {}  (Q+{} from GFC start)",
                b["date"],
                b["quarters_from_gfc_start"],
            )
    else:
        logger.info(
            "GFC evidence: no variance break inside the GFC window ({} to {}).",
            window[0],
            window[1],
        )


def main() -> None:
    configure_logging()
    logger.info("Loading dataset and regime configuration")
    df = _load_dataset()

    out_dir = _repo_root() / "results" / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running Chow tests at {} boundaries", len(_boundaries()))
    chow_results = _run_chow_tests(df)
    chow_path = out_dir / "chow_test_results.json"
    chow_path.write_text(json.dumps(chow_results, indent=2))
    logger.info("Wrote {}", chow_path.relative_to(_repo_root()))

    logger.info("Running Bai-Perron sensitivity sweep")
    sensitivity_df, sweep_payload, _ = _run_bai_perron_sweep(df)
    sensitivity_path = out_dir / "bai_perron_sensitivity.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    sweep_path = out_dir / "bai_perron_sweep.json"
    sweep_path.write_text(json.dumps(sweep_payload, indent=2))
    logger.info("Wrote {}", sensitivity_path.relative_to(_repo_root()))
    logger.info("Wrote {}", sweep_path.relative_to(_repo_root()))

    logger.info("Running ICSS variance break test on gdp_growth")
    icss_payload = _run_icss(df)
    icss_path = out_dir / "icss_results.json"
    icss_path.write_text(json.dumps(icss_payload, indent=2))
    logger.info("Wrote {}", icss_path.relative_to(_repo_root()))

    _print_summary(chow_results, sweep_payload, icss_payload)


if __name__ == "__main__":
    main()
