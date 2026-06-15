"""Master pipeline orchestrator.

End-to-end process:
  1. download_all_{ons,boe,fred,boe_yc}      raw (date, value) frames in memory plus on disk
  2. to_quarterly(df, method=cfg.aggregation) per series
  3. merge every quarterly frame on date (outer)   preserves full history pre-2000
  4. engineer_features(merged)                     adds 6 engineered features plus drops gilts
  5. trim to project date_range (2000-01-01 to 2025-12-31)
  6. check_final_dataset                           quality plus look-ahead invariants
  7. FINAL_DATASET_SCHEMA.validate                 pandera shape plus range checks
  8. write data/processed/final_dataset.parquet

CLI:
  --download-only   stop after step 1 (for `make download`)
  --process-only    skip step 1; read raw CSVs from disk (for `make process`)
  (no flag)         full pipeline (for `make data`)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

from data.schemas.processed_quarterly import FINAL_DATASET_SCHEMA
from src.data.aggregate import to_quarterly
from src.data.config import pipeline_config
from src.data.downloaders import (
    download_all_boe,
    download_all_boe_yc,
    download_all_fred,
    download_all_ons,
)
from src.data.quality import check_final_dataset
from src.features.engineer import engineer_features
from src.logging_setup import configure_logging

# Iteration order is preserved: ons to boe to fred to boe_yc, then series order
# inside each source comes from the YAML. The final dataset's column order
# is determined by this iteration plus the engineer step's additions.
_DOWNLOADERS: dict[str, Callable[[], dict[str, pd.DataFrame]]] = {
    "ons": download_all_ons,
    "boe": download_all_boe,
    "fred": download_all_fred,
    "boe_yc": download_all_boe_yc,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_all() -> dict[str, dict[str, pd.DataFrame]]:
    """Run every per-source downloader."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for source, fn in _DOWNLOADERS.items():
        logger.info("Downloading source: {}", source)
        out[source] = fn()
    return out


def load_raw_from_disk() -> dict[str, dict[str, pd.DataFrame]]:
    """Load every series's raw CSV from ``data/raw/<source>/<series>.csv``.

    Requires a prior ``--download-only`` (or full) run to have populated disk.
    """
    sources_cfg = pipeline_config()["data_sources"]
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for source, src_cfg in sources_cfg.items():
        out[source] = {}
        for series_name in src_cfg["series"]:
            csv_path = _repo_root() / "data" / "raw" / \
                source / f"{series_name}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Raw CSV missing: {csv_path}. Run `make download` first.")
            df = pd.read_csv(csv_path, parse_dates=["date"])
            out[source][series_name] = df
    return out


def aggregate_to_quarterly(
    raw: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Aggregate every raw series to QE and merge them all on date."""
    sources_cfg = pipeline_config()["data_sources"]
    frames: list[pd.DataFrame] = []
    for source, series_dict in raw.items():
        src_cfg = sources_cfg[source]["series"]
        for series_name, raw_df in series_dict.items():
            method = src_cfg[series_name]["aggregation"]
            quarterly = to_quarterly(raw_df, method=method)
            # Per-series column rename so the merge produces named columns.
            quarterly = quarterly.rename(columns={"value": series_name})
            frames.append(quarterly)

    merged = frames[0]
    for frame in frames[1:]:
        # Outer join keeps every date any series has, so engineered features
        # near the start of the project window can use pre-2000 history.
        merged = merged.merge(frame, on="date", how="outer")
    return merged.sort_values("date").reset_index(drop=True)


def _filter_to_project_window(df: pd.DataFrame) -> pd.DataFrame:
    cfg = pipeline_config()["project"]["date_range"]
    start = pd.Timestamp(cfg["start"])
    end = pd.Timestamp(cfg["end"])
    return df.loc[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def build(*, from_disk: bool = False) -> Path:
    """Run the full pipeline and write the final parquet.

    Args:
        from_disk: If True, load raw frames from ``data/raw/`` instead of
            re-downloading. Used by ``--process-only``.

    Returns:
        Absolute path of the written parquet file.
    """
    configure_logging()

    logger.info(
        "Step 1/5: acquire raw data ({})",
        "from disk" if from_disk else "live download",
    )
    raw = load_raw_from_disk() if from_disk else download_all()

    logger.info("Step 2/5: aggregate to quarterly + merge")
    merged = aggregate_to_quarterly(raw)
    logger.info("  merged shape pre-engineer: {}", merged.shape)

    logger.info("Step 3/5: engineer features")
    engineered = engineer_features(merged)
    logger.info("  shape post-engineer (pre-trim): {}", engineered.shape)

    logger.info("Step 4/5: trim to project window + quality checks")
    final = _filter_to_project_window(engineered)
    logger.info("  trimmed shape: {}", final.shape)
    check_final_dataset(final)

    logger.info("Step 5/5: schema validation + save parquet")
    FINAL_DATASET_SCHEMA.validate(final)

    out_path = _repo_root() / pipeline_config()["paths"]["final_dataset"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)
    logger.info("Saved final dataset: {} (shape={})", out_path, final.shape)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="UK GDP regime forecasting data pipeline.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true",
                      help="Download raw data only.")
    mode.add_argument(
        "--process-only",
        action="store_true",
        help="Skip downloads; read raw CSVs from disk.",
    )
    args = parser.parse_args(argv)

    if args.download_only:
        configure_logging()
        download_all()
        return

    build(from_disk=args.process_only)


if __name__ == "__main__":
    main()
