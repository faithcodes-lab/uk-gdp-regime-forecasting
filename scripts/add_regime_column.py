"""Add the regime column to data/processed/final_dataset.parquet.

What this script does:

- Loads the final dataset parquet file.
- Backs up the existing file to ``final_dataset.parquet.bak`` (single
  backup; overwrites any previous .bak).
- Drops any existing ``regime`` column (with a warning).
- Assigns each row to a regime via ``src.regimes.assign.assign_regimes``,
  using the regimes in ``config/regimes.yaml``.
- Writes the updated DataFrame back to ``final_dataset.parquet``.
- Validates the resulting parquet against ``FINAL_DATASET_SCHEMA``.

The script is idempotent: re-running it overwrites the previous regime
column and refreshes the backup. The base ``make data`` pipeline does
not produce the regime column; the documented rebuild is

    make data
    make regimes

Run with

    make regimes
or  PYTHONPATH=. python scripts/add_regime_column.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from loguru import logger

from data.schemas.processed_quarterly import FINAL_DATASET_SCHEMA
from src.data.config import pipeline_config
from src.logging_setup import configure_logging
from src.regimes.assign import assign_regimes


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    configure_logging()
    parquet_path = _repo_root() / pipeline_config()["paths"]["final_dataset"]
    backup_path = parquet_path.with_suffix(parquet_path.suffix + ".bak")

    logger.info("Reading final dataset from {}", parquet_path)
    df = pd.read_parquet(parquet_path)

    if "regime" in df.columns:
        logger.warning(
            "regime column already exists in {}; overwriting", parquet_path.name)
        df = df.drop(columns=["regime"])

    logger.info("Backing up existing parquet to {}", backup_path.name)
    shutil.copy2(parquet_path, backup_path)

    logger.info("Assigning regimes")
    df_with_regime = assign_regimes(df)

    logger.info(
        "Writing parquet with {} columns ({} rows)",
        len(df_with_regime.columns),
        len(df_with_regime),
    )
    df_with_regime.to_parquet(parquet_path, index=False)

    logger.info("Validating against FINAL_DATASET_SCHEMA")
    FINAL_DATASET_SCHEMA.validate(df_with_regime)

    logger.info("Done. Regime row counts:")
    for label, count in df_with_regime["regime"].value_counts().sort_index().items():
        logger.info("  {:30s} {}", label, count)


if __name__ == "__main__":
    main()
