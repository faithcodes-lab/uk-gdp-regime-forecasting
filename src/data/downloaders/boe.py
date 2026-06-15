"""Download Bank of England series from the Interactive Database (IADB).

The IADB export URLs are not easy to build by hand, so the resolved URLs are
stored directly in config/pipeline.yaml .
"""

from __future__ import annotations

import io

import pandas as pd
import requests
from loguru import logger

from data.schemas.raw_boe_rate import BOE_RATE_RAW_SCHEMA
from data.schemas.raw_ons import RAW_ONS_SCHEMA
from src.data.config import pipeline_config

from src.data.downloaders._common import (
    cache_raw_response,
    require_resolved,
    write_raw_csv,
    write_series_lineage,
)


def download_boe_series(series_name: str, *, save: bool = True) -> pd.DataFrame:
    """
    Turn a BoE IADB CSV export into a clean (date, value) table.

     IADB exports don't always use the same column names, so this lower-cases
     the column names, treats the first non-date column as the value, and reads
     dates in the IADB "DD MMM YYYY" format (e.g. 01 Jan 2000).
    """

    cfg = pipeline_config()["data_sources"]["boe"]
    series_cfg = cfg["series"][series_name]

    code = series_cfg["code"]
    url = series_cfg["url"]
    require_resolved(series_name, code=code, url=url)

    logger.info("BoE {} (code={}): GET {}", series_name, code, url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    cache_raw_response("boe", series_name, response.content, "csv")

    df = _parse_iadb_csv(response.text)

    if series_name == "bank_rate":
        BOE_RATE_RAW_SCHEMA.validate(df.rename(columns={"value": "bank_rate"}))
    else:
        RAW_ONS_SCHEMA.validate(df)

    if save:
        out = write_raw_csv(df, "boe", series_name)
        write_series_lineage(
            source="boe",
            series=series_name,
            code=code,
            raw_csv_path=out,
            transformations=["http_get", "parse_iadb_csv", "validate"],
            parameters={"code": code, "frequency": series_cfg["frequency"], "url": url},
        )
    return df


def download_all_boe() -> dict[str, pd.DataFrame]:
    """Download every series under ``data_sources.boe``."""
    cfg = pipeline_config()["data_sources"]["boe"]["series"]
    return {name: download_boe_series(name) for name in cfg}


def _parse_iadb_csv(text: str) -> pd.DataFrame:
    """Parse a BoE IADB CSV export into a normalised (date, value) frame.

    IADB exports vary by query; the loader normalises column names to
    lower-case, treats the first non-date column as the value, and parses
    dates with the  ``DD MMM YYYY`` IADB format.
    """
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    non_date_cols = [c for c in df.columns if c != "date"]
    if not non_date_cols:
        raise RuntimeError("BoE IADB CSV had no value column.")
    df = df.rename(columns={non_date_cols[0]: "value"})
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)


__all__ = ["download_boe_series", "download_all_boe"]
