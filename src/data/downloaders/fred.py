"""Download FRED series from the /fred/series/observations endpoint.

Reads the API key from the FRED_API_KEY environment variable (loaded from
your .env file). Pauses for rate_limit_sleep_seconds between requests so it
stay well within FRED's rate limits.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

from data.schemas.raw_fred import RAW_FRED_SCHEMA
from src.data.config import pipeline_config
from src.data.downloaders._common import (
    cache_raw_response,
    require_resolved,
    write_raw_csv,
    write_series_lineage,
)


def download_fred_series(series_name: str, *, save: bool = True) -> pd.DataFrame:
    """Download one FRED series by its config name.

    Args:
        series_name: A key under data_sources.fred.series in pipeline.yaml.
        save: If True, also save the raw CSV, a cached copy of the response,
            and a lineage record.

    Returns:
        (date, value) DataFrame.

    Raises:
        RuntimeError: If FRED_API_KEY is not set or the series code is
            still _PENDING_VERIFICATION.
    """
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set; add it to .env before downloading.")

    cfg = pipeline_config()["data_sources"]["fred"]
    series_cfg = cfg["series"][series_name]

    code = series_cfg["code"]
    require_resolved(series_name, code=code)

    api_base = cfg["api_base"]
    params = {"series_id": code, "api_key": api_key, "file_type": "json"}
    logger.info("FRED {} (code={}): GET {}", series_name, code, api_base)

    time.sleep(float(cfg.get("rate_limit_sleep_seconds", 0.5)))

    response = requests.get(api_base, params=params, timeout=30)
    response.raise_for_status()
    cache_raw_response("fred", series_name, response.content, "json")

    df = _parse_fred_response(response.json())
    RAW_FRED_SCHEMA.validate(df)

    if save:
        out = write_raw_csv(df, "fred", series_name)

        # Parameters logged without the API key never log sensitive info but are still useful for debugging.
        write_series_lineage(
            source="fred",
            series=series_name,
            code=code,
            raw_csv_path=out,
            transformations=["http_get", "parse_fred_json", "validate"],
            parameters={
                "code": code,
                "frequency": series_cfg["frequency"],
                "rate_limit_sleep_seconds": cfg.get("rate_limit_sleep_seconds", 0.5),
            },
        )
    return df


def download_all_fred() -> dict[str, pd.DataFrame]:
    """Download every series under ``data_sources.fred``."""
    cfg = pipeline_config()["data_sources"]["fred"]["series"]
    return {name: download_fred_series(name) for name in cfg}


def _parse_fred_response(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse the FRED observations array into a (date, value) frame.

    FRED encodes missing values as the literal string ``"."``; these rows
    are dropped here so the schema sees only real numerics.
    """
    rows = payload.get("observations") or []
    if not rows:
        raise RuntimeError("FRED response had no observations.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype(float)
    return df[["date", "value"]].dropna(subset=["value"]).reset_index(drop=True)


__all__ = ["download_fred_series", "download_all_fred"]
