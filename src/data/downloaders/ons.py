"""ONS time-series downloader.

Pulls ONS series from the ONS website time-series JSON endpoint:
``https://www.ons.gov.uk/<path>/timeseries/<CDID>/<dataset>/data``.
(The old ``api.ons.gov.uk`` v0 API this used to call was retired by ONS
in late 2024, so each series is now identified by its CDID *plus* its
dataset and section path.) The endpoint returns JSON with ``quarters``,
``months``, and ``years`` arrays, each entry holding ``date`` and
``value`` strings. This module parses the frequency-appropriate array
into a ``(date, value)`` DataFrame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
from loguru import logger

from data.schemas.raw_ons import RAW_ONS_SCHEMA
from src.data.config import pipeline_config
from src.data.downloaders._common import (
    cache_raw_response,
    require_resolved,
    write_raw_csv,
    write_series_lineage,
)

_QUARTER_END_MONTH_DAY = {"1": "03-31",
                          "2": "06-30", "3": "09-30", "4": "12-31"}
_MONTH_ABBR = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}  # fmt: skip
_FREQUENCY_KEY = {"quarterly": "quarters",
                  "monthly": "months", "annual": "years"}


def download_ons_series(series_name: str, *, save: bool = True) -> pd.DataFrame | None:
    """Download one ONS series by canonical name.

    Args:
        series_name: Key under ``data_sources.ons.series`` in pipeline config.
        save: If True, write the raw CSV, cache the API response, and write
            a lineage record. Set False for in-memory probing.

    Returns:
        A ``(date, value)`` DataFrame, or ``None`` if the series is deferred.
    """
    cfg = pipeline_config()["data_sources"]["ons"]
    series_cfg = cfg["series"][series_name]

    if series_cfg.get("deferred", False):
        logger.info("ONS {}: deferred — skipping", series_name)
        return None

    cdid = series_cfg["cdid"]
    path = series_cfg["path"]
    dataset = series_cfg["dataset"]
    require_resolved(series_name, cdid=cdid, path=path, dataset=dataset)

    api_base = cfg["api_base"].rstrip("/")
    url = f"{api_base}/{path}/timeseries/{cdid}/{dataset}/data"
    logger.info("ONS {} (CDID={}, dataset={}): GET {}",
                series_name, cdid, dataset, url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    cache_raw_response("ons", series_name, response.content, "json")

    df = _parse_ons_response(response.json(), series_cfg["frequency"])
    RAW_ONS_SCHEMA.validate(df)

    if save:
        out = write_raw_csv(df, "ons", series_name)
        write_series_lineage(
            source="ons",
            series=series_name,
            code=cdid,
            raw_csv_path=out,
            transformations=["http_get", "parse_ons_json", "validate"],
            parameters={
                "cdid": cdid,
                "path": path,
                "dataset": dataset,
                "frequency": series_cfg["frequency"],
                "url": url,
            },
        )
    return df


def download_all_ons() -> dict[str, pd.DataFrame | None]:
    """Download every series under ``data_sources.ons``."""
    cfg = pipeline_config()["data_sources"]["ons"]["series"]
    return {name: download_ons_series(name) for name in cfg}


def _parse_ons_response(payload: dict[str, Any], frequency: str) -> pd.DataFrame:
    """Parse the ONS /timeseries response JSON into a ``(date, value)`` frame.

    Args:
        payload: Parsed JSON body returned by the ONS endpoint.
        frequency: ``"quarterly"``, ``"monthly"``, or ``"annual"``.

    Returns:
        A two-column DataFrame ordered by date.

    Raises:
        ValueError: If ``frequency`` is not recognised.
        RuntimeError: If the expected array is empty.
    """
    key = _FREQUENCY_KEY.get(frequency)
    if key is None:
        raise ValueError(f"Unsupported ONS frequency: {frequency!r}")
    rows = payload.get(key) or []
    if not rows:
        raise RuntimeError(f"ONS response had no '{key}' array.")

    df = pd.DataFrame(rows)
    df["date"] = df["date"].apply(_ons_date_to_iso).pipe(pd.to_datetime)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def _ons_date_to_iso(raw: str) -> str:
    """Translate an ONS date string (``'2020 Q2'``, ``'2020 JAN'``) to ISO format.

    Quarterly strings resolve to the quarter-end date; monthly strings to the
    first of the month; bare years to YYYY-12-31.
    """
    text = raw.strip()
    if " Q" in text:
        year, quarter = text.split(" Q")
        return f"{year}-{_QUARTER_END_MONTH_DAY[quarter]}"
    if " " in text:
        year, month_abbr = text.split(" ")
        return f"{year}-{_MONTH_ABBR[month_abbr.upper()]}-01"
    if text.isdigit() and len(text) == 4:
        return f"{text}-12-31"
    return text


# Re-export for tests to monkeypatch the requests symbol cleanly.
__all__ = ["download_ons_series", "download_all_ons"]
