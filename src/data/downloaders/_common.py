"""Shared helpers for downloaders: sentinel checks, paths, caching, lineage."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.lineage import LineageRecord, write_lineage

PENDING_SENTINEL = "_PENDING_VERIFICATION"


def is_pending(value: Any) -> bool:
    """Return True if a config value is the unresolved-verification sentinel."""
    return value == PENDING_SENTINEL


def require_resolved(series_name: str, **fields: Any) -> None:
    """Raise :class: RuntimeError if any of fields is the pending sentinel.

    Args:
        series_name: The series being checked (shown in the error message).
        **fields: Mapping of field name to value to check.

    Raises:
        RuntimeError: With the list of unresolved fields and a hint to edit
            ``config/pipeline.yaml``.
    """
    unresolved = [name for name, value in fields.items() if is_pending(value)]
    if unresolved:
        raise RuntimeError(
            f"Series {series_name!r} has unresolved field(s): {unresolved}. "
            f"Resolve in config/pipeline.yaml before downloading."
        )


def repo_root() -> Path:
    """Return the repository root inferred from this module's location."""
    return Path(__file__).resolve().parents[3]


def raw_data_dir(source: str) -> Path:
    """Return the raw-data directory for ``source`` (created on demand)."""
    out = repo_root() / "data" / "raw" / source
    out.mkdir(parents=True, exist_ok=True)
    return out


def api_cache_dir(source: str) -> Path:
    """Return the raw-API-response cache directory for ``source``."""
    out = repo_root() / "data" / "raw" / "_api_responses" / source
    out.mkdir(parents=True, exist_ok=True)
    return out


def lineage_dir() -> Path:
    """Return the lineage directory (created on demand)."""
    out = repo_root() / "data" / "lineage"
    out.mkdir(parents=True, exist_ok=True)
    return out


def cache_raw_response(source: str, series: str, content: bytes, ext: str) -> Path:
    """Save a raw API response to disk and tidy up the older ones.

    Keeps only the five most recent saved responses for each
    source/series/file-type, deleting older ones so the cache doesn't grow
    without limit.

    Args:
        source: Short source name ("ons", "boe", "fred").
        series: The series' standard name.
        content: The raw response, as bytes.
        ext: The file type, without the dot ("json", "csv").

    Returns:
        The path of the file just written.
    """

    cache_dir = api_cache_dir(source)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = cache_dir / f"{series}__{ts}.{ext}"
    out.write_bytes(content)
    existing = sorted(cache_dir.glob(f"{series}__*.{ext}"))
    for old in existing[:-5]:
        old.unlink()
    return out


def write_raw_csv(df: pd.DataFrame, source: str, series: str) -> Path:
    """Save a raw (date, value) DataFrame to ``data/raw/<source>/<series>.csv``."""
    out = raw_data_dir(source) / f"{series}.csv"
    df.to_csv(out, index=False)
    return out


def write_series_lineage(
    *,
    source: str,
    series: str,
    code: str,
    raw_csv_path: Path,
    transformations: list[str],
    parameters: dict[str, Any],
) -> Path:
    """Write a :class:`LineageRecord` for a downloaded series.

    Args:
        source: Source slug.
        series: Canonical series name.
        code: Source-side identifier (CDID, FRED code, ticker).
        raw_csv_path: Path of the saved raw CSV.
        transformations: Ordered list of transformation names.
        parameters: Parameters captured for the lineage record.

    Returns:
        Path of the lineage JSON file.
    """
    record = LineageRecord(
        source=f"{source}:{code}",
        output_path=str(raw_csv_path.relative_to(repo_root())),
        transformations=transformations,
        parameters=parameters,
    )
    out = lineage_dir() / f"{source}__{series}.lineage.json"
    write_lineage(record, out)
    return out
