"""Unit and integration tests for the downloaders.

Unit tests use monkeypatch to redirect file output to tmp_path
and to fake HTTP responses; no network is touched. Integration tests are
marked @pytest.mark.integration and only run when invoked explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.data.downloaders._common as common_mod
import src.data.downloaders.boe as boe_mod
import src.data.downloaders.fred as fred_mod
import src.data.downloaders.ons as ons_mod
from src.data.downloaders._common import PENDING_SENTINEL

# Sentinel rejection


def _fake_ons_cfg(cdid: str, *, deferred: bool = False) -> dict:
    return {
        "data_sources": {
            "ons": {
                "api_base": "https://www.ons.gov.uk",
                "series": {
                    "test_series": {
                        "cdid": cdid,
                        "path": "test/path",
                        "dataset": "testds",
                        "frequency": "quarterly",
                        "deferred": deferred,
                    },
                },
            },
        },
    }


def _fake_boe_cfg(code: str, url: str, *, deferred: bool = False) -> dict:
    return {
        "data_sources": {
            "boe": {
                "iadb_base": "https://example.test/",
                "series": {
                    "test_series": {
                        "code": code,
                        "url": url,
                        "frequency": "monthly",
                        "deferred": deferred,
                    },
                },
            },
        },
    }


def _fake_fred_cfg(code: str, *, deferred: bool = False) -> dict:
    return {
        "data_sources": {
            "fred": {
                "api_base": "https://api.stlouisfed.org/fred/series/observations",
                "api_key_env": "FRED_API_KEY",
                "rate_limit_sleep_seconds": 0.0,
                "series": {
                    "test_series": {
                        "code": code,
                        "frequency": "monthly",
                        "deferred": deferred,
                    },
                },
            },
        },
    }


def test_ons_refuses_pending_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ons_mod, "pipeline_config",
                        lambda: _fake_ons_cfg(PENDING_SENTINEL))
    with pytest.raises(RuntimeError, match="unresolved"):
        ons_mod.download_ons_series("test_series", save=False)


def test_boe_refuses_pending_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boe_mod, "pipeline_config",
                        lambda: _fake_boe_cfg("CODE", PENDING_SENTINEL))
    with pytest.raises(RuntimeError, match="unresolved"):
        boe_mod.download_boe_series("test_series", save=False)


def test_fred_refuses_pending_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "pipeline_config",
                        lambda: _fake_fred_cfg(PENDING_SENTINEL))
    monkeypatch.setenv("FRED_API_KEY", "test_key")
    with pytest.raises(RuntimeError, match="unresolved"):
        fred_mod.download_fred_series("test_series", save=False)


def test_fred_refuses_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "pipeline_config",
                        lambda: _fake_fred_cfg("ANY"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(fred_mod, "load_dotenv", lambda *_a, **_kw: None)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        fred_mod.download_fred_series("test_series", save=False)


# Deferred skip


def test_ons_skips_deferred_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ons_mod, "pipeline_config",
                        lambda: _fake_ons_cfg("ANY", deferred=True))
    assert ons_mod.download_ons_series("test_series", save=False) is None


def test_boe_skips_deferred_2y_gilt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        boe_mod,
        "pipeline_config",
        lambda: _fake_boe_cfg(
            PENDING_SENTINEL, PENDING_SENTINEL, deferred=True),
    )
    assert boe_mod.download_boe_series("test_series", save=False) is None


def test_fred_skips_deferred_series(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fred_mod, "pipeline_config",
                        lambda: _fake_fred_cfg("ANY", deferred=True))
    monkeypatch.setenv("FRED_API_KEY", "test_key")
    assert fred_mod.download_fred_series("test_series", save=False) is None


# Parsers


ONS_QUARTERLY_PAYLOAD = {
    "description": {"title": "Test"},
    "quarters": [
        {"date": "2020 Q1", "value": "-2.0"},
        {"date": "2020 Q2", "value": "-19.4"},
        {"date": "2020 Q3", "value": "17.0"},
    ],
    "months": [],
    "years": [],
}

ONS_MONTHLY_PAYLOAD = {
    "quarters": [],
    "months": [
        {"date": "2020 JAN", "value": "100.0"},
        {"date": "2020 FEB", "value": "101.0"},
    ],
    "years": [],
}

FRED_PAYLOAD = {
    "observations": [
        {"date": "2020-01-01", "value": "50.0"},
        {"date": "2020-02-01", "value": "55.0"},
        {"date": "2020-03-01", "value": "."},
    ],
}


def test_ons_parser_quarterly() -> None:
    df = ons_mod._parse_ons_response(ONS_QUARTERLY_PAYLOAD, "quarterly")
    assert list(df["date"]) == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2020-09-30"),
    ]
    assert list(df["value"]) == [-2.0, -19.4, 17.0]


def test_ons_parser_monthly() -> None:
    df = ons_mod._parse_ons_response(ONS_MONTHLY_PAYLOAD, "monthly")
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert df["value"].iloc[1] == 101.0


def test_ons_parser_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="frequency"):
        ons_mod._parse_ons_response(ONS_QUARTERLY_PAYLOAD, "weekly")


def test_ons_parser_rejects_empty_array() -> None:
    with pytest.raises(RuntimeError, match="quarters"):
        ons_mod._parse_ons_response({"quarters": []}, "quarterly")


def test_fred_parser_drops_missing_value_dot() -> None:
    df = fred_mod._parse_fred_response(FRED_PAYLOAD)
    assert len(df) == 2
    assert df["value"].iloc[0] == 50.0


def test_fred_parser_rejects_empty_observations() -> None:
    with pytest.raises(RuntimeError, match="observations"):
        fred_mod._parse_fred_response({"observations": []})


def test_boe_parser_normalises_iadb_csv() -> None:
    text = "DATE,IUDBEDR\n01 Jan 2020,0.75\n01 Feb 2020,0.50\n"
    df = boe_mod._parse_iadb_csv(text)
    assert list(df.columns) == ["date", "value"]
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01")
    assert df["value"].iloc[1] == 0.5


# End-to-end with mocks: artefacts (CSV, lineage, cache) are written


def _redirect_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common_mod, "repo_root", lambda: tmp_path)


def test_ons_end_to_end_writes_artefacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_outputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ons_mod,
        "pipeline_config",
        lambda: {
            "data_sources": {
                "ons": {
                    "api_base": "https://www.ons.gov.uk",
                    "series": {
                        "gdp_growth": {
                            "cdid": "IHYQ",
                            "path": "economy/grossdomesticproductgdp",
                            "dataset": "qna",
                            "frequency": "quarterly",
                        },
                    },
                },
            },
        },
    )

    fake_response = MagicMock()
    fake_response.content = json.dumps(ONS_QUARTERLY_PAYLOAD).encode("utf-8")
    fake_response.json.return_value = ONS_QUARTERLY_PAYLOAD
    fake_response.raise_for_status = MagicMock()
    monkeypatch.setattr(ons_mod.requests, "get",
                        lambda *_a, **_kw: fake_response)

    df = ons_mod.download_ons_series("gdp_growth", save=True)
    assert df is not None and len(df) == 3

    csv_path = tmp_path / "data" / "raw" / "ons" / "gdp_growth.csv"
    assert csv_path.exists()

    lineage_path = tmp_path / "data" / "lineage" / "ons__gdp_growth.lineage.json"
    assert lineage_path.exists()
    record = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert record["source"] == "ons:IHYQ"
    assert "validate" in record["transformations"]

    cache_dir = tmp_path / "data" / "raw" / "_api_responses" / "ons"
    cached = list(cache_dir.glob("gdp_growth__*.json"))
    assert len(cached) == 1


def test_fred_end_to_end_writes_artefacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_outputs(monkeypatch, tmp_path)
    monkeypatch.setenv("FRED_API_KEY", "test_key")
    monkeypatch.setattr(fred_mod, "load_dotenv", lambda *_a, **_kw: None)
    monkeypatch.setattr(fred_mod, "pipeline_config",
                        lambda: _fake_fred_cfg("DCOILBRENTEU"))

    fake_response = MagicMock()
    fake_response.content = json.dumps(FRED_PAYLOAD).encode("utf-8")
    fake_response.json.return_value = FRED_PAYLOAD
    fake_response.raise_for_status = MagicMock()
    monkeypatch.setattr(fred_mod.requests, "get",
                        lambda *_a, **_kw: fake_response)

    df = fred_mod.download_fred_series("test_series", save=True)
    assert df is not None and len(df) == 2

    csv_path = tmp_path / "data" / "raw" / "fred" / "test_series.csv"
    assert csv_path.exists()
    lineage_path = tmp_path / "data" / "lineage" / "fred__test_series.lineage.json"
    assert lineage_path.exists()
    record = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert "api_key" not in record["parameters"]  # never log secrets


# Cache pruning


def test_cache_raw_response_keeps_only_last_five(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_outputs(monkeypatch, tmp_path)
    for i in range(7):
        common_mod.cache_raw_response(
            "ons", "x", f"payload{i}".encode(), "json")

    cache_dir = tmp_path / "data" / "raw" / "_api_responses" / "ons"
    files = sorted(cache_dir.glob("x__*.json"))
    assert len(files) == 5


# Integration: real network (manual only)


@pytest.mark.integration
def test_ons_gdp_growth_real_download() -> None:
    df = ons_mod.download_ons_series("gdp_growth", save=False)
    assert df is not None and len(df) > 80


@pytest.mark.integration
def test_fred_brent_oil_real_download() -> None:
    df = fred_mod.download_fred_series("brent_oil", save=False)
    assert df is not None and len(df) > 100
