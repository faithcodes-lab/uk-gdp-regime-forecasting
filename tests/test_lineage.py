"""Tests for :mod:`src.data.lineage`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.data.lineage import (
    LineageRecord,
    current_git_commit,
    read_lineage,
    write_lineage,
)

EXPECTED_FIELDS = {
    "source",
    "output_path",
    "transformations",
    "parameters",
    "timestamp_utc",
    "git_commit",
}


def test_lineage_record_autofills_timestamp_and_commit() -> None:
    record = LineageRecord(source="test", output_path="data/processed/test.csv")
    assert record.timestamp_utc != ""
    assert record.git_commit != ""


def test_lineage_round_trip(tmp_path: Path) -> None:
    record = LineageRecord(
        source="fred:DCOILBRENTEU",
        output_path="data/raw/fred/brent_oil.csv",
        transformations=["download", "validate"],
        parameters={"series_id": "DCOILBRENTEU", "rate_limit_s": 0.5},
    )
    out = tmp_path / "brent_oil.lineage.json"
    write_lineage(record, out)
    restored = read_lineage(out)
    assert restored == record


def test_lineage_file_has_all_expected_fields(tmp_path: Path) -> None:
    record = LineageRecord(source="t", output_path="o")
    out = tmp_path / "lineage.json"
    write_lineage(record, out)
    with out.open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert EXPECTED_FIELDS <= set(data.keys())


def test_write_lineage_creates_parent_directories(tmp_path: Path) -> None:
    record = LineageRecord(source="t", output_path="o")
    nested = tmp_path / "a" / "b" / "c" / "lineage.json"
    write_lineage(record, nested)
    assert nested.exists()


def test_current_git_commit_returns_sha_in_this_repo() -> None:
    commit = current_git_commit()
    base = commit.removesuffix("-dirty")
    assert len(base) == 40 and all(c in "0123456789abcdef" for c in base)


def test_current_git_commit_falls_back_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: object, **_kwargs: object) -> object:
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert current_git_commit() == "no-git"


def test_current_git_commit_falls_back_when_rev_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompleted:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    def fake_run(*_args: object, **_kwargs: object) -> FakeCompleted:
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert current_git_commit() == "no-git"
