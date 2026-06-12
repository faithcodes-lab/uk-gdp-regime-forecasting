"""Track where each pipeline output came from.

Every pipeline step saves a LineageRecord as a JSON file (by convention,
under data/lineage/). Each record stores the source the data came from,
where the output was written, what was done to it, the parameters used,
when it ran (UTC), and the git commit the code was on at the time.

write_lineage saves a record; read_lineage loads it back. 
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LineageRecord:
    """A single lineage record describing one pipeline artefact.

    Attributes:
        source: Where the data came from (e.g. "fred:DCOILBRENTEU").
        output_path: Where the output was written, relative to the repo root.
        transformations: The steps applied, in order.
        parameters: Any parameters used during the run.
        timestamp_utc: When it ran, in UTC. Filled in automatically if blank.
        git_commit: The git commit the code was on. Filled in automatically if blank.
    """

    source: str
    output_path: str
    transformations: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = ""
    git_commit: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        if not self.git_commit:
            self.git_commit = current_git_commit()


def write_lineage(record: LineageRecord, path: Path | str) -> None:
    """Save a LineageRecord to disk as a JSON file.

    Args:
        record: Record to write.
        path: Destination JSON file path. Parent directories are created.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, indent=2, sort_keys=True)


def read_lineage(path: Path | str) -> LineageRecord:
    """Read a JSON lineage file into a :class:LineageRecord.

    Args:
        path: JSON file written by :func:`write_lineage`.

    Returns:
        A :class:LineageRecord with the fields stored in the file.
    """
    src = Path(path)
    with src.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return LineageRecord(**data)


def current_git_commit() -> str:
    """Return the current git commit SHA, or a fallback marker.

    Returns:
        The commit ID (SHA) of the current HEAD. If there are uncommitted
        changes, "-dirty" is added on the end. If this isn't a git repo, or
        git can't be run, returns "no-git".
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return "no-git"
    if head.returncode != 0:
        return "no-git"
    commit = head.stdout.strip()
    if not commit:
        return "no-git"
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return commit
    if status.returncode == 0 and status.stdout.strip():
        return f"{commit}-dirty"
    return commit
