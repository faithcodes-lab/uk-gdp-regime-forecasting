"""Loguru-based logging configuration for the data pipeline.

Single entry point: :func:`configure_logging`. Idempotent — repeated calls
clear existing sinks and re-register them, so calling this in multiple
entry points (CLI scripts, downloaders, the orchestrator) does not stack
duplicate sinks.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOG_DIR = _REPO_ROOT / "logs"

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


def configure_logging(
    level_console: str = "INFO",
    level_file: str = "DEBUG",
    log_dir: Path | None = None,
) -> None:
    """Configure loguru sinks for the pipeline.

    Args:
        level_console: Minimum level emitted to stderr (default ``INFO``).
        level_file: Minimum level written to the rotating file sink
            (default ``DEBUG``).
        log_dir: Directory for the file sink. Defaults to ``<repo>/logs``.
    """
    target_dir = log_dir if log_dir is not None else _DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level_console,
        format=_CONSOLE_FORMAT,
    )
    logger.add(
        target_dir / "pipeline.log",
        level=level_file,
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )
