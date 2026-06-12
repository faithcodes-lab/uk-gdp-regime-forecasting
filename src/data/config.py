"""Configuration loaders for the UK GDP regime forecasting pipeline.

Exposes cached, typed accessors for the three YAML configuration files
under config/:

* :func:`pipeline_config`: data sources, paths, project metadata
* :func:`features_config`: target, predictors, engineered features
* :func:`regimes_config` : the six economic regime definitions

Each loader is cached with :func:`functools.lru_cache` so repeated calls
re use the parsed dict rather than re-reading the file.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# The root of the repository is two levels up from this file, and the config directory is under the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# the config/ folder holding the three YAML files
_CONFIG_DIR = _REPO_ROOT / "config"


def _load(filename: str) -> dict[str, Any]:
    """Load and parse a YAML file from ``config/``.

    Args:
        filename: File name (e.g. ``"pipeline.yaml"``) inside ``config/``.

    Returns:
        The parsed YAML contents as a dictionary.
    """

    # full path to the YAML file inside config/
    path = _CONFIG_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

# cache the result so the file is read from disk only once


@functools.lru_cache(maxsize=1)
def pipeline_config() -> dict[str, Any]:
    """Return the parsed contents of ``config/pipeline.yaml``."""
    return _load("pipeline.yaml")


@functools.lru_cache(maxsize=1)
def features_config() -> dict[str, Any]:
    """Return the parsed contents of ``config/features.yaml``."""
    return _load("features.yaml")


@functools.lru_cache(maxsize=1)
def regimes_config() -> dict[str, Any]:
    """Return the parsed contents of ``config/regimes.yaml``."""
    return _load("regimes.yaml")
