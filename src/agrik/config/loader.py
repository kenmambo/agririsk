"""Loader for ``config/pipeline.yaml``.

The rest of the codebase reads declarative settings (panel definition, dataset
registry, validation rules, feature windows, model hyper-parameters) through
:func:`get_pipeline_config` instead of hard-coding them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..settings import get_settings


def load_pipeline_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the pipeline configuration dictionary.

    Parameters
    ----------
    path:
        Explicit path to a YAML config. Defaults to ``config/pipeline.yaml``
        from :class:`~agrik.settings.Settings`.
    """
    config_path = Path(path) if path is not None else get_settings().pipeline_config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Pipeline config must be a mapping, got {type(config)!r}")
    return config


@lru_cache(maxsize=1)
def get_pipeline_config() -> dict[str, Any]:
    """Return a cached copy of the pipeline configuration."""
    return load_pipeline_config()
