"""Shared pytest fixtures.

The pipeline resolves data/model paths from ``Settings`` (CWD-relative). Tests
redirect those to a temporary directory via environment variables and clear the
relevant caches so nothing is written into the real ``data/`` or ``models/``.
"""

from __future__ import annotations

import pytest

from agrik import settings
from agrik.config import loader as cfg_loader


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point data + model artefacts at a temp dir and clear config caches."""
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AGRIK_MODELS_DIR", str(tmp_path / "models"))
    settings.get_settings.cache_clear()
    cfg_loader.get_pipeline_config.cache_clear()
    yield tmp_path
    settings.get_settings.cache_clear()
    cfg_loader.get_pipeline_config.cache_clear()
