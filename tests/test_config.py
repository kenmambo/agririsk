"""Tests for configuration management."""

from __future__ import annotations

import pytest

from agrik.config import get_pipeline_config, load_pipeline_config
from agrik.settings import get_settings


def test_pipeline_config_has_sections():
    cfg = get_pipeline_config()
    for section in ("panel", "datasets", "validation", "feature_engineering", "model"):
        assert section in cfg
    assert cfg["model"]["target"] == "food_security_risk_index"


def test_settings_paths_are_derived():
    s = get_settings()
    assert s.raw_dir == s.data_root / "raw"
    assert s.processed_dir == s.data_root / "processed"


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline_config(tmp_path / "does_not_exist.yaml")
