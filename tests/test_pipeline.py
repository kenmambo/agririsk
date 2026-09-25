"""End-to-end pipeline integration test (writes only to a temp dir)."""

from __future__ import annotations

import math

from agrik.pipeline import run_pipeline
from agrik.settings import get_settings


def test_full_pipeline_produces_artefacts(isolated_env):
    summary = run_pipeline(force_raw=True, seed=3)
    s = get_settings()

    # Raw synthetic + processed + feature store + model artefacts all exist.
    assert (s.raw_dir / "climate_monthly.csv").exists()
    assert (s.processed_dir / "master_panel.csv").exists()
    assert (s.features_dir / "features_panel.csv").exists()
    assert (s.features_dir / "features_manifest.json").exists()
    assert (s.models_dir / "baseline_ridge.joblib").exists()
    assert (s.models_dir / "model_card.json").exists()

    # The whole 21-county registry is exercised at county-monthly grain.
    assert summary["counties"] == 21
    assert summary["master_rows"] > 0

    metrics = summary["metrics"]
    assert math.isfinite(metrics["rmse"])
    assert metrics["n_features"] > 0
    assert metrics["n_test"] > 0
