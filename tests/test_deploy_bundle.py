"""Deploy-bundle tests: layout, honest synthetic flag, and API-serving-from-bundle.

The Docker image serves exclusively from ``deploy/seed/`` (see Dockerfile), so
these tests prove the bundle contains every artefact the serving layer needs -
the same contract the container relies on, verified offline.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agrik.api import create_app
from agrik.pipeline import run_pipeline
from agrik.settings import get_settings

_ROOT = Path(__file__).resolve().parents[1]


def _load_exporter():
    """Import scripts/export_deploy_bundle.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "export_deploy_bundle", _ROOT / "scripts" / "export_deploy_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_bundle_layout_and_honest_flag(isolated_env, monkeypatch):
    run_pipeline(force_raw=True, seed=3)
    s = get_settings()
    # A granule-cache file must never leak into a deployable bundle.
    granule = s.external_dir / "modis"
    granule.mkdir(parents=True, exist_ok=True)
    (granule / "dummy.h4").write_bytes(b"not a granule")

    exporter = _load_exporter()
    out = s.data_root.parent / "bundle"
    summary = exporter.export_bundle(out)

    assert (out / "data" / "raw" / "climate_monthly.csv").exists()
    assert (out / "data" / "processed" / "master_panel.csv").exists()
    assert (out / "data" / "features" / "features_panel.csv").exists()
    assert (out / "models" / "model_card.json").exists()
    assert (out / "models" / "baseline_gbm.joblib").exists()  # primary
    assert (out / "models" / "baseline_ridge.joblib").exists()  # compare floor
    assert not (out / "data" / "raw" / "external").exists()
    assert summary["files"] > 0 and summary["bytes"] > 0
    # Synthetic test-feed data must be labelled synthetic, never hidden.
    assert summary["data_is_synthetic"] is True
    assert "climate" in summary["dataset_sources"]


def test_export_bundle_without_artefacts_fails_loudly(isolated_env):
    exporter = _load_exporter()
    out = isolated_env / "bundle"
    with pytest.raises(SystemExit, match="python -m agrik"):
        exporter.export_bundle(out)


def test_api_serves_from_bundle_only(isolated_env, monkeypatch):
    """Point the API at the bundle (fresh dirs) - the container's exact situation."""
    run_pipeline(force_raw=True, seed=3)
    s = get_settings()
    source_card = json.loads((s.models_dir / "model_card.json").read_text(encoding="utf-8"))

    exporter = _load_exporter()
    bundle = s.data_root.parent / "bundle"
    exporter.export_bundle(bundle)

    # Re-home settings at the bundle, like WORKDIR /app inside the image.
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(bundle / "data"))
    monkeypatch.setenv("AGRIK_MODELS_DIR", str(bundle / "models"))
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        assert client.get("/health").json()["artefacts"]["model_card"] is True
        status = client.get("/data/status").json()
        assert status["rows"] > 0
        assert status["data_is_synthetic"] is True  # honest label survives the copy
        card = client.get("/model/card").json()
        assert card["metrics"]["rmse"] == pytest.approx(source_card["metrics"]["rmse"])
        preds = client.get("/model/predictions").json()
        assert len(preds["rows"]) > 0
    finally:
        get_settings.cache_clear()
