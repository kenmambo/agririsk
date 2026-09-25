"""End-to-end pipeline orchestration.

Wires the separated layers together in the correct order and is the ONLY module
allowed to touch all of them:

    ingestion  ->  processing (validate + clean + merge)  ->  features  ->  model

Run it with ``python -m agrik`` or ``agrik-build-data``. The Streamlit dashboard
consumes the artefacts this writes; it never runs ingestion or modelling itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import counties, schemas
from .config import get_pipeline_config
from .features import build_features, make_xy
from .ingestion import read_csv, write_csv, write_synthetic_raw
from .logging import get_logger, setup_logging
from .models import build_model, temporal_train_test_split
from .processing import clean_dataset, impute_panel, merge_datasets, validate_panel
from .settings import get_settings

LOGGER = get_logger("pipeline")
PIPELINE_VERSION = "0.1.0"


def _specs() -> dict:
    return schemas.dataset_specs(get_pipeline_config())


def step_ingest(force: bool, seed: int) -> dict[str, Path]:
    """Generate synthetic raw data if missing (or forced)."""
    settings = get_settings()
    settings.ensure_dirs()
    specs = _specs()
    missing = any(not (settings.raw_dir / s.file).exists() for s in specs.values())
    if force or missing:
        LOGGER.warning("Writing SYNTHETIC raw data (force=%s, missing=%s).", force, missing)
        return write_synthetic_raw(seed=seed)
    LOGGER.info("Raw data already present; skipping synthetic generation.")
    return {name: settings.raw_dir / spec.file for name, spec in specs.items()}


def step_process() -> pd.DataFrame:
    """Read raw datasets, validate, clean, merge and impute into a master panel."""
    specs = _specs()
    cleaned: dict[str, pd.DataFrame] = {}
    all_ok = True
    for name, spec in specs.items():
        df = read_csv(get_settings().raw_dir / spec.file)
        result = validate_panel(df)
        if not result.is_valid:
            all_ok = False
            LOGGER.error("Dataset '%s' failed validation: %s", name, result.errors)
        cleaned_df = clean_dataset(df)
        write_csv(cleaned_df, get_settings().processed_dir / spec.file)
        cleaned[name] = cleaned_df

    if not all_ok:
        raise RuntimeError("One or more raw datasets failed validation; aborting.")

    master = impute_panel(merge_datasets(cleaned))
    # Attach agro-ecological zone (county metadata) for filtering/display in the UI.
    zone_ref = counties.counties_frame()[["county_code", "agro_zone"]]
    master = master.merge(zone_ref, on="county_code", how="left")
    master_result = validate_panel(master)
    LOGGER.info("Master panel validation: %s", master_result.summary())
    write_csv(master, get_settings().processed_dir / "master_panel.csv")
    return master


def step_features(master: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the master panel and persist the feature store."""
    engineered = build_features(master)
    write_csv(engineered, get_settings().features_dir / "features_panel.csv")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "feature_version": _feature_version(),
        "rows": int(len(engineered)),
        "n_features_engineered": len(engineered.columns),
        "data_source": str(engineered[schemas.PROVENANCE_COLUMN].unique().tolist()),
        "note": schemas.SYNTHETIC_NOTE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (get_settings().features_dir / "features_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return engineered


def _feature_version() -> str:
    from .features import FEATURE_VERSION

    return FEATURE_VERSION


def step_train(engineered: pd.DataFrame):
    """Split, fit the baseline model, evaluate, and persist artefacts."""
    config = get_pipeline_config()
    test_size = float(config["model"].get("test_size", 0.25))

    X, y, feats = make_xy(engineered, config)
    time_index = engineered.loc[X.index, schemas.DATE_COLUMN].to_numpy()
    X_tr, X_te, y_tr, y_te = temporal_train_test_split(X, y, time_index, test_size=test_size)

    model = build_model(config=config)
    model.fit(X_tr, y_tr)
    metrics = model.evaluate(X_te, y_te)
    metrics["n_train"] = int(len(X_tr))
    metrics["n_test"] = int(len(X_te))
    metrics["n_features"] = len(feats)

    model_path = model.save(get_settings().models_dir / "baseline_ridge.joblib")
    card = model.model_card(metrics)
    card_path = get_settings().models_dir / "model_card.json"
    card_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Baseline model saved to %s; metrics=%s", model_path, metrics)
    return model, metrics, card


def run_pipeline(force_raw: bool = False, seed: int = 42) -> dict:
    """Run the full pipeline and return a summary dict."""
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    LOGGER.info("=== AgriRisk pipeline v%s starting (env=%s) ===",
                PIPELINE_VERSION, settings.environment)

    raw = step_ingest(force=force_raw, seed=seed)
    master = step_process()
    engineered = step_features(master)
    _, metrics, _ = step_train(engineered)

    summary = {
        "datasets": sorted(raw),
        "master_rows": int(len(master)),
        "counties": int(master["county_code"].nunique()),
        "metrics": metrics,
    }
    LOGGER.info("=== Pipeline complete: %s ===", summary["metrics"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AgriRisk data + baseline model.")
    parser.add_argument("--force-raw", action="store_true",
                        help="Regenerate synthetic raw data even if present.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic data RNG seed.")
    args = parser.parse_args(argv)
    run_pipeline(force_raw=args.force_raw, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
