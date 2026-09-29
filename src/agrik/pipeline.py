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
from .ingestion import ingest_datasets, read_csv, write_csv
from .logging import get_logger, setup_logging
from .models import (
    add_intervals,
    build_model,
    conformal_quantile,
    interval_metrics,
    reliability_table,
    temporal_three_way_split,
)
from .models.horizon import horizon_report
from .processing import clean_dataset, impute_panel, merge_datasets, validate_panel
from .settings import get_settings

LOGGER = get_logger("pipeline")
PIPELINE_VERSION = "0.2.0"


def _specs() -> dict:
    return schemas.dataset_specs(get_pipeline_config())


def _apply_source_overrides(overrides: list[str]) -> None:
    """Apply ``name=provider`` overrides onto the cached config (in-memory)."""
    cfg = get_pipeline_config()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--source expects name=provider, got {item!r}")
        name, provider = item.split("=", 1)
        if name not in cfg["datasets"]:
            raise ValueError(f"Unknown dataset {name!r} in --source override.")
        cfg["datasets"][name]["source"] = provider
        LOGGER.info("Config override: datasets.%s.source = %s", name, provider)


def _apply_set_overrides(overrides: list[str]) -> None:
    """Apply ``a.b.c=value`` dotted-path overrides onto the cached config."""
    cfg = get_pipeline_config()
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set expects dotted.key=value, got {item!r}")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        node: dict = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        value: object = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                pass
        node[keys[-1]] = value
        LOGGER.info("Config override: %s = %s", path, value)


def _dataset_provenance(frames: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {
        name: str(df[schemas.PROVENANCE_COLUMN].astype(str).unique().tolist())
        for name, df in frames.items()
    }


def _sources_note(prov: dict[str, str]) -> str:
    """Honest, dynamic provenance note derived from actual dataset sources."""
    synth = sorted(n for n, srcs in prov.items() if "synthetic" in srcs)
    real = {n: s for n, s in prov.items() if "synthetic" not in s}
    if not synth:
        return "All datasets are real feeds."
    note = (
        f"Datasets {synth} are SYNTHETIC sample data - no real analytical "
        "conclusions should be drawn from them."
    )
    if real:
        note += (
            f" Real feeds present: {real}. Note: modelling a synthetic target "
            "from partly-real features makes model metrics descriptive of the "
            "synthetic target only."
        )
    return note


def step_ingest(force: bool, seed: int) -> dict[str, Path]:
    """Build raw dataset files, routing each dataset to its configured source.

    Synthetic datasets are generated; real providers (CHIRPS, MODIS, ...) are
    used when configured, with clearly-labelled fallback on failure. Raw files
    are only rewritten when missing or forced - external rasters are cached
    separately under ``data/raw/external``.
    """
    settings = get_settings()
    settings.ensure_dirs()
    specs = _specs()
    missing = any(not (settings.raw_dir / s.file).exists() for s in specs.values())
    if not (force or missing):
        LOGGER.info("Raw data already present; skipping ingestion.")
        return {name: settings.raw_dir / spec.file for name, spec in specs.items()}

    LOGGER.info("Ingesting datasets (force=%s, missing=%s)...", force, missing)
    frames = ingest_datasets(seed=seed)
    prov = _dataset_provenance(frames)
    LOGGER.info("Dataset sources this run: %s", prov)
    written: dict[str, Path] = {}
    for name, df in frames.items():
        path = settings.raw_dir / specs[name].file
        written[name] = write_csv(df, path)
    return written


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


def _master_provenance(df: pd.DataFrame) -> dict[str, str]:
    """Per-dataset sources recovered from merged ``{name}_source`` columns."""
    out: dict[str, str] = {}
    for col in df.columns:
        if col.endswith("_source") and col != schemas.PROVENANCE_COLUMN:
            vals = sorted(df[col].dropna().astype(str).unique())
            out[col[: -len("_source")]] = ",".join(vals)
    return out


def step_features(master: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the master panel and persist the feature store."""
    engineered = build_features(master)
    write_csv(engineered, get_settings().features_dir / "features_panel.csv")
    prov = _master_provenance(engineered)
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "feature_version": _feature_version(),
        "rows": int(len(engineered)),
        "n_features_engineered": len(engineered.columns),
        "data_source": str(engineered[schemas.PROVENANCE_COLUMN].unique().tolist()),
        "dataset_sources": prov,
        "note": _sources_note(prov) if prov else schemas.SYNTHETIC_NOTE,
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
    """Fit primary + comparison models on a chronological split, calibrate
    conformal intervals, evaluate forecast horizons, persist artefacts.

    Artefacts written under ``models/``:
    - ``baseline_<name>.joblib``  : the primary model (and one per compared model)
    - ``model_card.json``         : primary metrics + uncertainty/calibration/horizon
    - ``comparison.json``         : side-by-side metrics on the identical split
    - ``test_predictions.csv``    : held-out predictions with 90% conformal band
    """
    config = get_pipeline_config()
    model_cfg = config["model"]
    test_size = float(model_cfg.get("test_size", 0.25))
    cal_size = float(model_cfg.get("cal_size", 0.15))
    unc_cfg = model_cfg.get("uncertainty", {})
    level = float(unc_cfg.get("level", 0.9))
    uncertainty_on = bool(unc_cfg.get("enabled", True))

    X, y, feats = make_xy(engineered, config)
    time_index = engineered.loc[X.index, schemas.DATE_COLUMN].to_numpy()
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = temporal_three_way_split(
        X, y, time_index, test_size=test_size, cal_size=cal_size
    )

    primary = build_model(config=config)
    primary.fit(X_tr, y_tr)
    metrics = primary.evaluate(X_te, y_te)
    metrics["n_train"] = int(len(X_tr))
    metrics["n_cal"] = int(len(X_cal))
    metrics["n_test"] = int(len(X_te))
    metrics["n_features"] = len(feats)

    extra: dict = {}
    preds_te = primary.predict(X_te)
    if uncertainty_on:
        # Conformal half-width from residuals the model never trained on.
        halfwidth = conformal_quantile(y_cal.to_numpy() - primary.predict(X_cal), level)
        metrics.update(interval_metrics(y_te.to_numpy(), preds_te, halfwidth))
        metrics["interval_level"] = level
        extra["interval_level"] = level
        extra["interval_halfwidth"] = halfwidth

    rel = reliability_table(y_te.to_numpy(), preds_te)
    extra["calibration_error"] = float(rel.attrs["calibration_error"])
    extra["reliability"] = rel.to_dict("records")

    # --- forecast-horizon degradation (dedicated refits per lead time) ------
    dates_tr = pd.to_datetime(engineered.loc[X_tr.index, schemas.DATE_COLUMN])
    dates_te = pd.to_datetime(engineered.loc[X_te.index, schemas.DATE_COLUMN])
    if model_cfg.get("horizons"):
        extra["horizon_metrics"] = horizon_report(
            engineered, config,
            model_factory=lambda: build_model(config=config),
            train_end=dates_tr.max(), test_start=dates_te.min(),
        )

    # --- side-by-side comparison on the identical split ---------------------
    compare_names = [str(n) for n in model_cfg.get("compare", [primary.name])]
    if primary.name not in compare_names:
        compare_names.insert(0, primary.name)
    comparison = []
    prov = _master_provenance(engineered)
    note = _sources_note(prov) if prov else schemas.SYNTHETIC_NOTE
    for name in compare_names:
        model = primary if name == primary.name else build_model(name, config=config)
        if model is not primary:
            model.fit(X_tr, y_tr)
        row = {"name": model.name, **model.evaluate(X_te, y_te)}
        if uncertainty_on:
            hw = conformal_quantile(
                y_cal.to_numpy() - model.predict(X_cal), level
            )
            row.update(interval_metrics(y_te.to_numpy(), model.predict(X_te), hw))
            row["interval_halfwidth"] = hw
        row["n_features"] = len(feats)
        comparison.append(row)
        path = model.save(get_settings().models_dir / f"baseline_{model.name}.joblib")
        LOGGER.info("Model '%s' saved to %s; test metrics=%s", model.name, path, row)

    # --- held-out predictions (+ interval) for the dashboard ----------------
    pred_frame = engineered.loc[X_te.index, [
        "county_code", schemas.COUNTY_NAME_COLUMN, schemas.DATE_COLUMN,
    ]].copy()
    lo, hi = (add_intervals(preds_te, extra.get("interval_halfwidth", 0.0))
              if uncertainty_on else (preds_te, preds_te))
    pred_frame["y_true"] = y_te.to_numpy()
    pred_frame["y_pred"] = preds_te
    pred_frame["y_lo"] = lo
    pred_frame["y_hi"] = hi
    pred_frame = pred_frame.sort_values(schemas.DATE_COLUMN)
    write_csv(pred_frame, get_settings().models_dir / "test_predictions.csv")

    card = primary.model_card(metrics)
    card.data_note = note
    extra["comparison"] = comparison
    extra["split"] = "chronological train/calibration/test (no shuffling)"
    if uncertainty_on:
        extra["interval_note"] = (
            f"Conformal {int(level * 100)}% intervals from calibration-block "
            "residuals; under a temporal split the exchangeability assumption "
            "is an approximation - treat coverage as indicative, not a "
            "guarantee."
        )
    card.extra = extra
    card_path = get_settings().models_dir / "model_card.json"
    card_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
    (get_settings().models_dir / "comparison.json").write_text(
        json.dumps({"primary": primary.name, "data_note": note, "rows": comparison},
                   indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Primary model '%s'; metrics=%s", primary.name, metrics)
    return primary, metrics, card


def run_pipeline(
    force_raw: bool = False,
    seed: int = 42,
    source_overrides: list[str] | None = None,
    set_overrides: list[str] | None = None,
) -> dict:
    """Run the full pipeline and return a summary dict."""
    setup_logging()
    if source_overrides:
        _apply_source_overrides(source_overrides)
    if set_overrides:
        _apply_set_overrides(set_overrides)
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
        "dataset_sources": _master_provenance(engineered),
        "metrics": metrics,
    }
    LOGGER.info("=== Pipeline complete: %s ===", summary["metrics"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build AgriRisk data + baseline model.")
    parser.add_argument("--force-raw", action="store_true",
                        help="Re-ingest raw datasets even if present.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic data RNG seed.")
    parser.add_argument("--source", action="append", default=[],
                        metavar="DATASET=PROVIDER",
                        help="Override a dataset source for this run, e.g. "
                             "--source climate=chirps (repeatable).")
    parser.add_argument("--set", dest="set_overrides", action="append", default=[],
                        metavar="DOTTED.KEY=VALUE",
                        help="Override any config path for this run, e.g. "
                             "--set external.openmeteo.enabled=true (repeatable).")
    args = parser.parse_args(argv)
    run_pipeline(force_raw=args.force_raw, seed=args.seed,
                 source_overrides=args.source, set_overrides=args.set_overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
