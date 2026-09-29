"""FastAPI service exposing the AgriRisk risk panel + model card.

Thin view over on-disk pipeline artefacts (same rule as the dashboard): no
ingestion, no validation, no training behind these routes. Every response
that carries data also carries provenance, and the honesty contract holds -
if the loaded panel contains synthetic components the API says so explicitly
rather than presenting numbers as observed.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .. import schemas
from ..logging import get_logger
from ..settings import get_settings
from . import artifacts

LOGGER = get_logger("api.app")

# Columns the panel endpoint returns by default (raw drivers + target, not the
# full engineered matrix - keeps responses readable; ?all_columns=true opts in).
_CORE_COLUMNS = [
    "county_code", schemas.COUNTY_NAME_COLUMN, "year", "month",
    schemas.DATE_COLUMN, "agro_zone",
    *schemas.BASE_COVARIATES, schemas.TARGET,
    "data_source",
]


def _artifact_or_503(loader: Any) -> Any:
    try:
        return loader()
    except artifacts.ArtifactMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def create_app() -> FastAPI:
    """Build the AgriRisk API app (fresh instance per server process)."""
    app = FastAPI(
        title="AgriRisk Kenya API",
        version="0.4.0",
        description=(
            "County x month food-security risk panel with per-row provenance. "
            "Read-only over artefacts written by `python -m agrik`."
        ),
    )

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service": "AgriRisk Kenya API",
            "docs": "/docs",
            "endpoints": [
                "/health", "/data/status", "/counties", "/panel",
                "/risk/summary", "/model/card", "/model/comparison",
                "/model/predictions",
            ],
        }

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        s = get_settings()
        present = {
            "features_panel": (s.features_dir / "features_panel.csv").exists(),
            "model_card": (s.models_dir / "model_card.json").exists(),
            "comparison": (s.models_dir / "comparison.json").exists(),
            "test_predictions": (s.models_dir / "test_predictions.csv").exists(),
        }
        version = None
        try:
            version = _artifact_or_503(artifacts.features_manifest)[
                "pipeline_version"
            ]
        except HTTPException:
            pass
        return {"status": "ok" if present["features_panel"] else "no-artefacts",
                "pipeline_version": version, "artefacts": present}

    @app.get("/data/status", tags=["meta"])
    def data_status() -> dict:
        df = _artifact_or_503(artifacts.features_panel)
        sources = artifacts.dataset_sources(df)
        return {
            "rows": int(len(df)),
            "counties": int(df["county_code"].nunique()),
            "period": {"from": str(df[schemas.DATE_COLUMN].min()),
                       "to": str(df[schemas.DATE_COLUMN].max())},
            "dataset_sources": sources,
            "data_is_synthetic": artifacts.data_is_synthetic(df),
            "note": (
                schemas.SYNTHETIC_NOTE
                if artifacts.data_is_synthetic(df)
                else "All datasets are real feeds."
            ),
        }

    @app.get("/counties", tags=["reference"])
    def counties_list() -> list[dict]:
        return artifacts.records(artifacts.county_reference())

    @app.get("/panel", tags=["panel"])
    def panel(
        county_code: str | None = Query(None, description="3-digit code, e.g. 015"),
        year_from: int | None = None,
        year_to: int | None = None,
        agro_zone: str | None = None,
        limit: int = Query(50, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        all_columns: bool = False,
    ) -> dict:
        df = _artifact_or_503(artifacts.features_panel)
        if county_code is not None:
            code = county_code.zfill(3)
            known = set(artifacts.county_reference()["county_code"])
            if code not in known:
                raise HTTPException(404, detail=f"Unknown county_code {code!r}.")
            df = df[df["county_code"] == code]
        if year_from is not None:
            df = df[df["year"] >= year_from]
        if year_to is not None:
            df = df[df["year"] <= year_to]
        if agro_zone is not None and "agro_zone" in df.columns:
            df = df[df["agro_zone"].str.lower() == agro_zone.lower()]
        cols = [c for c in _CORE_COLUMNS if c in df.columns]
        total = int(len(df))
        page = df.sort_values([schemas.DATE_COLUMN, "county_code"])
        page = page.iloc[offset: offset + limit]
        return {
            "total": total, "limit": limit, "offset": offset,
            "data_is_synthetic": artifacts.data_is_synthetic(
                _artifact_or_503(artifacts.features_panel)
            ),
            "rows": artifacts.records(page if all_columns else page[cols]),
        }

    @app.get("/risk/summary", tags=["panel"])
    def risk_summary(
        date: str | None = Query(
            None, pattern=r"^\d{4}-\d{2}(-\d{2})?$",
            description="Month ISO prefix, e.g. 2022-06. Default: latest."),
        county_code: str | None = None,
    ) -> dict:
        df = _artifact_or_503(artifacts.features_panel)
        if county_code is not None:
            code = county_code.zfill(3)
            known = set(artifacts.county_reference()["county_code"])
            if code not in known:
                raise HTTPException(404, detail=f"Unknown county_code {code!r}.")
            df = df[df["county_code"] == code]
        if date is not None:
            sel = df[schemas.DATE_COLUMN].astype(str).str.startswith(date)
            df = df[sel]
        if df.empty:
            raise HTTPException(404, detail="No rows match the given filters.")
        agg = (
            df.groupby(["county_code", schemas.COUNTY_NAME_COLUMN], as_index=False)
            [schemas.TARGET].mean()
            .rename(columns={schemas.TARGET: "risk_index"})
        )
        agg["band"] = agg["risk_index"].round(1).map(schemas.risk_band)
        agg = agg.sort_values("risk_index", ascending=False)
        return {
            "as_of": date or str(df[schemas.DATE_COLUMN].max()),
            "bands_are": "documented display thresholds, not model output",
            "data_is_synthetic": artifacts.data_is_synthetic(
                _artifact_or_503(artifacts.features_panel)
            ),
            "counties": artifacts.records(agg),
        }

    @app.get("/model/card", tags=["model"])
    def model_card() -> dict:
        return _artifact_or_503(artifacts.model_card)

    @app.get("/model/comparison", tags=["model"])
    def model_comparison() -> dict:
        return _artifact_or_503(artifacts.comparison)

    @app.get("/model/predictions", tags=["model"])
    def predictions(
        county_code: str | None = None,
        limit: int = Query(100, ge=1, le=5000),
        offset: int = Query(0, ge=0),
    ) -> dict:
        df = _artifact_or_503(artifacts.test_predictions)
        if county_code is not None:
            df = df[df["county_code"] == county_code.zfill(3)]
        total = int(len(df))
        page = df.iloc[offset: offset + limit]
        return {"total": total, "limit": limit, "offset": offset,
                "rows": artifacts.records(page)}

    return app
