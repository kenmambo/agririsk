"""FastAPI service exposing the AgriRisk risk panel + model card.

Thin view over on-disk pipeline artefacts (same rule as the dashboard): no
ingestion, no validation, no training behind these routes. Every response
that carries data also carries provenance, and the honesty contract holds -
if the loaded panel contains synthetic components the API says so explicitly
rather than presenting numbers as observed.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

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


# Human-readable landing page, served to browsers (Accept: text/html); JSON
# clients keep getting the machine-readable index. Provenance is fetched live
# so the page can never claim "real feeds" from a stale bundle.
_ENDPOINTS = [
    ("/health", "artefact presence + pipeline version"),
    ("/data/status", "rows / counties / period + per-dataset provenance"),
    ("/counties", "county registry reference (codes, centroids, zones)"),
    ("/panel", "county-month rows (filters: county_code, year_from/to, agro_zone)"),
    ("/risk/summary?date=YYYY-MM", "per-county risk + display band for a month"),
    ("/model/card", "metrics, uncertainty, calibration, horizons"),
    ("/model/comparison", "ridge vs gbm on the identical chronological split"),
    ("/model/predictions", "held-out predictions with conformal intervals"),
]


def _landing_html() -> HTMLResponse:
    try:
        df = artifacts.features_panel()
        rows, counties = len(df), df["county_code"].nunique()
        if artifacts.data_is_synthetic(df):
            badge = '<span class="badge warn">SYNTHETIC DATA</span>'
        else:
            badge = '<span class="badge ok">real observed feeds</span>'
    except artifacts.ArtifactMissing:
        rows = counties = None
        badge = '<span class="badge warn">no artefacts yet</span>'
    links = "\n".join(
        f'      <li><a href="{path}"><code>{path}</code></a> — {desc}</li>'
        for path, desc in _ENDPOINTS
    )
    stats = (
        f"<p>{rows:,} county-month rows &middot; {counties} counties &middot; "
        "read-only view over artefacts written by <code>python -m agrik</code> "
        "on the host.</p>"
        if rows
        else "<p>Run the pipeline to generate artefacts.</p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgriRisk Kenya API</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto;
         padding: 0 1rem; color: #1e2a22; line-height: 1.55; }}
  h1 {{ font-size: 1.6rem; }}
  .badge {{ padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.8rem;
            vertical-align: middle; }}
  .badge.ok {{ background: #e3f4e6; color: #1b5e20; }}
  .badge.warn {{ background: #fdecea; color: #b71c1c; }}
  ul {{ padding-left: 0; list-style: none; }}
  li {{ margin: 0.45rem 0; }}
  code {{ background: #f2f4f1; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  a {{ color: #1b6e3c; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .note {{ color: #5c6b60; font-size: 0.9rem; }}
</style>
</head>
<body>
  <h1>AgriRisk Kenya &middot; API {badge}</h1>
  {stats}
  <p>Interactive docs: <a href="/docs"><code>/docs</code></a> (Try-it UI).
     The decision-support dashboard runs separately on port 8502.</p>
  <h2>Endpoints</h2>
  <ul>
{links}
  </ul>
  <p class="note">Risk bands are documented display thresholds, not model
  output. Every data response echoes its provenance; missing artefacts return
  503 with a hint, never a fabricated number.</p>
</body>
</html>"""
    return HTMLResponse(html)


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
    def root(request: Request) -> Any:
        # Browsers get the readable landing page; API clients get JSON.
        if "text/html" in request.headers.get("accept", ""):
            return _landing_html()
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
