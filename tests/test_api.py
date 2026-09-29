"""Offline tests for the M4 FastAPI serving layer.

The synthetic pipeline is run into a temp dir, then the API is exercised over
those artefacts with FastAPI's TestClient - no network and no live server.
Covers the honesty contract too: synthetic data must be declared in every
provenance-bearing response, and missing artefacts must produce a helpful 503
instead of fabricated numbers.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agrik import schemas
from agrik.api import create_app
from agrik.pipeline import run_pipeline


@pytest.fixture()
def client_ready(isolated_env):
    run_pipeline(force_raw=True, seed=3)
    return TestClient(create_app())


@pytest.fixture()
def client_empty(isolated_env):
    # Fresh temp dir: the pipeline has never run, no artefacts on disk.
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# meta endpoints
# ---------------------------------------------------------------------------
def test_health_reports_artefacts(client_ready):
    h = client_ready.get("/health").json()
    assert h["status"] == "ok"
    assert h["pipeline_version"]
    assert h["artefacts"]["features_panel"] is True
    assert h["artefacts"]["model_card"] is True


def test_health_without_artefacts_is_honest(client_empty):
    e = client_empty.get("/health").json()
    assert e["status"] == "no-artefacts"
    assert e["pipeline_version"] is None
    assert e["artefacts"]["features_panel"] is False


def test_root_lists_endpoints(client_ready):
    body = client_ready.get("/").json()
    assert "/risk/summary" in body["endpoints"]


def test_missing_artefacts_return_503_with_hint(client_empty):
    for path in ("/panel", "/data/status", "/model/card"):
        r = client_empty.get(path)
        assert r.status_code == 503, path
        assert "python -m agrik" in r.json()["detail"]


# ---------------------------------------------------------------------------
# honesty: provenance is surfaced, never hidden
# ---------------------------------------------------------------------------
def test_data_status_declares_synthetic(client_ready):
    r = client_ready.get("/data/status")
    assert r.status_code == 200
    body = r.json()
    assert body["data_is_synthetic"] is True
    assert "synthetic" in body["note"].lower()
    assert body["counties"] == 21
    assert body["rows"] > 0


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------
def test_panel_pagination_and_filters(client_ready):
    base = client_ready.get("/panel", params={"limit": 5000}).json()
    assert base["total"] == 21 * 4 * 12  # full synthetic county-month grid

    page = client_ready.get("/panel", params={"limit": 10}).json()
    assert len(page["rows"]) == 10 and page["total"] == base["total"]

    tail = client_ready.get("/panel", params={"limit": 10, "offset": 1000}).json()
    assert len(tail["rows"]) == 8  # 1008 total -> last partial page

    years = client_ready.get(
        "/panel", params={"year_from": 2020, "year_to": 2021, "limit": 5000}
    ).json()
    assert years["total"] == 21 * 24

    county = client_ready.get("/panel", params={"county_code": "15"}).json()
    assert all(row["county_code"] == "015" for row in county["rows"])
    assert county["total"] == 48


def test_panel_unknown_county_404(client_ready):
    assert client_ready.get("/panel", params={"county_code": "099"}).status_code == 404


def test_panel_core_columns_by_default(client_ready):
    row = client_ready.get("/panel", params={"limit": 1}).json()["rows"][0]
    assert "precipitation_mm" in row and schemas.TARGET in row
    assert "precip_anom" not in row  # engineered matrix hidden by default
    full = client_ready.get("/panel", params={"limit": 1, "all_columns": "true"}).json()
    assert "precip_anom" in full["rows"][0]


def test_panel_zone_filter(client_ready):
    zones = client_ready.get("/counties").json()
    zone = zones[0]["agro_zone"]
    body = client_ready.get("/panel", params={"agro_zone": zone, "limit": 5000}).json()
    assert body["total"] > 0
    assert all(r["agro_zone"] == zone for r in body["rows"])


# ---------------------------------------------------------------------------
# risk summary + bands
# ---------------------------------------------------------------------------
def test_risk_summary_bands_and_ordering(client_ready):
    body = client_ready.get("/risk/summary", params={"date": "2022-06"}).json()
    assert body["as_of"] == "2022-06"
    assert len(body["counties"]) == 21
    risks = [c["risk_index"] for c in body["counties"]]
    assert risks == sorted(risks, reverse=True)
    valid_bands = {label for _, label in schemas.RISK_BANDS} | {"Below Watch"}
    labels = {c["band"] for c in body["counties"]}
    assert labels <= valid_bands
    assert "display thresholds" in body["bands_are"]


def test_risk_summary_single_county_and_unknown_date(client_ready):
    one = client_ready.get(
        "/risk/summary", params={"county_code": "001"}
    ).json()
    assert len(one["counties"]) == 1
    assert one["counties"][0]["county_code"] == "001"
    bad = client_ready.get("/risk/summary", params={"date": "2030-01"})
    assert bad.status_code == 404


# ---------------------------------------------------------------------------
# model artefacts
# ---------------------------------------------------------------------------
def test_model_card_and_comparison(client_ready):
    card = client_ready.get("/model/card").json()
    assert card["name"] in {"ridge", "gbm"}
    assert card["metrics"]["rmse"] > 0
    assert "synthetic" in card["data_note"].lower()

    comp = client_ready.get("/model/comparison").json()
    assert {r["name"] for r in comp["rows"]} >= {"ridge", "gbm"}


def test_model_predictions_interval_ordering(client_ready):
    body = client_ready.get("/model/predictions", params={"limit": 5000}).json()
    assert body["total"] > 0
    for row in body["rows"]:
        assert row["y_lo"] <= row["y_pred"] <= row["y_hi"]
    mars = client_ready.get(
        "/model/predictions", params={"county_code": "15"}
    ).json()
    assert all(r["county_code"] == "015" for r in mars["rows"])
