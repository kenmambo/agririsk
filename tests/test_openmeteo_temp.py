"""Offline tests for the Open-Meteo ERA5 temperature enrichment.

No network: the per-county JSON responses are pre-seeded into the raw
download cache (the same path ``download_cached`` would write), and the
provider enrichment is exercised through ``ingest_datasets``.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from agrik import counties, schemas
from agrik.config import get_pipeline_config, load_pipeline_config
from agrik.ingestion import openmeteo, providers
from agrik.ingestion.base import ExternalDataError

SY, EY = 2019, 2022


@pytest.fixture()
def cfg():
    c = deepcopy(load_pipeline_config())
    c["panel"]["start_year"] = SY
    c["panel"]["end_year"] = EY
    c["external"]["openmeteo"] = {
        "base_url": "https://invalid.invalid/archive",
        "enabled": True,
    }
    return c


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    from agrik.settings import get_settings

    get_settings.cache_clear()
    yield tmp_path / "data" / "raw" / "external"
    get_settings.cache_clear()


def _daily_payload(templ: float = 20.0) -> str:
    """4 full years of daily temps; month = templ + month_index/4 (distinct
    monthly means that stay inside Kenya's plausible climate)."""
    times, vals = [], []
    for y in range(SY, EY + 1):
        for m in range(1, 13):
            for d in (1, 15, 28):
                times.append(f"{y}-{m:02d}-{d:02d}")
                # one deliberate null day (Jan-15 of first year) -> dropped
                v = None if (y == SY and m == 1 and d == 15) else templ + (m - 1) / 4
                vals.append(v)
    return json.dumps({"daily": {"time": times, "temperature_2m_mean": vals}})


def _seed_county(cache_root, county, templ: float = 20.0) -> None:
    lat, lon = round(county.lat, 3), round(county.lon, 3)
    p = cache_root / "openmeteo" / f"temps_{lat}_{lon}_{SY}_{EY}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_daily_payload(templ), encoding="utf-8")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------
def test_api_url_shape():
    url = openmeteo.api_url("https://x/archive", 2.34, 37.99, 2019, 2022)
    assert url.startswith("https://x/archive?latitude=2.34&longitude=37.99")
    assert "start_date=2019-01-01" in url and "daily=temperature_2m_mean" in url


def test_parse_daily_temps_drops_nulls():
    payload = {"daily": {"time": ["2019-01-01", "2019-01-02", "2019-02-01"],
                         "temperature_2m_mean": [25.0, None, 27.0]}}
    df = openmeteo.parse_daily_temps(payload)
    assert len(df) == 2  # null day dropped
    assert df["m"].tolist() == ["01", "02"]


def test_parse_daily_temps_all_null_raises():
    with pytest.raises(ExternalDataError, match="no non-null"):
        openmeteo.parse_daily_temps({"daily": {"time": ["2019-01-01"],
                                               "temperature_2m_mean": [None]}})
    with pytest.raises(ExternalDataError, match="no non-null"):
        openmeteo.parse_daily_temps({})


# ---------------------------------------------------------------------------
# panel builder (pre-seeded cache)
# ---------------------------------------------------------------------------
def test_build_temperature_panel_all_counties(cfg, cache_root):
    registry = counties.all_counties()
    for i, c in enumerate(registry):
        _seed_county(cache_root, c, templ=20.0 + i / 10)
    df = openmeteo.build_temperature_panel(config=cfg)

    assert len(df) == 21 * 48
    assert df[openmeteo.TEMPERATURE_COLUMN].between(5, 40).all()
    m1 = df[(df["county_code"] == registry[0].code) & (df["month"] == 1)]
    m2 = df[(df["county_code"] == registry[0].code) & (df["month"] == 2)]
    # Jan mean over {20, 20} (null day dropped) vs Feb 20.25 -> means exact.
    assert m1[openmeteo.TEMPERATURE_COLUMN].unique().tolist() == [20.0]
    assert m2[openmeteo.TEMPERATURE_COLUMN].unique().tolist() == [20.25]


def test_build_temperature_panel_partial_failure(cfg, cache_root):
    registry = counties.all_counties()
    for c in registry[:-1]:
        _seed_county(cache_root, c)
    # One county has no cached response and the URL is unreachable -> skipped
    # loudly, others still return (NULL policy, not all-or-nothing).
    df = openmeteo.build_temperature_panel(config=cfg)
    assert df["county_code"].nunique() == 20
    assert registry[-1].code not in set(df["county_code"])


def test_build_temperature_panel_total_failure_raises(cfg, cache_root):
    with pytest.raises(ExternalDataError, match="no county"):
        openmeteo.build_temperature_panel(config=cfg)


# ---------------------------------------------------------------------------
# merge semantics (never faked, honest provenance)
# ---------------------------------------------------------------------------
def _climate_frame() -> pd.DataFrame:
    rows = [
        {"county_code": "015", "county_name": "Marsabit", "year": 2019, "month": 1,
         "date": "2019-01-01", "precipitation_mm": 12.0,
         schemas.PROVENANCE_COLUMN: "chirps"},
        {"county_code": "015", "county_name": "Marsabit", "year": 2019, "month": 2,
         "date": "2019-02-01", "precipitation_mm": 30.0,
         schemas.PROVENANCE_COLUMN: "chirps"},
        {"county_code": "006", "county_name": "Isiolo", "year": 2019, "month": 1,
         "date": "2019-01-01", "precipitation_mm": 8.0,
         schemas.PROVENANCE_COLUMN: "chirps"},
    ]
    return pd.DataFrame(rows)


def test_merge_stamps_only_matched_rows():
    temps = pd.DataFrame([
        {"county_code": "015", "year": 2019, "month": 1, "temp_mean_c": 24.5},
        {"county_code": "015", "year": 2019, "month": 2, "temp_mean_c": 26.0},
    ])
    out = openmeteo.merge_temperature(_climate_frame(), temps)
    assert out["temp_mean_c"].isna().tolist() == [False, False, True]  # Isiolo not faked
    assert out.loc[:1, "temp_mean_c"].tolist() == [24.5, 26.0]
    src = out[schemas.PROVENANCE_COLUMN].tolist()
    assert src[0] == src[1] == openmeteo.COMBINED_SOURCE
    assert src[2] == "chirps"  # untouched row keeps its own provenance


def test_merge_is_idempotent():
    temps = pd.DataFrame([
        {"county_code": "015", "year": 2019, "month": 1, "temp_mean_c": 24.5},
    ])
    once = openmeteo.merge_temperature(_climate_frame(), temps)
    twice = openmeteo.merge_temperature(once, temps)
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------------------------------
# provider-level enrichment through ingest_datasets
# ---------------------------------------------------------------------------
def test_ingest_enriches_chirps_frame(cfg, cache_root, monkeypatch):
    """Real-style CHIRPS frame (no temp column) gains honest temp + stamp."""
    for i, c in enumerate(counties.all_counties()):
        _seed_county(cache_root, c, templ=18.0 + i / 10)
    chirps_only = _climate_frame()
    monkeypatch.setattr(
        providers.chirps, "build_climate_panel",
        lambda config=None, settings=None: chirps_only.copy(),
    )
    cfg = deepcopy(cfg)
    cfg["datasets"]["climate"]["source"] = "chirps"
    frames = providers.ingest_datasets(config=cfg)
    climate = frames["climate"]
    assert "temp_mean_c" in climate.columns
    matched = climate["county_code"] == "015"
    assert climate.loc[matched, "temp_mean_c"].notna().all()
    assert climate.loc[matched, schemas.PROVENANCE_COLUMN].eq(
        openmeteo.COMBINED_SOURCE
    ).all()
    # Isiolo had no seeded series match? it did - so force-check the stamp rule
    # via the merge unit test instead of here (all 21 counties have temps).
    assert climate["temp_mean_c"].between(5, 40).all()


def test_ingest_does_not_restamp_synthetic_climate(cfg, cache_root):
    """Synthetic climate already HAS temp_mean_c - Open-Meteo must not claim
    attribution for values it did not provide (merge is a honest no-op)."""
    for i, c in enumerate(counties.all_counties()):
        _seed_county(cache_root, c, templ=18.0 + i / 10)
    frames = providers.ingest_datasets(config=cfg)
    climate = frames["climate"]
    assert climate[schemas.PROVENANCE_COLUMN].eq("synthetic").all()
    assert openmeteo.COMBINED_SOURCE not in set(climate[schemas.PROVENANCE_COLUMN])


def test_ingest_outage_keeps_climate_unfaked(cfg, cache_root, monkeypatch):
    def boom(config=None, settings=None):
        raise ExternalDataError("simulated Open-Meteo outage")

    monkeypatch.setattr(providers.chirps, "build_climate_panel",
                        lambda config=None, settings=None: _climate_frame())
    monkeypatch.setattr(openmeteo, "build_temperature_panel", boom)
    cfg = deepcopy(cfg)
    cfg["datasets"]["climate"]["source"] = "chirps"
    frames = providers.ingest_datasets(config=cfg)
    # Enrichment failed -> the CHIRPS frame comes back exactly as CHIRPS made
    # it: no temp column injected, no fake values, provenance untouched.
    assert "temp_mean_c" not in frames["climate"].columns
    assert frames["climate"][schemas.PROVENANCE_COLUMN].eq("chirps").all()


def test_ingest_skips_when_disabled(cache_root):
    c = deepcopy(load_pipeline_config())
    c["external"]["openmeteo"]["enabled"] = False
    frames = providers.ingest_datasets(config=c)
    assert openmeteo.COMBINED_SOURCE not in set(
        frames["climate"][schemas.PROVENANCE_COLUMN]
    )


# ---------------------------------------------------------------------------
# --set dotted-path CLI override
# ---------------------------------------------------------------------------
def test_set_override_coercion():
    from agrik.pipeline import _apply_set_overrides

    cfg = get_pipeline_config()
    original = deepcopy(cfg["external"]["openmeteo"])
    try:
        _apply_set_overrides([
            "external.openmeteo.enabled=true",
            "external.newsection.some_int=42",
            "external.newsection.text=hello",
        ])
        cfg = get_pipeline_config()
        assert cfg["external"]["openmeteo"]["enabled"] is True
        assert cfg["external"]["newsection"]["some_int"] == 42
        assert cfg["external"]["newsection"]["text"] == "hello"
        with pytest.raises(ValueError, match="--set"):
            _apply_set_overrides(["nodots=true"])
    finally:
        cfg["external"]["openmeteo"] = original
        cfg["external"].pop("newsection", None)
