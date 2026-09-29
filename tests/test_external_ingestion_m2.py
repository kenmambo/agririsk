"""Offline tests for the M2 external-feed layer.

FEWS NET (market + outcome), HDX (socioeconomic) and World Bank
(agriculture) connectors are exercised with fixture files PRE-SEEDED into
the raw-download cache, so ``download_cached`` always hits the cache and
no test ever touches the network.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from agrik import schemas
from agrik.config import load_pipeline_config
from agrik.ingestion import fewsnet, hdx, worldbank
from agrik.ingestion.base import ExternalDataError

# Two real registry counties plus one name that must NOT join.
COUNTY_A = "Marsabit"   # code 015
COUNTY_B = "Isiolo"     # code 006


@pytest.fixture()
def cfg():
    """Panel config pinned to a single year, with dummy external URLs
    (the cache is pre-seeded, so the URLs are never fetched)."""
    cfg = deepcopy(load_pipeline_config())
    cfg["panel"]["start_year"] = 2019
    cfg["panel"]["end_year"] = 2019
    cfg["external"]["fewsnet"] = {
        "base_url": "https://invalid.invalid/api",
        "iso3": "KE",
        "product": "maize",
        "price_type": "Retail",
    }
    cfg["external"]["hdx"] = {
        "mpi_url": "https://invalid.invalid/ken_mpi.csv",
        "rural_population_url": "https://invalid.invalid/ken_adm2_rural_population.csv",
    }
    cfg["external"]["worldbank"] = {
        "api_url": "https://invalid.invalid/wb",
        "indicator": "AG.PRD.FOOD.XD",
    }
    return cfg


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    """Point AGRIK_DATA_ROOT at tmp_path and return the external cache dir."""
    monkeypatch.setenv("AGRIK_DATA_ROOT", str(tmp_path / "data"))
    from agrik.settings import get_settings

    get_settings.cache_clear()
    root = tmp_path / "data" / "raw" / "external"
    yield root
    get_settings.cache_clear()


def _seed(rel: str, content: str, cache_root) -> None:
    path = cache_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# FEWS NET pure helpers
# ---------------------------------------------------------------------------
def test_county_from_unit_parses_admin_parent():
    # County as the administrative parent of a named zone (LHZ style).
    assert (
        fewsnet.county_from_unit("Northern Pastoral Zone, Marsabit, Eastern, Kenya")
        == "Marsabit"
    )
    # County-named unit: the county sits one token closer to the end.
    assert fewsnet.county_from_unit("Marsabit, Eastern, Kenya") == "Marsabit"
    # Registry-name token wins regardless of position; no match -> "" (never guessed).
    assert fewsnet.county_from_unit("Baltimore, Marsabit, Eastern, Kenya") == "Marsabit"
    assert fewsnet.county_from_unit("Kenya") == ""
    assert fewsnet.county_from_unit("Refugee Camp, Dadaab, Kenya") == ""


def test_norm_handles_registry_hyphenation():
    assert fewsnet._norm("Tharaka Nithi") == fewsnet._norm("Tharaka-Nithi")
    assert fewsnet._norm("  Nyandarwa. ") == "nyandarwa"


def test_phase_to_risk_documented_rescale():
    assert [fewsnet.phase_to_risk(p) for p in (1, 2, 3, 4, 5)] == [0.0, 25.0, 50.0, 75.0, 100.0]
    with pytest.raises(ValueError):
        fewsnet.phase_to_risk(6)


# ---------------------------------------------------------------------------
# FEWS NET market panel (fixture CSV in cache)
# ---------------------------------------------------------------------------
def _seed_prices(cache_root):
    df = pd.DataFrame(
        {
            "period_date": ["2019-01-05", "2019-01-20", "2019-02-08", "2019-01-10",
                            "2019-01-11", "2019-01-12", "2019-01-15", "2018-01-10"],
            "product": ["Maize Grain (White)"] * 3 + ["Maize Flour"]
                       + ["Sorghum White", "Maize Grain (White)", "Maize Grain (White)",
                          "Maize Grain (White)"],
            "unit": ["kg", "kg", "kg", "kg", "kg", "2_kg", "kg", "kg"],
            "currency": ["KES"] * 5 + ["KES", "USD", "KES"],
            "value": [100.0, 110.0, 120.0, 150.0, 99.0, 200.0, 90.0, 80.0],
            "admin_1": [COUNTY_A, COUNTY_A, COUNTY_B, "Nairobi",
                        COUNTY_A, COUNTY_A, COUNTY_A, COUNTY_A],
        }
    )
    _seed("fewsnet/prices_KE_maize_2019_2019.csv", df.to_csv(index=False), cache_root)


def test_market_panel_filters_aggregates_and_labels(cfg, cache_root):
    _seed_prices(cache_root)
    df = fewsnet.build_market_panel(config=cfg)

    # Kept: maize grain only, kg only, KES only, panel years only.
    # Marsabit Jan: mean(100, 110) = 105. Isiolo Feb: 120.
    assert len(df) == 2
    mars = df[df["county_name"] == COUNTY_A].iloc[0]
    assert float(mars["maize_price_kes_kg"]) == 105.0
    assert int(mars["month"]) == 1
    assert df[schemas.PROVENANCE_COLUMN].eq(fewsnet.PRICES_SOURCE).all()
    # Deterministic order: dates ascending within each county block.
    assert df.groupby("county_code")[schemas.DATE_COLUMN].apply(
        lambda s: s.is_monotonic_increasing
    ).all()


def test_market_panel_empty_extract_raises(cfg, cache_root):
    _seed("fewsnet/prices_KE_maize_2019_2019.csv",
          "period_date,product,unit,currency,value,admin_1\n", cache_root)
    with pytest.raises(ExternalDataError):
        fewsnet.build_market_panel(config=cfg)


# ---------------------------------------------------------------------------
# FEWS NET outcome panel (fixture IPC CSV in cache)
# ---------------------------------------------------------------------------
def _seed_ipc(cache_root):
    df = pd.DataFrame(
        {
            "scenario_name": ["Current Situation", "Current Situation", "Current Situation",
                              "Forecast", "Current Situation", "Current Situation"],
            "unit_type": ["fsc_admin_lhz", "fsc_admin_lhz", "fsc_admin",
                          "fsc_admin_lhz", "idp_camp", "fsc_admin_lhz"],
            "value": [3, 2, 4, 5, 5, 3],
            "geographic_unit_full_name": [
                f"South Zone, {COUNTY_A}, Eastern, Kenya",
                f"North Zone, {COUNTY_A}, Eastern, Kenya",
                f"{COUNTY_B}, Eastern, Kenya",
                "Meru, Eastern, Kenya",
                "Refugee Camp, Dadaab, Kenya",
                f"Pastoral Zone, {COUNTY_A}, Eastern, Kenya",
            ],
            "projection_start": ["2019-01-01"] * 6,
            "projection_end": ["2019-02-01", "2019-06-01", "2019-01-01",
                               "2019-01-01", "2019-01-01", "2019-05-01"],
        }
    )
    _seed("fewsnet/ipcphase_KE_CS.csv", df.to_csv(index=False), cache_root)


def test_outcome_panel_worst_phase_per_county_month(cfg, cache_root):
    _seed_ipc(cache_root)
    df = fewsnet.build_outcome_panel(config=cfg)

    # Marsabit: Jan-Feb worst(3,3)=50, Mar-May worst(3,2)=50, Jun only the
    # phase-2 zone -> 25. Isiolo: phase 4 = 75, Jan only.
    # Excluded: Forecast scenario, idp_camp unit, Meru (only a Forecast row),
    # and months outside 2019.
    risk = {(r.county_name, r.month): r.food_security_risk_index for r in df.itertuples()}
    assert risk[(COUNTY_A, 1)] == 50.0
    assert risk[(COUNTY_A, 4)] == 50.0    # phase-3 pastoral zone still covers Apr
    assert risk[(COUNTY_A, 6)] == 25.0    # only the phase-2 zone reaches Jun
    assert (COUNTY_A, 7) not in risk
    assert risk[(COUNTY_B, 1)] == 75.0
    assert ("Meru", 1) not in risk          # Forecast-only row excluded
    mars = df[df["county_name"] == COUNTY_A]
    assert set(df[schemas.PROVENANCE_COLUMN].unique()) == {fewsnet.IPC_SOURCE}
    assert "ipc_crisis_households" not in df.columns  # omitted, not faked
    assert df["food_security_risk_index"].between(0, 100).all()
    assert len(mars) == 6  # Jan..Jun


def test_outcome_panel_no_match_raises(cfg, cache_root):
    df = pd.DataFrame(
        {"scenario_name": ["Current Situation"], "unit_type": ["fsc_admin_lhz"],
         "value": [3], "geographic_unit_full_name": ["Some Zone, Antarctica, Kenya"],
         "projection_start": ["2019-01-01"], "projection_end": ["2019-06-01"]}
    )
    _seed("fewsnet/ipcphase_KE_CS.csv", df.to_csv(index=False), cache_root)
    with pytest.raises(ExternalDataError):
        fewsnet.build_outcome_panel(config=cfg)


# ---------------------------------------------------------------------------
# HDX socioeconomic panel
# ---------------------------------------------------------------------------
def _seed_hdx(cache_root):
    mpi = pd.DataFrame(
        {
            "Admin 1 PCode": ["KE015", "KE006", "KE999", None],
            "Admin 1 Name": [COUNTY_A, COUNTY_B, "Somewhere", "National"],
            "MPI": [0.2, 0.1, 0.3, 0.15],
            "Headcount Ratio": [66.5, 32.1, 10.0, float("nan")],
        }
    )
    rural = pd.DataFrame(
        {
            "ADM_PCODE": ["KE015101", "KE015102", "KE006101", "KE006101", "KE099101"],
            "total_pop_rural": [100000.0, 50000.0, 20000.0, 1000.0, 7.0],
        }
    )
    _seed("hdx/ken_mpi.csv", mpi.to_csv(index=False), cache_root)
    _seed("hdx/ken_adm2_rural_population.csv", rural.to_csv(index=False), cache_root)


def test_hdx_helper_conversions(cfg, cache_root):
    _seed_hdx(cache_root)
    root = cache_root / "hdx"
    registry = hdx._county_by_norm()
    pov = hdx.poverty_by_county(root / "ken_mpi.csv", registry)
    rpop = hdx.rural_pop_by_county(root / "ken_adm2_rural_population.csv", registry)
    assert pov["015"] == pytest.approx(0.665)   # percent -> 0-1 fraction
    assert rpop["006"] == 21000                  # ADM2 rows summed, incl. dupe
    assert "999" not in pov and "999" not in rpop  # unknown pcode/name dropped


def test_socioeconomic_panel_broadcasts_usable_counties(cfg, cache_root):
    _seed_hdx(cache_root)
    df = hdx.build_socioeconomic_panel(config=cfg)

    # Only the 2 counties present in BOTH sources; 12 months each.
    assert len(df) == 2 * 12
    assert set(df["county_code"]) == {"015", "006"}
    assert df["poverty_rate"].between(0, 1).all()
    assert df[df["county_code"] == "015"]["rural_pop"].eq(150000).all()
    assert "coping_capacity_index" not in df.columns  # omitted, not faked
    assert df[schemas.PROVENANCE_COLUMN].eq(hdx.SOURCE_NAME).all()


def test_socioeconomic_panel_join_failure_raises(cfg, cache_root):
    _seed("hdx/ken_mpi.csv", "Admin 1 PCode,Admin 1 Name,Headcount Ratio\n", cache_root)
    _seed("hdx/ken_adm2_rural_population.csv",
          "ADM_PCODE,total_pop_rural\nKE015101,10\n", cache_root)
    with pytest.raises(ExternalDataError, match="join failed"):
        hdx.build_socioeconomic_panel(config=cfg)


# ---------------------------------------------------------------------------
# World Bank agriculture panel
# ---------------------------------------------------------------------------
def _wb_payload():
    return [
        [{"name": "Last updated", "value": "2024-01-01"}, {"name": "Country", "value": "Kenya"}],
        [
            {"date": "2019", "value": 101.234},
            {"date": "2018", "value": 95.0},   # outside the pinned panel window
            {"date": "2020", "value": None},   # unpublished -> dropped
        ],
    ]


def test_parse_wb_indicator_drops_nulls():
    assert worldbank.parse_wb_indicator(_wb_payload()) == {2019: 101.234, 2018: 95.0}
    with pytest.raises(ExternalDataError):
        worldbank.parse_wb_indicator([[{}], [{"date": "2019", "value": None}]])
    with pytest.raises(ExternalDataError):
        worldbank.parse_wb_indicator([[]])


def test_agriculture_panel_broadcasts_index(cfg, cache_root):
    _seed("worldbank/KEN_AG_PRD_FOOD_XD.json", json.dumps(_wb_payload()), cache_root)
    df = worldbank.build_agriculture_panel(config=cfg)

    # 21 counties x 12 months of the one indexed year, rounded to 2 dp.
    assert len(df) == 21 * 12
    assert df["prod_index"].eq(101.23).all()
    assert set(df["year"]) == {2019}
    assert df[schemas.PROVENANCE_COLUMN].eq(worldbank.SOURCE_NAME).all()


def test_agriculture_panel_no_published_year_raises(cfg, cache_root):
    payload = [[{}], [{"date": "2015", "value": 90.0}]]
    _seed("worldbank/KEN_AG_PRD_FOOD_XD.json", json.dumps(payload), cache_root)
    with pytest.raises(ExternalDataError, match="nothing published"):
        worldbank.build_agriculture_panel(config=cfg)
