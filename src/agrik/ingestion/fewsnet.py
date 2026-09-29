"""FEWS NET Data Warehouse (FDW) connector - REAL data.

Serves two schema datasets from USDA FEWS NET's public REST API:

* ``market``    - county-level monthly white-maize *retail* prices
  (KES/kg) from ``marketpricefacts`` (admin_1 = Kenyan county).
* ``outcome``   - county-level IPC Acute Food Insecurity phases from
  ``ipcphase`` (Current Situation). FEWS NET analyses food-security
  zones; the county is the registry-name token inside each zone's full
  name (``"Zone, <County>, Region, Kenya"`` or ``"<County>, Region,
  Kenya"``), found by scanning tokens right-to-left. Where a county has
  several zones in the same window the WORST (highest) phase wins - the
  conservative early-warning choice.

Documented derivations (never silent):
* ``food_security_risk_index`` = (IPC phase - 1) * 25  -> {0,25,50,75,100}
  for phases 1-5. This is a transparent linear rescale of an expert
  classification, NOT a model output.
* ``ipc_crisis_households`` is NOT produced (county-level crisis
  household counts are not published in the FEWS NET time-series API -
  only national ones). The column stays absent; the pipeline logs it.

Provenance: rows carry ``data_source`` = ``fewsnet_prices`` / ``fewsnet_ipc``.
CSV extracts are cached under ``data/raw/external/fewsnet/`` so re-runs
are offline.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .utils import download_cached

LOGGER = get_logger("ingestion.fewsnet")

PRICES_SOURCE = "fewsnet_prices"
IPC_SOURCE = "fewsnet_ipc"


def _norm(name: str) -> str:
    """Normalised county key: lowercase, hyphen->space, no punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", str(name).replace("-", " ").lower())).strip()


def county_from_unit(full_name: str, registry: dict[str, Any] | None = None) -> str:
    """County name from a FEWS NET unit full name, matched to the registry.

    Tokens are scanned right-to-left (skipping the trailing country) and
    the first token that names a registry county wins:
    ``"Northern Pastoral Zone, Marsabit, Eastern, Kenya"`` -> ``"Marsabit"``;
    ``"Moyale, Marsabit, Eastern, Kenya"`` -> ``"Marsabit"`` (the LHZ token
    is not a county). Returns "" when no token matches - the county is
    never guessed.
    """
    registry = registry if registry is not None else _registry_by_name()
    parts = [p.strip() for p in str(full_name).split(",")]
    for token in reversed(parts):
        if _norm(token) in registry:
            return token
    return ""


def _registry_by_name() -> dict[str, Any]:
    """Normalised county-name -> :class:`agrik.counties.County`."""
    return {_norm(c.name): c for c in counties.all_counties()}


def phase_to_risk(phase: float) -> float:
    """IPC phase 1-5 -> 0-100 risk index (documented linear rescale)."""
    p = int(phase)
    if p not in (1, 2, 3, 4, 5):
        raise ValueError(f"IPC phase {phase!r} out of range 1-5")
    return float((p - 1) * 25)


def _unit_code(full_name: str, registry: dict[str, Any]) -> str | None:
    """Registry county code for a unit full name (None if no token matches)."""
    county = county_from_unit(full_name, registry)
    entry = registry.get(_norm(county)) if county else None
    return entry.code if entry is not None else None


def _panel_span(config: dict[str, Any]) -> tuple[int, int]:
    panel = config["panel"]
    return int(panel["start_year"]), int(panel["end_year"])


def _fetch_csv(url: str, dest: Path, settings: Settings) -> Path:
    return download_cached(url, dest, timeout=settings.http_timeout_s)


# --- market: county monthly maize retail prices -----------------------------

def build_market_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Real county-monthly maize retail price (KES/kg) from FEWS NET FDW."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["fewsnet"]
    sy, ey = _panel_span(config)
    cache = settings.external_dir / "fewsnet"
    url = (
        f"{ext['base_url']}/marketpricefacts.csv"
        f"?country_code={ext['iso3']}&product={ext['product']}"
        f"&price_type={ext['price_type']}&start_date={sy}-01-01&end_date={ey}-12-31"
    )
    csv = _fetch_csv(url, cache / f"prices_{ext['iso3']}_{ext['product']}_{sy}_{ey}.csv", settings)
    df = pd.read_csv(csv, encoding="utf-8-sig")
    if df.empty:
        raise ExternalDataError("FEWS NET prices: empty extract for the panel window.")

    df["product"] = df["product"].astype(str)
    maize = df[df["product"].str.contains("maize", case=False, na=False)]
    grain = maize[maize["product"].str.contains("grain", case=False, na=False)]
    use = grain if not grain.empty else maize  # staple grain, not flour/meal
    use = use[
        (use["unit"].astype(str).str.lower() == "kg")
        & (use["currency"].astype(str).str.upper() == "KES")
        & (pd.to_numeric(use["value"], errors="coerce") > 0)
    ].copy()

    reg = _registry_by_name()
    use["county_code"] = use["admin_1"].map(
        lambda n: reg[_norm(n)].code if _norm(n) in reg else None
    )
    use = use[use["county_code"].notna()]
    if use.empty:
        raise ExternalDataError("FEWS NET prices: no rows matched the county registry.")

    use["period_date"] = pd.to_datetime(
        use["period_date"], errors="coerce", format="ISO8601"
    )
    use = use[use["period_date"].notna()]
    use["year"] = use["period_date"].dt.year
    use["month"] = use["period_date"].dt.month
    # Honour the configured panel window even if the extract is wider.
    use = use[use["year"].between(sy, ey)]
    agg = use.groupby(["county_code", "year", "month"])["value"].mean().reset_index()

    names = {c.code: c.name for c in counties.all_counties()}
    rows = [
        {
            "county_code": r.county_code,
            schemas.COUNTY_NAME_COLUMN: names[r.county_code],
            "year": int(r.year),
            "month": int(r.month),
            schemas.DATE_COLUMN: date(int(r.year), int(r.month), 1).isoformat(),
            "maize_price_kes_kg": round(float(r.value), 2),
            schemas.PROVENANCE_COLUMN: PRICES_SOURCE,
        }
        for r in agg.itertuples()
    ]
    out = pd.DataFrame(rows)
    out = out.sort_values(["county_code", schemas.DATE_COLUMN]).reset_index(drop=True)
    LOGGER.info(
        "FEWS NET market panel: %d county-months of REAL maize retail prices "
        "(%s products, KES/kg).", len(out), ", ".join(sorted(use["product"].unique())),
    )
    return out


# --- outcome: county IPC phases -> documented risk index --------------------

def _month_span(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[int, int]]:
    """(year, month) list covered by a validity window, inclusive."""
    if pd.isna(start) or pd.isna(end):
        return []
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month) and y <= end.year + 1:
        out.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def build_outcome_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Real county-monthly food-security risk from FEWS NET IPC phases."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["fewsnet"]
    sy, ey = _panel_span(config)
    cache = settings.external_dir / "fewsnet"
    url = f"{ext['base_url']}/ipcphase.csv?country_code={ext['iso3']}&scenario=CS&preference=best"
    csv = _fetch_csv(url, cache / f"ipcphase_{ext['iso3']}_CS.csv", settings)
    df = pd.read_csv(csv, encoding="utf-8-sig")
    if df.empty:
        raise ExternalDataError("FEWS NET IPC: empty extract.")

    df = df[df["scenario_name"].astype(str).str.contains("Current Situation", na=False)]
    df = df[df["unit_type"].astype(str).isin(["fsc_admin", "fsc_admin_lhz"])]
    df["phase"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["phase"].isin([1, 2, 3, 4, 5])]

    reg = _registry_by_name()
    df["county_code"] = df["geographic_unit_full_name"].map(lambda n: _unit_code(n, reg))
    df = df[df["county_code"].notna()].copy()
    if df.empty:
        raise ExternalDataError("FEWS NET IPC: no analysis units matched the county registry.")

    df["ps"] = pd.to_datetime(df["projection_start"], errors="coerce", format="ISO8601")
    df["pe"] = pd.to_datetime(df["projection_end"], errors="coerce", format="ISO8601")

    # county-month -> WORST phase across the county's zones (documented).
    worst: dict[tuple[str, int, int], float] = {}
    for r in df.itertuples():
        for (y, m) in _month_span(r.ps, r.pe):
            if not (sy <= y <= ey):
                continue
            key = (r.county_code, y, m)
            risk = phase_to_risk(r.phase)
            worst[key] = max(worst.get(key, 0.0), risk)

    if not worst:
        raise ExternalDataError("FEWS NET IPC: no county-months inside the panel window.")

    names = {c.code: c.name for c in counties.all_counties()}
    rows = [
        {
            "county_code": code,
            schemas.COUNTY_NAME_COLUMN: names[code],
            "year": y,
            "month": m,
            schemas.DATE_COLUMN: date(y, m, 1).isoformat(),
            "food_security_risk_index": risk,
            schemas.PROVENANCE_COLUMN: IPC_SOURCE,
        }
        for (code, y, m), risk in sorted(worst.items())
    ]
    out = pd.DataFrame(rows)
    n_counties = out["county_code"].nunique()
    if n_counties < len(names):
        missing = sorted(set(names) - set(out["county_code"]))
        LOGGER.warning(
            "FEWS NET IPC: %d/%d counties have published analyses in the window; "
            "no IPC rows (left absent, not faked) for: %s",
            n_counties, len(names), ", ".join(missing),
        )
    LOGGER.info(
        "FEWS NET outcome panel: %d county-months of REAL IPC-derived risk "
        "(risk = (phase-1)*25, worst zone per county-month).", len(out),
    )
    return out
