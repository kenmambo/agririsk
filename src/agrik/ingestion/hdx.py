"""HDX / KNBS-derived socioeconomic connector - REAL reference data.

Builds the ``socioeconomic`` dataset from two published, machine-readable
sources (no scraping, both cached as raw CSVs):

* ``poverty_rate``   - county Multidimensional Poverty Index *headcount
  ratio* (share of people multidimensionally poor), UNDP/OPHI global MPI
  subnational data for Kenya (KDHS 2022 survey round), published on HDX.
  Converted from percent to the schema's 0-1 fraction.
* ``rural_pop``      - rural population summed from sub-county (ADM2)
  rural-population estimates (WorldPop / HDX ``Kenya - Risk Assessment
  Indicators`` inputs) to county level via the official KNBS county PCode
  (``KE0xx`` -> registry code ``0xx``).

Both are SLOWLY-VARYING county attributes: the same value is broadcast
across every panel month, which is honest for structural covariates
(the synthetic generator's month-to-month churn was the fiction).

``coping_capacity_index`` has no free county-level source and is therefore
*omitted* (the providers registry logs the absence; the column never fakes
values in).

Provenance: rows carry ``data_source = "hdx_knbs"``.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .utils import download_cached

LOGGER = get_logger("ingestion.hdx")

SOURCE_NAME = "hdx_knbs"


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", str(name).replace("-", " ").lower())).strip()


def _county_by_norm() -> dict[str, Any]:
    return {_norm(c.name): c for c in counties.all_counties()}


def poverty_by_county(mpi_csv, registry: dict[str, Any]) -> dict[str, float]:
    """county_code -> multidimensional poverty headcount ratio (0-1)."""
    df = pd.read_csv(mpi_csv)
    df = df[df["Admin 1 PCode"].notna() & df["Admin 1 Name"].notna()]
    out: dict[str, float] = {}
    for _, r in df.iterrows():
        c = registry.get(_norm(r["Admin 1 Name"]))
        ratio = float(r["Headcount Ratio"])
        if c is not None and 0.0 <= ratio <= 100.0:
            out[c.code] = round(ratio / 100.0, 4)
    return out


def rural_pop_by_county(rural_csv, registry: dict[str, Any]) -> dict[str, int]:
    """county_code -> rural population (ADM2 rows summed via KE0xx pcode)."""
    df = pd.read_csv(rural_csv)
    df["adm1"] = df["ADM_PCODE"].astype(str).str.upper().str[:5]
    agg = df.groupby("adm1")["total_pop_rural"].sum()
    codes = {c.code for c in counties.all_counties()}
    out: dict[str, int] = {}
    for adm1, pop in agg.items():
        code = str(adm1).replace("KE", "")
        if code in codes:
            out[code] = int(round(float(pop)))
    return out


def build_socioeconomic_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Real county socioeconomic attributes broadcast over the panel months."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["hdx"]
    panel = config["panel"]
    sy, ey = int(panel["start_year"]), int(panel["end_year"])
    cache = settings.external_dir / "hdx"
    cache.mkdir(parents=True, exist_ok=True)

    mpi_csv = download_cached(
        ext["mpi_url"], cache / "ken_mpi.csv", timeout=settings.http_timeout_s
    )
    rural_csv = download_cached(
        ext["rural_population_url"], cache / "ken_adm2_rural_population.csv",
        timeout=settings.http_timeout_s,
    )

    registry = _county_by_norm()
    pov = poverty_by_county(mpi_csv, registry)
    rpop = rural_pop_by_county(rural_csv, registry)
    if not pov or not rpop:
        raise ExternalDataError(
            f"HDX socioeconomic: poverty rows={len(pov)}, rural-pop rows={len(rpop)} "
            "- county join failed (names/pcodes changed upstream?)."
        )
    usable = sorted(set(pov) & set(rpop))
    missing = sorted({c.code for c in counties.all_counties()} - set(usable))
    if missing:
        LOGGER.warning(
            "HDX socioeconomic: no real reference values for %d registry counties "
            "(rows left absent, not faked): %s",
            len(missing), ", ".join(missing),
        )

    names = {c.code: c.name for c in counties.all_counties()}
    rows = []
    for code in usable:
        for y in range(sy, ey + 1):
            for m in range(1, 13):
                rows.append(
                    {
                        "county_code": code,
                        schemas.COUNTY_NAME_COLUMN: names[code],
                        "year": y,
                        "month": m,
                        schemas.DATE_COLUMN: date(y, m, 1).isoformat(),
                        "rural_pop": rpop[code],
                        "poverty_rate": pov[code],
                        schemas.PROVENANCE_COLUMN: SOURCE_NAME,
                    }
                )
    out = pd.DataFrame(rows)
    LOGGER.info(
        "HDX socioeconomic panel: %d counties x %d months REAL static attributes "
        "(MPI headcount KDHS-2022 survey; rural pop WorldPop ADM2 sum).",
        len(usable), ey - sy + 1,
    )
    return out
