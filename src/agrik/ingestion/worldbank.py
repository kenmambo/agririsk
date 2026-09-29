"""World Bank API connector - REAL (national-grain) agriculture signal.

Builds the ``agriculture`` dataset from the World Bank's public
Open Data API: the FAO-based **Food production index** (indicator
``AG.PRD.FOOD.XD``, 2014-2016 = 100) for Kenya, one value per year.

Honest grain note: no free, machine-readable *county x month* crop
production series exists for Kenya (KNBS publishes county production
only in annual PDF reports whose county breakdowns are chart images).
This provider therefore uses the NATIONAL annual index and broadcasts
it to every county-month, labelled ``data_source = "wb_foodindex"``.
It is real observed data, but it carries no county-level contrast -
per-county agricultural conclusions remain invalid. A county-grain feed
(KNBS/KEPHIS) should replace this when one is available.

The response JSON is cached under ``data/raw/external/worldbank/``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from .base import ExternalDataError
from .utils import download_cached

LOGGER = get_logger("ingestion.worldbank")

SOURCE_NAME = "wb_foodindex"


def parse_wb_indicator(payload: list) -> dict[int, float]:
    """World Bank v2 JSON ``[{meta}, [{data}]]`` -> {year: index value}.

    Rows with null values (year not yet published) are dropped.
    """
    if not isinstance(payload, list) or len(payload) < 2:
        raise ExternalDataError("World Bank API: unexpected JSON envelope.")
    out: dict[int, float] = {}
    for rec in payload[1]:
        try:
            year = int(str(rec.get("date")))
            value = rec.get("value")
        except (TypeError, ValueError):
            continue
        if value is not None:
            out[year] = float(value)
    if not out:
        raise ExternalDataError("World Bank API: no non-null values in response.")
    return out


def build_agriculture_panel(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """National food-production index broadcast to county-month rows."""
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    ext = config["external"]["worldbank"]
    panel = config["panel"]
    sy, ey = int(panel["start_year"]), int(panel["end_year"])
    cache = settings.external_dir / "worldbank"
    url = (
        f"{ext['api_url']}?format=json&date={sy}:{ey}"
        f"&per_page={ (ey - sy + 4) * 2 }"
    )
    raw = download_cached(
        url, cache / f"KEN_{ext['indicator'].replace('.', '_')}.json",
        timeout=settings.http_timeout_s,
    )
    import json

    index = parse_wb_indicator(json.loads(raw.read_text(encoding="utf-8")))
    years_missing = [y for y in range(sy, ey + 1) if y not in index]
    if years_missing:
        LOGGER.warning(
            "World Bank food index: no value for %s - those county-months stay "
            "absent (not interpolated, not faked).", years_missing,
        )

    names = {c.code: c.name for c in counties.all_counties()}
    rows = []
    for code in names:
        for y in range(sy, ey + 1):
            if y not in index:
                continue
            for m in range(1, 13):
                rows.append(
                    {
                        "county_code": code,
                        schemas.COUNTY_NAME_COLUMN: names[code],
                        "year": y,
                        "month": m,
                        schemas.DATE_COLUMN: date(y, m, 1).isoformat(),
                        "prod_index": round(index[y], 2),
                        schemas.PROVENANCE_COLUMN: SOURCE_NAME,
                    }
                )
    if not rows:
        raise ExternalDataError(
            f"World Bank food index: nothing published for {sy}-{ey}."
        )
    LOGGER.info(
        "World Bank agriculture panel: %d county-months, NATIONAL food "
        "production index (2014-16=100) broadcast - real but no county "
        "contrast (documented grain limitation).", len(rows),
    )
    return pd.DataFrame(rows)
