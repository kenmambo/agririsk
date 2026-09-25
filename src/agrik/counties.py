"""County reference data (shared master dimensions).

Kenya has 47 counties. This module ships a documented *subset* (official codes
001-021) that spans the country's main agro-ecological zones, which is enough
to exercise the county-monthly MVP. Codes 022-047 should be added when the
authoritative boundaries / codes source (KNBS / Divisions of Kenya) is wired in.

``lat``/``lon`` are APPROXIMATE representative centroids (county-headquarters
towns) intended only for simple dashboard mapping; they are NOT authoritative
boundaries. Replace with a real GeoJSON / GeoPackage when the ``[geo]`` extra
and a boundaries layer are integrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class County:
    """A Kenyan county with approximate reference geometry."""

    code: str          # official KNBS 3-digit code, e.g. "001"
    name: str
    lat: float         # approximate centroid latitude (WGS84)
    lon: float         # approximate centroid longitude (WGS84)
    agro_zone: str     # coarse agro-ecological label for grouping/filtering


# Representative subset (codes 001-021). Coordinates are approximate.
_COUNTIES: tuple[County, ...] = (
    County("001", "Mombasa", -4.04, 39.66, "Coastal"),
    County("002", "Kwale", -4.23, 39.46, "Coastal"),
    County("003", "Kilifi", -3.53, 39.85, "Coastal"),
    County("004", "Tana River", -2.35, 40.10, "Riversine / ASAL"),
    County("005", "Lamu", -2.27, 40.90, "Coastal / Arid ASAL"),
    County("006", "Isiolo", 0.35, 37.58, "Arid ASAL"),
    County("007", "Meru", 0.05, 37.70, "Highland / Mixed"),
    County("008", "Tharaka-Nithi", -0.35, 37.60, "Highland"),
    County("009", "Embu", -0.53, 37.45, "Medium Highland"),
    County("010", "Kitui", -1.37, 38.01, "Semi-arid"),
    County("011", "Machakos", -1.52, 37.26, "Semi-arid"),
    County("012", "Makueni", -2.04, 37.50, "Semi-arid"),
    County("013", "Wajir", 1.75, 40.05, "Arid ASAL"),
    County("014", "Mandera", 3.94, 41.87, "Arid ASAL"),
    County("015", "Marsabit", 2.34, 37.99, "Arid ASAL"),
    County("016", "Garissa", -0.46, 39.65, "Semi-arid ASAL"),
    County("017", "Taita Taveta", -3.40, 38.35, "Semi-arid"),
    County("018", "Nakuru", 0.30, 36.08, "Rift Valley / Mixed"),
    County("019", "Narok", -1.08, 35.86, "Rift Valley / Pastoral"),
    County("020", "Kajiado", -1.85, 36.90, "Semi-arid"),
    County("021", "Kiambu", -1.17, 36.83, "Medium Highland"),
)


def all_counties() -> list[County]:
    """Return the full county registry as a list of :class:`County`."""
    return list(_COUNTIES)


def codes() -> list[str]:
    """Return the list of county codes in the registry."""
    return [c.code for c in _COUNTIES]


def get_county(code: str) -> County:
    """Return a single :class:`County` by code (zero-padded to 3 digits)."""
    padded = str(code).zfill(3)
    for county in _COUNTIES:
        if county.code == padded:
            return county
    raise KeyError(f"Unknown county_code: {code!r}")


def counties_frame() -> pd.DataFrame:
    """Return the registry as a DataFrame (code, name, lat, lon, agro_zone)."""
    return pd.DataFrame([c.__dict__ for c in _COUNTIES]).rename(
        columns={"code": "county_code"}
    )
