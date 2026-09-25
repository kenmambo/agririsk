"""SYNTHETIC sample-data generator.

WHY THIS EXISTS
---------------
Real county-level climate / vegetation / market / socioeconomic feeds have not
been integrated yet. To let the rest of the pipeline (validation, features,
modelling, UI) be built and tested, this module produces *clearly-labelled
synthetic* data. It is NOT real and must NEVER be used to draw analytical
conclusions about any actual county.

DESIGN (so it is honest and learnable, not random noise)
--------------------------------------------------------
Data follows a simple, transparent causal chain per county:

    rainfall -> vegetation (NDVI/EVI) -> production -> maize price -> risk

plus a county-level socioeconomic "vulnerability" term. The target
``food_security_risk_index`` is an explicit (documented) function of these
drivers, so a baseline regressor has genuine signal to recover. Every row is
stamped with ``data_source = "synthetic"``.

When a real feed is connected, add a loader elsewhere in ``ingestion/`` and
point ``config/pipeline.yaml`` at it; nothing downstream should change because
all layers depend on the schema, not on this generator.
"""

from __future__ import annotations

from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from .. import counties, schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from .base import raw_path, write_csv

LOGGER = get_logger("ingestion.synthetic")

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA - not real observations. Generated only to exercise the "
    "pipeline until real feeds are integrated. Do not interpret per-county values."
)

# Bimodal Kenyan rainfall climatology share by month (index 0 = January).
# Long rains Mar-May, short rains Oct-Dec. Sums to 1.0.
_MONTHLY_RAIN_SHARE = np.array(
    [0.02, 0.03, 0.16, 0.22, 0.14, 0.05, 0.02, 0.02, 0.04, 0.14, 0.13, 0.03]
)


def _zone_params(agro_zone: str) -> dict[str, float]:
    """Coarse climate/vulnerability baselines per agro-ecological zone."""
    z = agro_zone.lower()
    if "arid" in z and "semi" not in z:
        return dict(annual_rain=350, base_temp=28.0, ndvi=0.18, poverty=0.60, coping=0.30)
    if "semi-arid" in z:
        return dict(annual_rain=650, base_temp=26.0, ndvi=0.30, poverty=0.45, coping=0.45)
    if "highland" in z:
        return dict(annual_rain=1200, base_temp=20.0, ndvi=0.62, poverty=0.25, coping=0.65)
    if "coastal" in z:
        return dict(annual_rain=900, base_temp=26.0, ndvi=0.42, poverty=0.35, coping=0.55)
    if "rift" in z:
        return dict(annual_rain=800, base_temp=22.0, ndvi=0.45, poverty=0.35, coping=0.55)
    return dict(annual_rain=700, base_temp=24.0, ndvi=0.35, poverty=0.40, coping=0.50)


def _z(a: np.ndarray) -> np.ndarray:
    """Standardise an array (guarding against zero variance)."""
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-9 else 1.0)


def _lag1(a: np.ndarray) -> np.ndarray:
    """Return a one-step lagged copy, holding the first value."""
    out = np.empty_like(a)
    out[0] = a[0]
    out[1:] = a[:-1]
    return out


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def build_master_panel(
    start_year: int | None = None,
    end_year: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build one long-format county-monthly master panel with all variables.

    The returned frame contains every raw variable across all datasets plus
    keys, ``county_name`` and the ``data_source`` provenance marker.
    """
    cfg = get_pipeline_config()
    panel = cfg["panel"]
    start_year = start_year if start_year is not None else panel["start_year"]
    end_year = end_year if end_year is not None else panel["end_year"]

    rng = np.random.default_rng(seed)
    months = [(y, m) for y, m in product(range(start_year, end_year + 1), range(1, 13))]
    n = len(months)

    frames: list[pd.DataFrame] = []
    for county in counties.all_counties():
        p = _zone_params(county.agro_zone)
        # County-level random multipliers for heterogeneity.
        rain_mult = rng.uniform(0.8, 1.2)
        pop_base = rng.uniform(3.0e5, 1.6e6)
        poverty = _clip(np.array([p["poverty"] + rng.normal(0, 0.03)]), 0.0, 1.0)[0]
        coping = _clip(np.array([p["coping"] + rng.normal(0, 0.03)]), 0.0, 1.0)[0]

        # Multi-year ENSO-like wet/dry episodes (one factor per year).
        year_factor = {
            y: float(np.clip(rng.normal(1.0, 0.18), 0.45, 1.35))
            for y in range(start_year, end_year + 1)
        }

        years = np.array([y for y, _ in months])
        mons = np.array([m for _, m in months])

        # --- Rainfall (gamma noise => positive, right-skewed) --------------
        clim = _MONTHLY_RAIN_SHARE[mons - 1] * p["annual_rain"] * rain_mult * np.array(
            [year_factor[y] for y in years]
        )
        shape = 1.2
        precip = rng.gamma(shape=shape, scale=clim / shape)
        precip = np.where(clim < 20, precip * 0.7, precip)  # dry months stay dry

        temp_mean = (
            p["base_temp"]
            + 2.0 * np.cos((mons - 1) / 12 * 2 * np.pi)
            + rng.normal(0, 0.8, n)
        )

        rain_z = _z(precip)

        # --- Vegetation responds to current + lagged rainfall --------------
        ndvi = _clip(
            p["ndvi"] + 0.12 * rain_z + 0.06 * _lag1(rain_z) + rng.normal(0, 0.02, n),
            0.05, 0.85,
        )
        evi = _clip(0.55 * ndvi + rng.normal(0, 0.01, n), 0.02, 0.65)
        ndvi_z = _z(ndvi)

        # --- Agriculture: production tracks growing-season greenness -------
        prod_index = _clip(100 * (0.45 + 0.85 * ndvi) + rng.normal(0, 4, n), 10, 200)

        # --- Market: urban maize price, inflation trend + supply response ---
        t = np.arange(n)
        price = (
            42.0 * (1 + 0.05 * t / max(n - 1, 1))
            * (1 - 0.0016 * (prod_index - 100))
            + rng.normal(0, 1.5, n)
        )
        price = _clip(price, 20, 160)
        price_z = _z(price)

        # --- Socioeconomic (county-level, slowly varying) ------------------
        rural_pop = pop_base * (1 + 0.03 * t / 12.0)  # mild growth

        # --- Outcome: documented risk function of the drivers --------------
        drought_stress = _clip(-rain_z, 0, 3) / 3
        veg_stress = _clip(-ndvi_z, 0, 3) / 3
        price_stress = _clip(price_z, 0, 3) / 3
        vulnerability = 0.5 * poverty + 0.5 * (1 - coping)
        risk_index = _clip(
            100
            * (
                0.35 * drought_stress
                + 0.25 * veg_stress
                + 0.15 * price_stress
                + 0.25 * vulnerability
            )
            + rng.normal(0, 3, n),
            0, 100,
        )
        ipc_crisis = _clip(0.05 + 0.6 * (risk_index / 100) + rng.normal(0, 0.02, n), 0, 1)

        df = pd.DataFrame(
            {
                "county_code": county.code,
                schemas.COUNTY_NAME_COLUMN: county.name,
                "year": years,
                "month": mons,
                schemas.DATE_COLUMN: [date(y, m, 1).isoformat() for y, m in months],
                "precipitation_mm": precip.round(2),
                "temp_mean_c": temp_mean.round(2),
                "ndvi": ndvi.round(4),
                "evi": evi.round(4),
                "prod_index": prod_index.round(2),
                "maize_price_kes_kg": price.round(2),
                "rural_pop": rural_pop.round(0),
                "poverty_rate": np.round(np.full(n, poverty), 4),
                "coping_capacity_index": np.round(np.full(n, coping), 4),
                "food_security_risk_index": risk_index.round(2),
                "ipc_crisis_households": ipc_crisis.round(4),
                schemas.PROVENANCE_COLUMN: "synthetic",
            }
        )
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    LOGGER.warning("Built SYNTHETIC master panel: %s rows x %s cols. %s",
                   *master.shape, SYNTHETIC_BANNER)
    return master


def build_synthetic_datasets(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Return the master panel split into per-dataset long-format frames.

    Keys match the ``datasets`` section of ``config/pipeline.yaml``.
    """
    master = build_master_panel(seed=seed)
    keys = ["county_code", schemas.COUNTY_NAME_COLUMN, "year", "month", schemas.DATE_COLUMN]
    datasets: dict[str, pd.DataFrame] = {}
    for name, cols in schemas.DATASET_VARIABLES.items():
        sub = master[keys + cols + [schemas.PROVENANCE_COLUMN]].copy()
        datasets[name] = sub
    return datasets


def write_synthetic_raw(seed: int = 42) -> dict[str, Path]:
    """Generate synthetic datasets and write each to ``data/raw``.

    Returns a mapping of dataset name -> written path.
    """
    cfg = get_pipeline_config()
    specs = schemas.dataset_specs(cfg)
    datasets = build_synthetic_datasets(seed=seed)
    written: dict[str, Path] = {}
    for name, df in datasets.items():
        filename = specs[name].file if name in specs else f"{name}.csv"
        path = write_csv(df, raw_path(filename))
        written[name] = path
        LOGGER.info("Wrote synthetic '%s' -> %s (%s rows).", name, path, len(df))
    return written
