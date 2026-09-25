"""Reusable feature-engineering pipeline.

Every derived feature is computed **within a county** and uses only current or
past information (rolling windows / lags), so no future observations leak into a
row's features. The set of engineered features is fully described by
:func:`engineered_feature_names` for transparency and testing.

Feature families
----------------
- ``*_anom``   : rolling z-score anomaly (SPI-like) of a driver, window = zscore_window
- ``*_roll{w}``: rolling mean of a driver over window w
- ``*_lag{l}`` : l of a driver shifted l months
- ``price_pct3``: 3-month percentage change in maize price
- ``month_sin/cos``: cyclical encoding of calendar month (seasonality)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import schemas
from ..config import get_pipeline_config
from ..logging import get_logger

LOGGER = get_logger("features.engineering")

# Bump when the feature definition changes, to invalidate cached feature stores.
FEATURE_VERSION = "1.0.0"

# base_var -> short prefix used in engineered column names
_ENGINEERED_SOURCES: dict[str, str] = {
    "precipitation_mm": "precip",
    "temp_mean_c": "temp",
    "ndvi": "ndvi",
    "evi": "evi",
    "maize_price_kes_kg": "price",
}

# Columns that are never model inputs (grain, labels, provenance, meta).
_META_COLUMNS = set(schemas.KEY_COLUMNS) | {
    schemas.DATE_COLUMN,
    schemas.COUNTY_NAME_COLUMN,
    schemas.PROVENANCE_COLUMN,
}


def engineered_feature_names(config: dict[str, Any] | None = None) -> list[str]:
    """Return the names of all derived (non-raw) features this pipeline creates."""
    config = config or get_pipeline_config()
    fe = config["feature_engineering"]
    names: list[str] = ["month_sin", "month_cos", "price_pct3"]
    for prefix in _ENGINEERED_SOURCES.values():
        names.append(f"{prefix}_anom")
        for w in fe["rolling_windows"]:
            names.append(f"{prefix}_roll{w}")
        for lag in fe["lag_periods"]:
            names.append(f"{prefix}_lag{lag}")
    return names


def build_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered feature columns appended.

    Operates per county on the time-sorted series; windows and lags never cross
    county boundaries.
    """
    config = config or get_pipeline_config()
    fe = config["feature_engineering"]
    zwin = int(fe["zscore_window"])
    rwindows = list(fe["rolling_windows"])
    lags = list(fe["lag_periods"])
    keys = list(config["panel"]["key_columns"])

    out = df.sort_values(keys).copy()

    # Seasonality (county-independent). month may be nullable Int64 -> coerce.
    month = pd.to_numeric(out["month"], errors="coerce").fillna(0).astype(float)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    def _county(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for var, prefix in _ENGINEERED_SOURCES.items():
            if var not in g.columns:
                continue
            series = pd.to_numeric(g[var], errors="coerce")
            roll_mean = series.rolling(zwin, min_periods=max(3, zwin // 4)).mean()
            roll_std = series.rolling(zwin, min_periods=max(3, zwin // 4)).std()
            g[f"{prefix}_anom"] = (series - roll_mean) / roll_std
            for w in rwindows:
                g[f"{prefix}_roll{w}"] = series.rolling(w, min_periods=1).mean()
            for lag in lags:
                g[f"{prefix}_lag{lag}"] = series.shift(lag)
        if "maize_price_kes_kg" in g.columns:
            g["price_pct3"] = (
                pd.to_numeric(g["maize_price_kes_kg"], errors="coerce").pct_change(3) * 100
            )
        return g

    out = (
        out.groupby("county_code", group_keys=False)[list(out.columns)]
        .apply(_county)
        .reset_index(drop=True)
    )
    LOGGER.info(
        "build_features: %s rows, FEATURE_VERSION=%s, +%d engineered cols.",
        len(out), FEATURE_VERSION, len(engineered_feature_names(config)),
    )
    return out


def _split(
    df: pd.DataFrame,
    target: str,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a feature frame into X, y and the resolved feature-name list."""
    numeric = df.select_dtypes(include=[np.number]).columns
    feats = [c for c in numeric if c not in _META_COLUMNS and c != target]
    if feature_names is not None:
        feats = [c for c in feature_names if c in df.columns]
    X = df[feats].astype(float)
    y = pd.to_numeric(df[target], errors="coerce").astype(float)
    return X, y, feats


def make_xy(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build model matrices from an already-engineered feature frame.

    Returns ``(X, y, feature_names)``. Rows with any NaN among the used features
    or the target are dropped by default (early warm-up windows).
    """
    config = config or get_pipeline_config()
    target = config["model"]["target"]
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not present in feature frame.")
    X, y, feats = _split(df, target)
    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
    LOGGER.info("make_xy: X=%s y=%s (%d features).", X.shape, y.shape, len(feats))
    return X, y, feats


class FeaturePipeline:
    """A minimal, reusable fit/transform facade over the functional pipeline.

    ``fit`` learns the feature-name set from a training panel; ``transform``
    rebuilds features and returns ``(X, y)`` aligned to that set. Kept deliberately
    light (no heavy state) so it composes with scikit-learn downstream.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or get_pipeline_config()
        self.target = self.config["model"]["target"]
        self.feature_names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        engineered = build_features(df, self.config)
        _, _, feats = _split(engineered, self.target)
        self.feature_names_ = feats
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        engineered = build_features(df, self.config)
        X, y, _ = _split(engineered, self.target, self.feature_names_ or None)
        mask = X.notna().all(axis=1) & y.notna()
        return X[mask], y[mask]

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return self.fit(df).transform(df)
