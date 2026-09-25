"""Cached data loaders for the dashboard.

All UI-facing data access funnels through here so the rest of the app just
renders. Returns empty artefacts (with a flag) when the pipeline has not been
run yet, so the UI can show a helpful message instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .. import counties, schemas
from ..settings import get_settings


@st.cache_data(show_spinner=False)
def load_feature_panel() -> pd.DataFrame:
    """Load the engineered feature store produced by the pipeline."""
    path = get_settings().features_dir / "features_panel.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"county_code": str})
    df["county_code"] = df["county_code"].str.zfill(3)
    return df


@st.cache_data(show_spinner=False)
def load_model_card() -> dict:
    """Load the baseline model card (metrics + provenance note)."""
    path = get_settings().models_dir / "model_card.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_manifest() -> dict:
    """Load the feature-store manifest (provenance / versions)."""
    path = get_settings().features_dir / "features_manifest.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def county_reference() -> pd.DataFrame:
    """County registry (code, name, lat, lon, agro_zone) for map/labels."""
    return counties.counties_frame()


def pipeline_ready() -> bool:
    """True if both the feature store and a model card exist on disk."""
    s = get_settings()
    return (
        (s.features_dir / "features_panel.csv").exists()
        and (s.models_dir / "model_card.json").exists()
    )


def dataset_sources(df: pd.DataFrame) -> dict[str, str]:
    """Per-dataset provenance recovered from merged ``{name}_source`` columns."""
    out: dict[str, str] = {}
    for col in df.columns:
        if col.endswith("_source") and col != schemas.PROVENANCE_COLUMN:
            vals = sorted(df[col].dropna().astype(str).unique())
            if vals:
                out[col[: -len("_source")]] = ",".join(vals)
    return out


def data_is_synthetic(df: pd.DataFrame) -> bool:
    """Whether the loaded data still contains synthetic components.

    Prefers the per-dataset ``{name}_source`` columns (survive mixed runs);
    falls back to the overall ``data_source`` column. Unknown -> assume
    synthetic for safe labelling.
    """
    sources = dataset_sources(df)
    if sources:
        return any("synthetic" in s for s in sources.values())
    if schemas.PROVENANCE_COLUMN in df.columns:
        return bool((df[schemas.PROVENANCE_COLUMN].astype(str).str.lower() == "synthetic").any())
    return True  # unknown -> assume synthetic for safe labelling


@st.cache_resource(show_spinner=False)
def load_baseline_model():
    """Load the trained baseline model (artefact) for coefficient inspection.

    This only *reads* a saved artefact - the dashboard never fits a model.
    """
    from ..models import RiskModel

    path = get_settings().models_dir / "baseline_ridge.joblib"
    if not path.exists():
        return None
    try:
        return RiskModel.load(path)
    except Exception:  # noqa: BLE001 - UI must degrade gracefully
        return None
