"""Read-only access to pipeline artefacts for the serving layer.

The API never ingests, validates, fits or writes - it exposes exactly what
``python -m agrik`` put on disk, so serving and data science stay decoupled
(the same rule the dashboard follows). Content is cached by file path *and*
modification time, so a re-run of the pipeline is picked up without a restart.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from .. import counties, schemas
from ..settings import get_settings


class ArtifactMissing(RuntimeError):
    """Raised when an artefact is not on disk yet (pipeline not run)."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Artefact {path} not found - run `python -m agrik` first."
        )
        self.path = path


# path -> (mtime, parsed payload). Tiny hand-rolled cache: mtime invalidates.
_CACHE: dict[str, tuple[float, Any]] = {}


def _load(path: Path, loader: Callable[[Path], Any]) -> Any:
    if not path.exists():
        raise ArtifactMissing(path)
    mtime = path.stat().st_mtime
    key = str(path)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    value = loader(path)
    _CACHE[key] = (mtime, value)
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"county_code": str})
    df["county_code"] = df["county_code"].str.zfill(3)
    return df


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def features_panel() -> pd.DataFrame:
    return _load(get_settings().features_dir / "features_panel.csv", _read_csv)


def features_manifest() -> dict:
    return _load(get_settings().features_dir / "features_manifest.json", _read_json)


def model_card() -> dict:
    return _load(get_settings().models_dir / "model_card.json", _read_json)


def comparison() -> dict:
    return _load(get_settings().models_dir / "comparison.json", _read_json)


def test_predictions() -> pd.DataFrame:
    return _load(get_settings().models_dir / "test_predictions.csv", _read_csv)


def county_reference() -> pd.DataFrame:
    """County registry reference data (not an artefact - always available)."""
    return counties.counties_frame()


# --- provenance helpers (streamlit-free twins of the dashboard loaders) -----
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
    sources = dataset_sources(df)
    if sources:
        return any("synthetic" in s for s in sources.values())
    if schemas.PROVENANCE_COLUMN in df.columns:
        return bool(
            (df[schemas.PROVENANCE_COLUMN].astype(str).str.lower() == "synthetic").any()
        )
    return True  # unknown -> assume synthetic for safe labelling


def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe list of row dicts (numpy scalars and NaN -> native/null)."""
    return json.loads(df.to_json(orient="records"))
