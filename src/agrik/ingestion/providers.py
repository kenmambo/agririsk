"""Provider registry: routes each dataset to synthetic or a real feed.

This is the single switchboard between ``config/pipeline.yaml`` dataset sources
and the connectors that implement them. Rules:

- ``source: synthetic``  -> clearly-labelled generator (default, offline).
- ``source: <provider>`` -> registered connector (CHIRPS, MODIS, ...).
- Real-feed failure      -> loud error; optional, clearly-labelled fallback to
  the synthetic slice when ``external.allow_synthetic_fallback`` is true.
  Fallback is *never* silent: it is logged as ERROR and visible in the
  per-dataset ``*_source`` provenance columns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .. import schemas
from ..config import get_pipeline_config
from ..logging import get_logger
from ..settings import Settings, get_settings
from . import chirps, modis
from .synthetic import build_synthetic_datasets

LOGGER = get_logger("ingestion.providers")

# source name -> (dataset it serves, builder). Builders are resolved through
# the module attribute at call time so tests can monkeypatch the connectors.
_REGISTRY: dict[str, tuple[str, Callable[..., pd.DataFrame]]] = {
    "chirps": ("climate", lambda **kw: chirps.build_climate_panel(**kw)),
    "modis": ("vegetation", lambda **kw: modis.build_vegetation_panel(**kw)),
}


def register_provider(name: str, dataset: str, builder: Callable[..., pd.DataFrame]) -> None:
    """Extension hook for future feeds (FEWSNET, KNBS, ERA5, ...)."""
    _REGISTRY[name.lower()] = (dataset, builder)


def available_sources() -> list[str]:
    """Registered real-feed provider names."""
    return sorted(_REGISTRY)


def ingest_datasets(
    config: dict[str, Any] | None = None,
    settings: Settings | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build one frame per configured dataset, honouring its ``source``.

    Synthetic frames are generated once and sliced as needed, so a mixed
    real/synthetic configuration is cheap and deterministic.
    """
    config = config or get_pipeline_config()
    settings = settings or get_settings()
    specs = schemas.dataset_specs(config)
    allow_fallback = bool(config.get("external", {}).get("allow_synthetic_fallback", True))

    def _synth_slice(name: str) -> pd.DataFrame:
        nonlocal synthetic_frames
        if synthetic_frames is None:
            synthetic_frames = build_synthetic_datasets(seed=seed)
        return synthetic_frames[name]

    synthetic_frames: dict[str, pd.DataFrame] | None = None
    frames: dict[str, pd.DataFrame] = {}
    for name, spec in specs.items():
        source = spec.source.lower().strip()
        if source == "synthetic":
            frames[name] = _synth_slice(name)
            continue
        entry = _REGISTRY.get(source)
        if entry is None:
            raise ValueError(
                f"Dataset {name!r}: unknown source {spec.source!r}. "
                f"Registered providers: {available_sources()} or 'synthetic'."
            )
        target_dataset, builder = entry
        if target_dataset != name:
            raise ValueError(
                f"Provider {source!r} serves dataset {target_dataset!r}, not {name!r}."
            )
        try:
            df = builder(config=config, settings=settings)
            absent = [c for c in spec.variables if c not in df.columns]
            if absent:
                LOGGER.warning(
                    "Provider %s omits schema variables %s for dataset %s "
                    "(not faked - columns stay absent).", source, absent, name,
                )
            frames[name] = df
        except Exception as exc:  # noqa: BLE001 - decide policy centrally
            if not allow_fallback:
                raise
            LOGGER.error(
                "Provider %s failed for dataset %r (%s). FALLING BACK to "
                "SYNTHETIC data for that dataset - provenance columns will "
                "show '%s_source=synthetic'. Do NOT interpret it as real.",
                source, name, exc, name,
            )
            frames[name] = _synth_slice(name)
    return frames
