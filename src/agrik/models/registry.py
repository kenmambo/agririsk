"""Model registry / factory.

Maps the ``model.baseline.type`` config value to a concrete :class:`RiskModel`,
so new algorithms can be added without touching the pipeline or UI.
"""

from __future__ import annotations

from typing import Any

from ..config import get_pipeline_config
from .base import RiskModel
from .baseline import RidgeRiskModel

_AVAILABLE: dict[str, type[RiskModel]] = {
    "ridge": RidgeRiskModel,
}


def build_model(name: str | None = None, config: dict[str, Any] | None = None) -> RiskModel:
    """Instantiate a model by name using hyper-parameters from the config."""
    config = config or get_pipeline_config()
    params: dict[str, Any] = dict(config["model"].get("baseline", {}))
    name = (name or params.pop("type", "ridge")).lower()
    if name not in _AVAILABLE:
        raise KeyError(
            f"Unknown model type {name!r}. Available: {sorted(_AVAILABLE)}"
        )
    model_cls = _AVAILABLE[name]
    return model_cls(**params)


def register_model(name: str, model_cls: type[RiskModel]) -> None:
    """Register an additional model implementation at runtime."""
    _AVAILABLE[name.lower()] = model_cls
