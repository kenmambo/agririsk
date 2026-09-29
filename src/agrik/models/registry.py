"""Model registry / factory.

Maps the ``model.baseline.type`` config value to a concrete :class:`RiskModel`,
so new algorithms can be added without touching the pipeline or UI.
"""

from __future__ import annotations

from typing import Any

from ..config import get_pipeline_config
from .base import RiskModel
from .baseline import RidgeRiskModel
from .gbm import GradientBoostingRiskModel

_AVAILABLE: dict[str, type[RiskModel]] = {
    "ridge": RidgeRiskModel,
    "gbm": GradientBoostingRiskModel,
}


def build_model(name: str | None = None, config: dict[str, Any] | None = None) -> RiskModel:
    """Instantiate a model by name using hyper-parameters from the config.

    With no ``name``, the primary model (``model.baseline.type``) is built and
    its parameters are read from ``model.<type>`` when that section exists
    (falling back to ``model.baseline``), so each algorithm keeps its own
    hyper-parameters in the YAML.
    """
    config = config or get_pipeline_config()
    model_cfg = config["model"]
    baseline = dict(model_cfg.get("baseline", {}))
    baseline_type = str(baseline.get("type", "ridge")).lower()
    if name is None:
        params: dict[str, Any] = dict(baseline)
        name = str(params.pop("type", "ridge"))
        section = model_cfg.get(name.lower())
        if isinstance(section, dict):
            params = {k: v for k, v in section.items() if k != "type"}
    else:
        section = model_cfg.get(name.lower())
        if isinstance(section, dict):
            params = {k: v for k, v in section.items() if k != "type"}
        elif name.lower() == baseline_type:
            # same algorithm as the primary: reuse its hyper-parameters
            params = {k: v for k, v in baseline.items() if k != "type"}
        else:
            params = {}
    name = name.lower()
    if name not in _AVAILABLE:
        raise KeyError(
            f"Unknown model type {name!r}. Available: {sorted(_AVAILABLE)}"
        )
    model_cls = _AVAILABLE[name]
    return model_cls(**params)


def register_model(name: str, model_cls: type[RiskModel]) -> None:
    """Register an additional model implementation at runtime."""
    _AVAILABLE[name.lower()] = model_cls
