"""Modelling layer.

Defines a reusable model interface (:class:`RiskModel`) and a concrete baseline
regressor. This layer depends only on feature matrices (numpy / pandas) - it
never reads raw files or renders UI, keeping modelling independent of ingestion
and presentation.
"""

from .base import ModelCard, RiskModel
from .baseline import RidgeRiskModel
from .evaluation import compute_metrics, temporal_train_test_split
from .registry import build_model

__all__ = [
    "RiskModel",
    "ModelCard",
    "RidgeRiskModel",
    "compute_metrics",
    "temporal_train_test_split",
    "build_model",
]
