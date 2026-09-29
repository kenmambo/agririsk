"""Modelling layer.

Defines a reusable model interface (:class:`RiskModel`) and a concrete baseline
regressor. This layer depends only on feature matrices (numpy / pandas) - it
never reads raw files or renders UI, keeping modelling independent of ingestion
and presentation.
"""

from .base import ModelCard, RiskModel
from .baseline import RidgeRiskModel
from .evaluation import compute_metrics, temporal_three_way_split, temporal_train_test_split
from .gbm import GradientBoostingRiskModel
from .horizon import horizon_report
from .registry import build_model
from .uncertainty import (
    add_intervals,
    conformal_quantile,
    interval_metrics,
    reliability_table,
)

__all__ = [
    "RiskModel",
    "ModelCard",
    "RidgeRiskModel",
    "GradientBoostingRiskModel",
    "compute_metrics",
    "temporal_train_test_split",
    "temporal_three_way_split",
    "build_model",
    "conformal_quantile",
    "add_intervals",
    "interval_metrics",
    "reliability_table",
    "horizon_report",
]
