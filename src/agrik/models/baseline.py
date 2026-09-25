"""A concrete baseline model: Ridge regression behind the RiskModel interface.

Deliberately simple and interpretable - it establishes a performance floor and a
reference feature set against which future models (gradient boosting, spatial
models, IPC classifiers) can be compared via the same interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..logging import get_logger
from .base import RiskModel

LOGGER = get_logger("models.baseline")


class RidgeRiskModel(RiskModel):
    """Standardised Ridge regression baseline."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        super().__init__(alpha=alpha, random_state=random_state)
        self.alpha = alpha
        self.random_state = random_state

    def _build_estimator(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeRiskModel:
        self.feature_names_ = list(X.columns)
        self.estimator_ = self._build_estimator().fit(X, y)
        LOGGER.info("RidgeRiskModel fitted on %d rows, %d features.", len(X), X.shape[1])
        return self

    def coefficients(self) -> pd.Series:
        """Standardised Ridge coefficients (feature importances for inspection).

        Note: with correlated engineered features these describe the fitted
        synthetic relationship only - they are not causal real-world effects.
        """
        if self.estimator_ is None:
            raise RuntimeError("Model not fitted.")
        ridge: Any = self.estimator_.named_steps["ridge"]
        return pd.Series(np.asarray(ridge.coef_).ravel(), index=self.feature_names_)
