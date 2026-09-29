"""Gradient-boosting model behind the same RiskModel interface as Ridge.

``HistGradientBoostingRegressor`` captures the non-linear, saturating
relationships the linear baseline cannot (e.g. NDVI anomaly -> risk flattens at
the green-up ceiling) and handles NaN natively, so genuinely absent driver
months stay absent instead of being imputed into submission.

Selected through ``config/pipeline.yaml`` (``model.baseline.type: gbm``) or
``build_model("gbm")``; hyper-parameters come from the ``model.gbm`` section.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from ..logging import get_logger
from .base import RiskModel

LOGGER = get_logger("models.gbm")


class GradientBoostingRiskModel(RiskModel):
    """Hist gradient boosting regressor with a permutation-importance view."""

    name = "gbm"

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.06,
        max_depth: int | None = 4,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        # kept so coefficients() can compute permutation importance post-fit
        self._train_X_: pd.DataFrame | None = None
        self._train_y_: pd.Series | None = None

    def _build_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingRiskModel:
        self.feature_names_ = list(X.columns)
        self._train_X_, self._train_y_ = X.copy(), y.copy()
        self.estimator_ = self._build_estimator().fit(X, y)
        LOGGER.info(
            "GradientBoostingRiskModel fitted on %d rows, %d features.",
            len(X), X.shape[1],
        )
        return self

    def coefficients(self, n_repeats: int = 5) -> pd.Series:
        """Permutation importances on the training data.

        These are *predictive* importances (how much shuffling a column hurts
        the fit) - not coefficients and not causal effects.
        """
        if self.estimator_ is None or self._train_X_ is None:
            raise RuntimeError("Model not fitted.")
        result: Any = permutation_importance(
            self.estimator_,
            self._train_X_,
            self._train_y_,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="r2",
        )
        return pd.Series(
            result.importances_mean, index=self.feature_names_, name="importance"
        )
