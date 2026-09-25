"""Abstract model interface and model-card metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelCard:
    """Lightweight provenance/metadata record for a trained model."""

    name: str
    feature_names: list[str]
    metrics: dict[str, float]
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data_note: str = (
        "Trained on SYNTHETIC sample data - metrics describe the synthetic "
        "generative process only and carry no real-world meaning."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "trained_at": self.trained_at,
            "data_note": self.data_note,
        }


class RiskModel(ABC):
    """Interface every AgriRisk model must implement.

    Concrete models wrap scikit-learn estimators but expose a stable
    fit/predict/evaluate/save contract so the pipeline and UI never depend on a
    specific algorithm.
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.feature_names_: list[str] = []
        self.estimator_: Any = None

    # --- to be provided by subclasses -------------------------------------
    @abstractmethod
    def _build_estimator(self) -> Any:
        """Construct and return the underlying scikit-learn estimator/pipeline."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> RiskModel:
        """Fit the model, populating ``feature_names_`` and ``estimator_``."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the risk index for the given feature matrix."""
        if self.estimator_ is None:
            raise RuntimeError("Model is not fitted yet - call fit() first.")
        X = self._align(X)
        return np.asarray(self.estimator_.predict(X))

    # --- shared helpers ----------------------------------------------------
    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder/subset columns to the trained feature set."""
        if self.feature_names_:
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise KeyError(f"Prediction features missing columns: {missing}")
            return X[self.feature_names_]
        return X

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute regression metrics on the given split."""
        from .evaluation import compute_metrics

        preds = self.predict(X)
        return compute_metrics(y.to_numpy(), preds)

    def model_card(self, metrics: dict[str, float]) -> ModelCard:
        return ModelCard(name=self.name, feature_names=list(self.feature_names_), metrics=metrics)

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> RiskModel:
        return joblib.load(Path(path))
