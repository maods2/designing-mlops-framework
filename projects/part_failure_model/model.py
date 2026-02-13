"""Part-failure classification model implementation."""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class PartFailureModel:
    """
    Part-failure classification model using Random Forest.

    Binary classification: will_fail (1) / ok (0).
    Demonstrates model-agnostic design - any ML framework can be used.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42):
        """
        Initialize the part-failure model.

        Args:
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PartFailureModel":
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target labels (0=ok, 1=will_fail)

        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_names is None:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
