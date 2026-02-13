"""Preprocess step for part-failure classification pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split

from custom.data_loader import create_synthetic_training_data, load_raw_data
from custom.feature_engineering import build_features
from mlops_framework.core.step_types import PreprocessStep


class PartFailurePreprocess(PreprocessStep):
    """
    Preprocess step: load raw data, build features, split, save train_data.
    Uses custom modules for business logic.
    """

    input_schema = {"raw_data": "PATH"}
    output_schema = {"train_data": "DATASET"}

    def run(self) -> None:
        train_path = self.context.config.get("train_path", "data/train.csv")
        test_size = self.context.config.get("test_size", 0.2)
        random_state = self.context.config.get("random_state", 42)

        self.log(f"Loading data from {train_path}")

        df = load_raw_data(train_path)
        if df is None:
            self.log("Data file not found. Creating synthetic data for demo...")
            X, y = create_synthetic_training_data(seed=random_state)
            df = pd.concat([X, y.rename("target")], axis=1)

        df_clean = df.dropna()
        X = build_features(df_clean, drop_target=True)
        y = df_clean.iloc[:, -1]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        train_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }
        self.save_artifact("train_data", train_data)
        self.log(f"Saved train_data with {len(X_train)} train, {len(X_val)} val samples")


if __name__ == "__main__":
    from mlops_framework import run_local

    run_local("steps.preprocess.PartFailurePreprocess")
