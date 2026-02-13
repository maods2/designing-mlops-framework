"""Data loading utilities for part-failure pipeline.

Data scientists extend this module with custom loaders.
"""

import os
from typing import Optional

import pandas as pd


def load_raw_data(data_path: str) -> Optional[pd.DataFrame]:
    """
    Load raw data from CSV.

    Args:
        data_path: Path to CSV file

    Returns:
        DataFrame or None if file not found
    """
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)


def create_synthetic_training_data(n_samples: int = 1000, n_features: int = 10, seed: int = 42) -> tuple:
    """Create synthetic training data for demo/testing."""
    import numpy as np

    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    return X, y


def create_synthetic_inference_data(n_samples: int = 100, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create synthetic inference data for demo/testing."""
    import numpy as np

    np.random.seed(seed)
    return pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
