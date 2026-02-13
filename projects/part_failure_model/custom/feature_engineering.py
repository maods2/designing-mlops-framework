"""Feature engineering for part-failure classification.

Data scientists extend this module with custom transforms.
"""

from typing import Optional

import pandas as pd


def build_features(df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
    """
    Build features from raw data.

    Args:
        df: Raw DataFrame (may include target column)
        drop_target: If True and last column looks like target, exclude from features

    Returns:
        Feature DataFrame ready for model
    """
    df = df.dropna()
    if drop_target and df.shape[1] > 1:
        X = df.iloc[:, :-1].copy()
    else:
        X = df.copy()
    return X
