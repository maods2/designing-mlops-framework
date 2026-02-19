"""Shared utilities for template model."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
