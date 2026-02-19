"""Shared utilities for example model."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame."""
    return pd.read_csv(path)


def load_parquet(path: str) -> pd.DataFrame:
    """Load Parquet file into DataFrame."""
    return pd.read_parquet(path)


def load_file(path: str) -> pd.DataFrame:
    """Load CSV or Parquet based on file extension."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return load_parquet(path)
    return load_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
