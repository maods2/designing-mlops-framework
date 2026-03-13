"""Shared utilities for example model."""

from pathlib import Path
from typing import Any

import pandas as pd

_PACKAGE_DIR = Path(__file__).resolve().parent


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


def load_sql_template(name: str) -> str:
    """Load a SQL template from the example_model/sql/ directory.

    Args:
        name: Filename (e.g. ``"load_training_data.sql"``).

    Returns:
        Raw SQL string with ``{param}`` placeholders.
    """
    sql_path = _PACKAGE_DIR / "sql" / name
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL template not found: {sql_path}")
    return sql_path.read_text()


def render_sql(template: str, params: dict[str, Any]) -> str:
    """Render a SQL template by substituting ``{param}`` placeholders.

    Args:
        template: SQL string with ``{key}`` placeholders.
        params: Mapping of placeholder names to values.

    Returns:
        Rendered SQL ready for execution.
    """
    return template.format(**params)


def run_bq_query(sql: str) -> pd.DataFrame:
    """Execute a BigQuery SQL query and return results as a DataFrame."""
    from google.cloud import bigquery

    client = bigquery.Client()
    return client.query(sql).to_dataframe()
