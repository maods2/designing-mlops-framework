"""Data source backends: CSV, Parquet, BigQuery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        SparkSession = None  # type: ignore


def load_csv(config: dict[str, Any], format: str, spark: Any = None) -> Any:
    """Load data from CSV. Returns pandas or Spark DataFrame."""
    path = config.get("path")
    if not path:
        raise ValueError("CSV source requires 'path'")
    if format == "pandas":
        import pandas as pd

        return pd.read_csv(path)
    elif format == "spark":
        if spark is None:
            raise ValueError("spark session required for Spark format")
        return spark.read.option("header", "true").csv(path)
    raise ValueError(f"Unknown format: {format}")


def load_parquet(config: dict[str, Any], format: str, spark: Any = None) -> Any:
    """Load data from Parquet. Returns pandas or Spark DataFrame."""
    path = config.get("path")
    if not path:
        raise ValueError("Parquet source requires 'path'")
    if format == "pandas":
        import pandas as pd

        return pd.read_parquet(path)
    elif format == "spark":
        if spark is None:
            raise ValueError("spark session required for Spark format")
        return spark.read.parquet(path)
    raise ValueError(f"Unknown format: {format}")


def load_bigquery(config: dict[str, Any], format: str, spark: Any = None) -> Any:
    """Load data from BigQuery. Returns pandas or Spark DataFrame."""
    table = config.get("table")
    query = config.get("query")
    if not table and not query:
        raise ValueError("BigQuery source requires 'table' or 'query'")
    if table and query:
        raise ValueError("BigQuery source: provide either 'table' or 'query', not both")

    if format == "pandas":
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "google-cloud-bigquery required for BigQuery. pip install google-cloud-bigquery"
            )
        client = bigquery.Client()
        if query:
            return client.query(query).to_dataframe()
        return client.query(f"SELECT * FROM `{table}`").to_dataframe()
    elif format == "spark":
        if spark is None:
            raise ValueError("spark session required for Spark format")
        if query:
            return (
                spark.read.format("bigquery")
                .option("query", query)
                .load()
            )
        return (
            spark.read.format("bigquery")
            .option("table", table)
            .load()
        )
    raise ValueError(f"Unknown format: {format}")
