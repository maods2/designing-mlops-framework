"""Unified data loader for CSV, Parquet, and BigQuery."""

from __future__ import annotations

from typing import Any

from mlplatform.data.sources import load_bigquery, load_csv, load_parquet


def load_data(
    config: dict[str, Any],
    format: str = "pandas",
    spark: Any = None,
) -> Any:
    """
    Load data from a configured source.

    Args:
        config: Data source config. Examples:
            - {"type": "csv", "path": "data/inference.csv"}
            - {"type": "parquet", "path": "gs://bucket/data.parquet"}
            - {"type": "bigquery", "table": "project.dataset.table"}
            - {"type": "bigquery", "query": "SELECT ... FROM project.dataset.table"}
        format: "pandas" or "spark"
        spark: SparkSession (required when format="spark")

    Returns:
        pandas.DataFrame or pyspark.sql.DataFrame
    """
    source_type = config.get("type")
    if not source_type:
        raise ValueError("config must have 'type' (csv, parquet, bigquery)")

    if source_type == "csv":
        return load_csv(config, format, spark)
    elif source_type == "parquet":
        return load_parquet(config, format, spark)
    elif source_type == "bigquery":
        return load_bigquery(config, format, spark)
    else:
        raise ValueError(f"Unknown source type: {source_type}. Use csv, parquet, or bigquery.")
