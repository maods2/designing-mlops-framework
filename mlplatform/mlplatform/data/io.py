"""Prediction data loading and writing utilities.

Inference strategies delegate to these functions rather than implementing
I/O themselves. Supports CSV, Parquet, and BigQuery sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from mlplatform.config.models import ModelConfig

log = logging.getLogger("mlplatform.data.io")


def load_prediction_input(model_cfg: ModelConfig) -> pd.DataFrame:
    """Load prediction input from file or BigQuery based on ModelConfig.

    At least one source must be defined: ``input_path`` (CSV/Parquet) or
    ``prediction_dataset_name`` + ``prediction_table_name`` (BigQuery).
    File-based sources are preferred when both are present.
    """
    if model_cfg.input_path:
        return _load_from_file(model_cfg.input_path)
    if model_cfg.prediction_dataset_name and model_cfg.prediction_table_name:
        return _load_from_bigquery(
            model_cfg.prediction_dataset_name,
            model_cfg.prediction_table_name,
        )
    raise ValueError(
        "No prediction input source configured. "
        "Set input_path (CSV/Parquet) or prediction_dataset_name + prediction_table_name (BigQuery)."
    )


def write_prediction_output(df: pd.DataFrame, model_cfg: ModelConfig) -> None:
    """Write prediction output to file or BigQuery based on ModelConfig.

    Prefers file output when ``output_path`` is set; falls back to BigQuery
    via ``prediction_output_dataset_table``. Silently skips if neither is set.
    """
    if model_cfg.output_path:
        _write_to_file(df, model_cfg.output_path)
    elif model_cfg.prediction_output_dataset_table:
        _write_to_bigquery(df, model_cfg.prediction_output_dataset_table)
    else:
        log.warning("No output destination configured; prediction results not persisted.")


def _load_from_file(path: str) -> pd.DataFrame:
    log.info("Loading prediction input from file: %s", path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_from_bigquery(dataset: str, table: str) -> pd.DataFrame:
    from google.cloud import bigquery

    fqn = f"{dataset}.{table}"
    log.info("Loading prediction input from BigQuery: %s", fqn)
    client = bigquery.Client()
    return client.query(f"SELECT * FROM `{fqn}`").to_dataframe()


def _write_to_file(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    log.info("Prediction output written to file: %s (%d rows)", path, len(df))


def _write_to_bigquery(df: pd.DataFrame, dataset_table: str) -> None:
    from google.cloud import bigquery

    log.info("Writing prediction output to BigQuery: %s (%d rows)", dataset_table, len(df))
    client = bigquery.Client()
    client.load_table_from_dataframe(df, dataset_table).result()
