"""Prediction data loading and writing utilities.

Inference strategies delegate to these functions rather than implementing
I/O themselves. Supports CSV, Parquet, and BigQuery sources.

Works with both PipelineConfig (V3) and legacy ModelConfig via attribute access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("mlplatform.data.io")


def load_prediction_input(model_cfg: Any) -> pd.DataFrame:
    """Load prediction input from file or BigQuery based on config.

    Accepts PipelineConfig, ModelConfig, or any object with the relevant attributes.
    """
    input_path = getattr(model_cfg, "input_path", None)
    if input_path:
        return _load_from_file(input_path)

    ds_name = getattr(model_cfg, "prediction_dataset_name", None)
    tbl_name = getattr(model_cfg, "prediction_table_name", None)
    if ds_name and tbl_name:
        return _load_from_bigquery(ds_name, tbl_name)

    raise ValueError(
        "No prediction input source configured. "
        "Set input_path (CSV/Parquet) or prediction_dataset_name + prediction_table_name (BigQuery)."
    )


def write_prediction_output(df: pd.DataFrame, model_cfg: Any) -> None:
    """Write prediction output to file or BigQuery based on config."""
    output_path = getattr(model_cfg, "output_path", None)
    out_table = getattr(model_cfg, "prediction_output_dataset_table", None)

    if output_path:
        _write_to_file(df, output_path)
    elif out_table:
        _write_to_bigquery(df, out_table)
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
