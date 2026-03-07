"""Spark batch invocation: distributed prediction via mapInPandas."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.config.schema import ModelConfig
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class SparkBatchInvocation(InvocationStrategy):
    """Distribute prediction across Spark partitions using mapInPandas.

    Input/output sources are resolved from ModelConfig:
    - File-based: ``input_path`` / ``output_path`` (CSV or Parquet)
    - BigQuery: ``prediction_dataset_name.prediction_table_name`` via
      the spark-bigquery-connector

    Each partition independently loads the model and runs predict.
    Works identically for local Spark and cloud Spark (Dataproc/VertexAI).
    """

    def __init__(self, app_name: str = "MLPlatform-Spark-Inference") -> None:
        self.app_name = app_name

    def invoke(
        self,
        predictor: BasePredictor,
        context: ExecutionContext,
        model_cfg: ModelConfig,
    ) -> Any:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import DoubleType, StructField, StructType

        spark = SparkSession.builder.appName(self.app_name).getOrCreate()
        context.log.info("Spark session created: %s", self.app_name)

        sdf = self._read_input(spark, model_cfg, context)

        predict_fn = self._build_partition_fn(predictor, context)
        result_schema = StructType(
            list(sdf.schema.fields) + [StructField("prediction", DoubleType())]
        )
        result = sdf.mapInPandas(predict_fn, schema=result_schema)

        self._write_output(result, model_cfg, context)
        return result

    @staticmethod
    def _read_input(spark: Any, model_cfg: ModelConfig, context: Any) -> Any:
        """Read input data into a Spark DataFrame from file or BigQuery."""
        if model_cfg.input_path:
            path = model_cfg.input_path
            context.log.info("Spark reading input from file: %s", path)
            if path.lower().endswith(".parquet"):
                return spark.read.parquet(path)
            return spark.read.option("header", "true").option("inferSchema", "true").csv(path)

        if model_cfg.prediction_dataset_name and model_cfg.prediction_table_name:
            bq_table = f"{model_cfg.prediction_dataset_name}.{model_cfg.prediction_table_name}"
            context.log.info("Spark reading input from BigQuery: %s", bq_table)
            return spark.read.format("bigquery").option("table", bq_table).load()

        raise ValueError(
            "SparkBatchInvocation requires input_path or "
            "prediction_dataset_name + prediction_table_name in model config."
        )

    @staticmethod
    def _write_output(result: Any, model_cfg: ModelConfig, context: Any) -> None:
        """Write Spark DataFrame output to file or BigQuery."""
        if model_cfg.output_path:
            result.write.mode("overwrite").parquet(model_cfg.output_path)
            context.log.info("Predictions written to %s", model_cfg.output_path)
        elif model_cfg.prediction_output_dataset_table:
            bq_out = model_cfg.prediction_output_dataset_table
            result.write.format("bigquery").option("table", bq_out).mode("overwrite").save()
            context.log.info("Predictions written to BigQuery: %s", bq_out)
        else:
            result.show()

    @staticmethod
    def _build_partition_fn(
        predictor: BasePredictor, context: ExecutionContext
    ) -> Any:
        """Build a mapInPandas-compatible function that reuses the predictor class."""
        predictor_module = type(predictor).__module__
        predictor_class_name = type(predictor).__name__

        ctx_kwargs = {
            "feature_name": context.feature_name,
            "model_name": context.model_name,
            "version": context.version,
            "optional_configs": context.optional_configs,
            "pipeline_type": context._pipeline_type,
            "storage_base": context.storage.base_path
            if hasattr(context.storage, "base_path")
            else "./artifacts",
            "profile": "local",  # TODO: serialize actual profile name from context
        }

        def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            from mlplatform.core.context import ExecutionContext as Ctx
            from mlplatform.core.prediction_schema import get_schema_from_predictor
            from mlplatform.profiles.registry import get_profile

            mod = importlib.import_module(predictor_module)
            pred_cls = getattr(mod, predictor_class_name)

            # Use the profile system to construct the worker context,
            # rather than hard-coding LocalFileSystem + NoneTracker.
            prof = get_profile(ctx_kwargs["profile"])
            worker_ctx = Ctx.from_profile(
                profile=prof,
                feature_name=ctx_kwargs["feature_name"],
                model_name=ctx_kwargs["model_name"],
                version=ctx_kwargs["version"],
                base_path=ctx_kwargs["storage_base"],
                pipeline_type=ctx_kwargs["pipeline_type"],
                optional_configs=ctx_kwargs["optional_configs"],
            )

            pred = pred_cls()
            pred.context = worker_ctx
            pred.load_model()
            schema = get_schema_from_predictor(pred)

            for batch in iterator:
                if schema:
                    schema.validate(batch)
                result = pred.predict(batch)
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({"prediction": result})
                yield result

        return predict_partition
