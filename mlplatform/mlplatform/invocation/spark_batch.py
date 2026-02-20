"""Spark batch invocation: distributed prediction via mapInPandas."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class SparkBatchInvocation(InvocationStrategy):
    """Distribute prediction across Spark partitions using mapInPandas.

    Each partition independently loads the model and runs predict_chunk.
    Works identically for local Spark and cloud Spark (Dataproc/VertexAI).
    """

    def __init__(
        self,
        input_path: str | None = None,
        output_path: str | None = None,
        app_name: str = "MLPlatform-Spark-Inference",
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.app_name = app_name

    def invoke(self, predictor: BasePredictor, context: ExecutionContext) -> Any:
        from pyspark.sql import SparkSession
        from pyspark.sql.types import DoubleType, StructField, StructType

        input_path = self.input_path or context.optional_configs.get("input_path")
        output_path = self.output_path or context.optional_configs.get("output_path")

        if not input_path:
            raise ValueError(
                "SparkBatchInvocation requires input_path "
                "(set via constructor or optional_configs['input_path'])"
            )

        spark = SparkSession.builder.appName(self.app_name).getOrCreate()
        context.log.info("Spark session created: %s", self.app_name)

        if input_path.lower().endswith(".parquet"):
            sdf = spark.read.parquet(input_path)
        else:
            sdf = spark.read.option("header", "true").option("inferSchema", "true").csv(input_path)
        context.log.info("Input data loaded from %s", input_path)

        predict_fn = self._build_partition_fn(predictor, context)
        result_schema = StructType(
            list(sdf.schema.fields) + [StructField("prediction", DoubleType())]
        )
        result = sdf.mapInPandas(predict_fn, schema=result_schema)

        if output_path:
            result.write.mode("overwrite").parquet(output_path)
            context.log.info("Predictions written to %s", output_path)
        else:
            result.show()

        return result

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
        }

        def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            from mlplatform.core.context import ExecutionContext as Ctx
            from mlplatform.log import get_logger
            from mlplatform.storage.local import LocalFileSystem
            from mlplatform.tracking.none import NoneTracker

            mod = importlib.import_module(predictor_module)
            pred_cls = getattr(mod, predictor_class_name)

            base = ctx_kwargs["storage_base"]
            worker_ctx = Ctx(
                storage=LocalFileSystem(base_path=base),
                experiment_tracker=NoneTracker(),
                feature_name=ctx_kwargs["feature_name"],
                model_name=ctx_kwargs["model_name"],
                version=ctx_kwargs["version"],
                optional_configs=ctx_kwargs["optional_configs"],
                log=get_logger(f"mlplatform.spark.worker.{ctx_kwargs['model_name']}"),
                _pipeline_type=ctx_kwargs["pipeline_type"],
            )

            pred = pred_cls()
            pred.context = worker_ctx
            pred.load_model()

            for batch in iterator:
                result = pred.predict_chunk(batch)
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({"prediction": result})
                yield result

        return predict_partition
