"""Distributed invocation strategy using Spark mapInPandas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    import pandas as pd
    from mlplatform.core.predictor import BasePredictor


class DistributedInvocation(InvocationStrategy):
    """Drives Spark mapInPandas with the predictor for distributed batch inference.

    Requires a SparkSession and input data (path or DataFrame).
    The predictor's load_model() is called per partition, then predict_chunk()
    processes each batch.
    """

    def invoke(self, predictor: "BasePredictor", **kwargs: Any) -> Any:
        storage = kwargs.get("storage")
        model_path = kwargs.get("model_path")
        input_path = kwargs.get("input_path")
        input_source = kwargs.get("input_source")
        output_path = kwargs.get("output_path")
        spark = kwargs.get("spark")

        if spark is None:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.appName("MLPlatform-Distributed").getOrCreate()

        sdf = self._read_input(spark, input_path, input_source)
        predict_fn = self._make_map_fn(type(predictor), storage, model_path)

        from pyspark.sql.types import DoubleType
        result_schema = sdf.schema.add("prediction", DoubleType())
        result = sdf.mapInPandas(predict_fn, schema=result_schema)

        if output_path:
            result.write.mode("overwrite").parquet(output_path)
            return {"output_path": output_path}
        return result

    @staticmethod
    def _read_input(spark: Any, input_path: str | None, input_source: dict | None) -> Any:
        if input_source:
            from mlplatform.data import load_data
            return load_data(input_source, format="spark", spark=spark)
        if input_path:
            if input_path.lower().endswith(".parquet"):
                return spark.read.parquet(input_path)
            return spark.read.option("header", "true").csv(input_path)
        raise ValueError("Either input_path or input_source required for distributed inference")

    @staticmethod
    def _make_map_fn(
        predictor_cls: type,
        storage: Any,
        model_path: str | None,
    ) -> Any:
        """Build the mapInPandas callable that loads model per partition."""
        import pandas as pd

        storage_cfg = {
            "type": type(storage).__name__,
            "base_path": getattr(storage, "base_path", "./artifacts"),
        }

        def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            from mlplatform.storage.local import LocalFileSystem
            part_storage = LocalFileSystem(base_path=str(storage_cfg["base_path"]))
            inst = predictor_cls()
            inst.load_model(part_storage, model_path or "model.pkl")

            for batch in iterator:
                result = inst.predict_chunk(batch)
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame({"prediction": result})
                yield result

        return predict_partition
