"""Prediction: MyPredictor - implements BasePredictor.

The predictor focuses solely on model logic.  Data ingestion and output
writing are handled by the framework's InvocationStrategy (InProcess,
SparkBatch, or FastAPI).

Run directly for local development/debugging:
    python example_model/predict.py
    python -m example_model.predict
"""

import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent
for _p in [str(_repo_root), str(_repo_root / "mlplatform")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from mlplatform.core.predictor import BasePredictor

import example_model.constants as cons


class MyPredictor(BasePredictor):
    """Load model and scaler, run predictions on inference data."""

    def load_model(self) -> Any:
        """Load model and scaler from storage via the ArtifactRegistry."""
        self._model = self.context.load_artifact(cons.MODEL_ARTIFACT)
        self._scaler = self.context.load_artifact(cons.SCALER_ARTIFACT)
        return self._model

    def predict(self, data: Any) -> pd.DataFrame:
        """Run prediction on a chunk of data.

        Receives a DataFrame from the invocation strategy (InProcess reads
        from CSV/BQ, SparkBatch passes a partition).  Returns the input
        DataFrame with a ``prediction`` column appended.
        """
        model = getattr(self, "_model", None)
        scaler = getattr(self, "_scaler", None)
        if model is None or scaler is None:
            raise RuntimeError("Model/scaler not loaded. Call load_model first.")

        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        feature_cols = [c for c in df.columns if c in cons.FEATURE_COLUMNS]
        if not feature_cols:
            feature_cols = [
                c for c in df.columns
                if str(c) != "target" and not str(c).startswith("pred")
            ]
        X = df[feature_cols] if feature_cols else df
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        return df.assign(prediction=predictions)


if __name__ == "__main__":
    import argparse

    from example_model.utils import load_file
    from mlplatform.runner import dev_predict

    parser = argparse.ArgumentParser(description="Run local prediction.")
    parser.add_argument(
        "--profile", default="local", choices=["local", "local-spark"],
        help="local = in-process, local-spark = PySpark mapInPandas",
    )
    parser.add_argument(
        "--input", help="Path to CSV/Parquet file (skip framework I/O)",
    )
    args = parser.parse_args()

    data = load_file(args.input) if args.input else None
    result = dev_predict(
        dag_path="example_model/pipeline/predict.yaml",
        data=data,
        profile=args.profile,
    )
    print(result)
