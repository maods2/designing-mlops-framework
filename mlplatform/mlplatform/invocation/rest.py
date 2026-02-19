"""REST invocation strategy for online inference serving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.core.predictor import BasePredictor


class RESTInvocation(InvocationStrategy):
    """Exposes a predictor via a FastAPI HTTP endpoint for online serving."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = port

    def invoke(self, predictor: "BasePredictor", **kwargs: Any) -> Any:
        import pandas as pd
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn

        app = FastAPI(title="MLPlatform Prediction Service")

        class PredictionRequest(BaseModel):
            data: list[dict[str, Any]]

        class PredictionResponse(BaseModel):
            predictions: list[Any]

        @app.post("/predict", response_model=PredictionResponse)
        def predict(request: PredictionRequest) -> PredictionResponse:
            df = pd.DataFrame(request.data)
            result = predictor.predict_chunk(df)
            if isinstance(result, pd.DataFrame):
                preds = result["prediction"].tolist() if "prediction" in result.columns else result.iloc[:, -1].tolist()
            else:
                preds = list(result)
            return PredictionResponse(predictions=preds)

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "healthy"}

        uvicorn.run(app, host=self.host, port=self.port)
