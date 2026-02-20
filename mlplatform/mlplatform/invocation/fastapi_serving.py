"""FastAPI serving invocation: expose predict_chunk as a REST endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlplatform.invocation.base import InvocationStrategy

if TYPE_CHECKING:
    from mlplatform.core.context import ExecutionContext
    from mlplatform.core.predictor import BasePredictor


class FastAPIInvocation(InvocationStrategy):
    """Start a FastAPI server exposing predict_chunk as POST /predict.

    Request body: ``{"records": [{"col": val, ...}, ...]}``
    Response body: ``{"predictions": [{"col": val, "prediction": val}, ...]}``
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self.host = host
        self.port = port

    def invoke(self, predictor: BasePredictor, context: ExecutionContext) -> Any:
        import pandas as pd
        import uvicorn
        from fastapi import FastAPI

        context.log.info("Loading model for serving: %s", context.model_name)
        predictor.load_model()
        context.log.info("Model loaded, starting FastAPI server on %s:%d", self.host, self.port)

        app = FastAPI(title=f"MLPlatform - {context.model_name}")

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "model": context.model_name, "version": context.version}

        @app.post("/predict")
        async def predict(body: dict[str, Any]) -> dict[str, Any]:
            records = body.get("records", [])
            df = pd.DataFrame(records)
            result = predictor.predict_chunk(df)
            return {"predictions": result.to_dict(orient="records")}

        uvicorn.run(app, host=self.host, port=self.port)
