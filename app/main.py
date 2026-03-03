from fastapi import FastAPI
from app.predict import load_bundle, ModelBundle
from app.schemas import ChurnRequest, ChurnResponse
from app.predict import predict_one

app = FastAPI(title="Telco Churn API", version="0.1.0")

bundle: ModelBundle | None = None
startup_error: str | None = None


@app.on_event("startup")
def _startup() -> None:
    global bundle, startup_error
    try:
        bundle = load_bundle()
    except Exception as e:
        startup_error = repr(e)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "startup_error": startup_error,
    }


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest) -> ChurnResponse:
    if bundle is None:
        raise RuntimeError(f"Model not loaded: {startup_error}")
    proba, will_churn = predict_one(bundle, req.model_dump())
    return ChurnResponse(churn_probability=proba, threshold=bundle.threshold, will_churn=will_churn)