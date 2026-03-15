from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import MODEL_VERSION
from backend.app.logging_utils import log_prediction
from backend.app.model_loader import load_model_bundle
from backend.app.predictor import make_prediction
from backend.app.recommender import generate_recommendations
from backend.app.schemas import (
    ExplainResponse,
    HealthResponse,
    PredictionResponse,
    RecommendationsResponse,
    StudentInput,
)
from backend.app.shap_service import build_explainability


app = FastAPI(title="Student Career Guide API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models() -> None:
    app.state.model_bundle = load_model_bundle(MODEL_VERSION)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    version_id = app.state.model_bundle.version_id
    return HealthResponse(status="ok", model_version=version_id)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentInput) -> PredictionResponse:
    start = perf_counter()

    payload_dict = payload.model_dump()
    result = make_prediction(app.state.model_bundle, payload_dict)

    duration_ms = (perf_counter() - start) * 1000
    log_prediction(
        {
            "event": "predict",
            "latency_ms": duration_ms,
            "input": payload_dict,
            "output": result,
            "model_version": app.state.model_bundle.version_id,
        }
    )

    return PredictionResponse(
        **result,
        model_version=app.state.model_bundle.version_id,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: StudentInput) -> ExplainResponse:
    df = payload.model_dump()
    X_input = app.state.model_bundle.preprocessing_pipeline.transform(
        pd.DataFrame([df])
    )

    explainability = build_explainability(
        app.state.model_bundle.classification_model,
        X_input,
    )

    return ExplainResponse(
        **explainability,
        model_version=app.state.model_bundle.version_id,
    )


@app.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(payload: StudentInput) -> RecommendationsResponse:
    df = payload.model_dump()
    X_input = app.state.model_bundle.preprocessing_pipeline.transform(
        pd.DataFrame([df])
    )

    placement_prob = app.state.model_bundle.classification_model.predict_proba(X_input)[0][1]
    salary_pred = app.state.model_bundle.regression_model.predict(X_input)[0]

    recs = generate_recommendations(
        X_input,
        app.state.model_bundle.classification_model,
        app.state.model_bundle.regression_model,
        placement_prob,
        salary_pred,
    )

    return RecommendationsResponse(
        recommendations=recs,
        model_version=app.state.model_bundle.version_id,
    )
