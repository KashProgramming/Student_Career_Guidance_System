from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from backend.app.config import CURRENT_MODEL_DIR, MODEL_REGISTRY_PATH, VERSIONED_MODELS_DIR
from ml_pipeline.config import ARTIFACT_NAMES


@dataclass
class ModelBundle:
    classification_model: object
    regression_model: object
    preprocessing_pipeline: object
    feature_metadata: dict
    metrics: dict
    version_id: Optional[str]


def _resolve_model_dir(version_id: Optional[str]) -> Path:
    if version_id:
        return VERSIONED_MODELS_DIR / version_id

    if MODEL_REGISTRY_PATH.exists():
        registry = json.loads(MODEL_REGISTRY_PATH.read_text())
        latest = registry.get("latest_version")
        if latest:
            return VERSIONED_MODELS_DIR / latest

    return CURRENT_MODEL_DIR


def load_model_bundle(version_id: Optional[str] = None) -> ModelBundle:
    model_dir = _resolve_model_dir(version_id)

    classification_model = joblib.load(model_dir / ARTIFACT_NAMES["classification_model"])
    regression_model = joblib.load(model_dir / ARTIFACT_NAMES["regression_model"])
    preprocessing_pipeline = joblib.load(model_dir / ARTIFACT_NAMES["preprocessing_pipeline"])

    feature_metadata_path = model_dir / ARTIFACT_NAMES["feature_metadata"]
    metrics_path = model_dir / ARTIFACT_NAMES["model_metrics"]

    feature_metadata = json.loads(feature_metadata_path.read_text()) if feature_metadata_path.exists() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    resolved_version = model_dir.name if model_dir.name != CURRENT_MODEL_DIR.name else None

    return ModelBundle(
        classification_model=classification_model,
        regression_model=regression_model,
        preprocessing_pipeline=preprocessing_pipeline,
        feature_metadata=feature_metadata,
        metrics=metrics,
        version_id=resolved_version,
    )
