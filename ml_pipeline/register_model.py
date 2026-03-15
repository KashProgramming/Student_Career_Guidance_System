from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib

from ml_pipeline.config import (
    ARTIFACT_NAMES,
    CURRENT_MODEL_DIR,
    MODEL_REGISTRY_PATH,
    VERSIONED_MODELS_DIR,
)


def _load_registry() -> Dict:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text())
    return {"latest_version": None, "versions": []}


def _write_registry(registry: Dict) -> None:
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def register_model(
    classification_model,
    regression_model,
    preprocessing_pipeline,
    feature_metadata_path: Path,
    metrics: Dict,
    data_version: str,
    version_id: str | None = None,
) -> Dict:
    version = version_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    version_dir = VERSIONED_MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    CURRENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "classification_model": version_dir / ARTIFACT_NAMES["classification_model"],
        "regression_model": version_dir / ARTIFACT_NAMES["regression_model"],
        "preprocessing_pipeline": version_dir / ARTIFACT_NAMES["preprocessing_pipeline"],
        "feature_metadata": version_dir / ARTIFACT_NAMES["feature_metadata"],
        "model_metrics": version_dir / ARTIFACT_NAMES["model_metrics"],
    }

    joblib.dump(classification_model, artifacts["classification_model"])
    joblib.dump(regression_model, artifacts["regression_model"])
    joblib.dump(preprocessing_pipeline, artifacts["preprocessing_pipeline"])

    feature_metadata_path.replace(artifacts["feature_metadata"])
    artifacts["model_metrics"].write_text(json.dumps(metrics, indent=2))

    current_artifacts = {
        "classification_model": CURRENT_MODEL_DIR / ARTIFACT_NAMES["classification_model"],
        "regression_model": CURRENT_MODEL_DIR / ARTIFACT_NAMES["regression_model"],
        "preprocessing_pipeline": CURRENT_MODEL_DIR / ARTIFACT_NAMES["preprocessing_pipeline"],
        "feature_metadata": CURRENT_MODEL_DIR / ARTIFACT_NAMES["feature_metadata"],
        "model_metrics": CURRENT_MODEL_DIR / ARTIFACT_NAMES["model_metrics"],
    }

    joblib.dump(classification_model, current_artifacts["classification_model"])
    joblib.dump(regression_model, current_artifacts["regression_model"])
    joblib.dump(preprocessing_pipeline, current_artifacts["preprocessing_pipeline"])
    current_artifacts["feature_metadata"].write_text(artifacts["feature_metadata"].read_text())
    current_artifacts["model_metrics"].write_text(artifacts["model_metrics"].read_text())

    registry = _load_registry()
    entry = {
        "version_id": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "data_version": data_version,
        "artifacts": {k: str(path) for k, path in artifacts.items()},
        "metrics": metrics,
    }

    registry["latest_version"] = version
    registry["versions"].append(entry)
    _write_registry(registry)

    return entry
