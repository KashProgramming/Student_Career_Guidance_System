from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_DATA_VERSION = "v1"

MODEL_REGISTRY_PATH = MODELS_DIR / "metadata.json"
VERSIONED_MODELS_DIR = MODELS_DIR / "versions"
CURRENT_MODEL_DIR = MODELS_DIR / "current"

ARTIFACT_NAMES = {
    "classification_model": "classification_model.pkl",
    "regression_model": "regression_model.pkl",
    "preprocessing_pipeline": "preprocessing_pipeline.pkl",
    "feature_metadata": "feature_metadata.json",
    "model_metrics": "model_metrics.json",
}
