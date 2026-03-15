from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_REGISTRY_PATH = MODELS_DIR / "metadata.json"
CURRENT_MODEL_DIR = MODELS_DIR / "current"
VERSIONED_MODELS_DIR = MODELS_DIR / "versions"

LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = os.getenv("MODEL_VERSION")
