from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ml_pipeline.config import DATA_DIR, DEFAULT_DATA_VERSION


def resolve_dataset_path(data_version: Optional[str] = None) -> Path:
    version = data_version or DEFAULT_DATA_VERSION
    metadata_path = DATA_DIR / "metadata.json"

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        versions = metadata.get("versions", {})
        if version in versions:
            resolved = Path(versions[version]["path"])
            if not resolved.is_absolute():
                return (DATA_DIR.parent / resolved).resolve()
            return resolved

    return DATA_DIR / version / "campus_placement_data.csv"


def load_dataset(data_version: Optional[str] = None) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(data_version)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return pd.read_csv(dataset_path)
