from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sklearn.pipeline import Pipeline

from shared.preprocessing import (
    CATEGORICAL_MAPS,
    FEATURE_ORDER,
    RAW_INPUT_FIELDS,
    CategoricalMapper,
    ColumnOrderer,
    FeatureEngineer,
)


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("categorical_mapping", CategoricalMapper()),
            ("order_columns", ColumnOrderer(FEATURE_ORDER)),
        ]
    )


def build_feature_metadata() -> Dict:
    return {
        "raw_input_fields": RAW_INPUT_FIELDS,
        "feature_order": FEATURE_ORDER,
        "categorical_maps": CATEGORICAL_MAPS,
    }


def save_feature_metadata(path: Path) -> None:
    metadata = build_feature_metadata()
    path.write_text(json.dumps(metadata, indent=2))
