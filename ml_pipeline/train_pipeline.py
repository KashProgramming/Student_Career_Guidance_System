from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from sklearn.model_selection import train_test_split

from ml_pipeline.config import ARTIFACT_NAMES
from ml_pipeline.data_ingestion import load_dataset
from ml_pipeline.data_validation import validate_schema
from ml_pipeline.evaluate_models import evaluate_classifier, evaluate_regressor
from ml_pipeline.feature_engineering import build_preprocessing_pipeline, save_feature_metadata
from ml_pipeline.register_model import register_model
from ml_pipeline.train_classification import train_classifier
from ml_pipeline.train_regression import train_regressor


def run_training(data_version: Optional[str] = None, version_id: Optional[str] = None) -> Dict:
    df = load_dataset(data_version)
    validate_schema(df)

    pipeline = build_preprocessing_pipeline()

    X = df.drop(columns=["placed", "salary_lpa"])
    y = df["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=0,
    )

    pipeline.fit(X_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    clf_model = train_classifier(X_train_transformed, y_train)
    clf_metrics = evaluate_classifier(clf_model, X_test_transformed, y_test)

    df_reg = df[df["placed"] == 1].copy()
    X_reg = df_reg.drop(columns=["placed", "salary_lpa"])
    y_reg = df_reg["salary_lpa"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg,
        y_reg,
        test_size=0.3,
        random_state=0,
    )

    X_train_reg_transformed = pipeline.transform(X_train_reg)
    X_test_reg_transformed = pipeline.transform(X_test_reg)

    reg_model = train_regressor(X_train_reg_transformed, y_train_reg)
    reg_metrics = evaluate_regressor(reg_model, X_test_reg_transformed, y_test_reg)

    metrics = {
        "classification": clf_metrics,
        "regression": reg_metrics,
    }

    feature_metadata_path = Path(ARTIFACT_NAMES["feature_metadata"])
    save_feature_metadata(feature_metadata_path)

    entry = register_model(
        classification_model=clf_model,
        regression_model=reg_model,
        preprocessing_pipeline=pipeline,
        feature_metadata_path=feature_metadata_path,
        metrics=metrics,
        data_version=data_version or "v1",
        version_id=version_id,
    )

    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--data-version", default=None, help="Dataset version (e.g., v1)")
    parser.add_argument("--model-version", default=None, help="Optional model version ID")
    args = parser.parse_args()

    result = run_training(args.data_version, args.model_version)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
