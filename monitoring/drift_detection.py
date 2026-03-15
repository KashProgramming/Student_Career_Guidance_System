from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from ml_pipeline.data_ingestion import load_dataset
from monitoring.prediction_logger import load_prediction_logs
from shared.preprocessing import RAW_INPUT_FIELDS


def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    if expected.empty or actual.empty:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    psi_value = np.sum((expected_pct - actual_pct) * np.log((expected_pct + 1e-6) / (actual_pct + 1e-6)))
    return float(psi_value)


def detect_drift(data_version: str = "v1") -> Dict[str, float]:
    baseline = load_dataset(data_version)
    logs = load_prediction_logs()

    if logs.empty:
        return {}

    feature_columns: List[str] = [f for f in RAW_INPUT_FIELDS if f in baseline.columns]
    input_columns = [f"input.{field}" for field in RAW_INPUT_FIELDS]
    input_df = logs[input_columns].copy() if set(input_columns).issubset(logs.columns) else pd.DataFrame()
    input_df.columns = [col.replace("input.", "") for col in input_df.columns]

    drift_scores = {}
    for feature in feature_columns:
        if feature in input_df.columns and pd.api.types.is_numeric_dtype(baseline[feature]):
            drift_scores[feature] = _psi(baseline[feature], input_df[feature])

    return drift_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PSI drift scores.")
    parser.add_argument("--data-version", default="v1", help="Baseline dataset version")
    args = parser.parse_args()

    scores = detect_drift(args.data_version)
    for feature, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {score:.4f}")


if __name__ == "__main__":
    main()
