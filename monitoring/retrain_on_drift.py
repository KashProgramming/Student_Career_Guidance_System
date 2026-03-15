from __future__ import annotations

import argparse

from ml_pipeline.train_pipeline import run_training
from monitoring.drift_detection import detect_drift


def retrain_if_drift(data_version: str, psi_threshold: float) -> bool:
    scores = detect_drift(data_version)
    if not scores:
        return False

    if any(score >= psi_threshold for score in scores.values()):
        run_training(data_version=data_version)
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger retraining when drift exceeds a threshold.")
    parser.add_argument("--data-version", default="v1", help="Dataset version")
    parser.add_argument("--psi-threshold", type=float, default=0.2, help="PSI threshold")
    args = parser.parse_args()

    retrained = retrain_if_drift(args.data_version, args.psi_threshold)
    print("Retrained" if retrained else "No drift-triggered retraining")


if __name__ == "__main__":
    main()
