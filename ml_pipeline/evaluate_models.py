from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
    }


def evaluate_regressor(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
        "r2": r2_score(y_test, preds),
    }
