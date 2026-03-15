from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def get_risk_tier(prob: float) -> str:
    if prob >= 0.85:
        return "Low Risk 🟢"
    if prob >= 0.4:
        return "Moderate Risk 🟡"
    return "High Risk 🔴"


def make_prediction(model_bundle, user_input: Dict) -> Dict:
    df = pd.DataFrame([user_input])
    X_input = model_bundle.preprocessing_pipeline.transform(df)

    placement_prob = model_bundle.classification_model.predict_proba(X_input)[0][1]
    salary_pred = model_bundle.regression_model.predict(X_input)[0]

    risk_tier = get_risk_tier(placement_prob)

    return {
        "placement_probability": float(placement_prob),
        "expected_salary": float(salary_pred),
        "risk_tier": risk_tier
    }
