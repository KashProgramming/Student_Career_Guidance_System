from __future__ import annotations

from typing import Dict, List

import pandas as pd


MODIFIABLE_FEATURES = [
    "internships_count",
    "projects_count",
    "certifications_count",
    "skills_score",
    "aptitude_score",
    "leadership_roles",
    "extracurricular_activities",
]

FEATURE_INCREMENTS = {
    "internships_count": [1, 2],
    "projects_count": [1, 2, 3],
    "certifications_count": [1, 2],
    "skills_score": [0.5, 1.0],
    "aptitude_score": [5, 10],
    "leadership_roles": [1],
    "extracurricular_activities": [1, 2],
}


def simulate_improvement(
    base_input: pd.DataFrame,
    clf_model,
    reg_model,
    feature: str,
    increment: float,
    current_prob: float,
    current_salary: float,
) -> Dict:
    temp = base_input.copy()
    temp[feature] += increment

    new_prob = clf_model.predict_proba(temp)[0][1]
    new_salary = reg_model.predict(temp)[0]

    delta_prob = (new_prob - current_prob) * 100
    delta_salary = new_salary - current_salary

    return {
        "feature": feature,
        "increment": increment,
        "new_prob": float(new_prob),
        "new_salary": float(new_salary),
        "delta_prob": float(delta_prob),
        "delta_salary": float(delta_salary),
        "impact_score": float(delta_prob * 0.7 + delta_salary * 0.3),
    }


def generate_recommendations(
    X_input: pd.DataFrame,
    clf_model,
    reg_model,
    current_prob: float,
    current_salary: float,
    top_n: int = 5,
) -> List[Dict]:
    recommendations = []

    for feature in MODIFIABLE_FEATURES:
        if feature not in X_input.columns:
            continue

        for increment in FEATURE_INCREMENTS.get(feature, [1]):
            recommendations.append(
                simulate_improvement(
                    X_input,
                    clf_model,
                    reg_model,
                    feature,
                    increment,
                    current_prob,
                    current_salary,
                )
            )

    recommendations.sort(key=lambda x: x["impact_score"], reverse=True)
    return recommendations[:top_n]
