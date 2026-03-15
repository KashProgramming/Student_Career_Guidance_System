import pandas as pd
from typing import List, Dict


# Features that students can realistically improve
MODIFIABLE_FEATURES = [
    "internships_count",
    "projects_count",
    "certifications_count",
    "skills_score",
    "aptitude_score",
    "leadership_roles",
    "extracurricular_activities"
]

# Realistic increment caps
FEATURE_INCREMENTS = {
    "internships_count": [1, 2],
    "projects_count": [1, 2, 3],
    "certifications_count": [1, 2],
    "skills_score": [0.5, 1.0],
    "aptitude_score": [5, 10],
    "leadership_roles": [1],
    "extracurricular_activities": [1, 2]
}


def simulate_improvement(
    base_input: pd.DataFrame,
    clf_model,
    reg_model,
    feature: str,
    increment: float,
    current_prob: float,
    current_salary: float
) -> Dict:
    """
    Simulate impact of improving a single feature.
    
    Args:
        base_input: Current student feature vector
        clf_model: Placement classifier
        reg_model: Salary regressor
        feature: Feature to modify
        increment: Amount to increase feature by
        current_prob: Current placement probability
        current_salary: Current predicted salary
        
    Returns:
        Dictionary with simulation results
    """
    temp = base_input.copy()
    temp[feature] += increment
    
    new_prob = clf_model.predict_proba(temp)[0][1]
    new_salary = reg_model.predict(temp)[0]
    
    delta_prob = (new_prob - current_prob) * 100  # Convert to percentage points
    delta_salary = new_salary - current_salary
    
    return {
        "feature": feature,
        "increment": increment,
        "new_prob": new_prob,
        "new_salary": new_salary,
        "delta_prob": delta_prob,
        "delta_salary": delta_salary,
        "impact_score": delta_prob * 0.7 + delta_salary * 0.3  # Weighted score
    }


def generate_recommendations(
    X_input: pd.DataFrame,
    clf_model,
    reg_model,
    current_prob: float,
    current_salary: float,
    top_n: int = 5
) -> List[Dict]:
    """
    Generate ranked improvement recommendations.
    
    Args:
        X_input: Preprocessed student input
        clf_model: Placement classifier
        reg_model: Salary regressor
        current_prob: Current placement probability
        current_salary: Current predicted salary
        top_n: Number of recommendations to return
        
    Returns:
        List of recommendation dictionaries, sorted by impact
    """
    recommendations = []
    
    for feature in MODIFIABLE_FEATURES:
        if feature not in X_input.columns:
            continue
            
        for increment in FEATURE_INCREMENTS.get(feature, [1]):
            result = simulate_improvement(
                X_input,
                clf_model,
                reg_model,
                feature,
                increment,
                current_prob,
                current_salary
            )
            recommendations.append(result)
    
    # Sort by impact score
    recommendations.sort(key=lambda x: x["impact_score"], reverse=True)
    
    return recommendations[:top_n]


def format_recommendation(rec: Dict) -> str:
    feature_names = {
        "internships_count": "internships",
        "projects_count": "projects",
        "certifications_count": "certifications",
        "skills_score": "skills score",
        "aptitude_score": "aptitude score",
        "leadership_roles": "leadership roles",
        "extracurricular_activities": "extracurricular activities"
    }
    
    feature_display = feature_names.get(rec["feature"], rec["feature"])
    
    sign_prob = "+" if rec["delta_prob"] > 0 else ""
    sign_salary = "+" if rec["delta_salary"] > 0 else ""
    
    return (
        f"**Increase {feature_display} by {rec['increment']:.1f}**\n"
        f"→ {sign_prob}{rec['delta_prob']:.1f}% placement probability\n"
        f"→ {sign_salary}{rec['delta_salary']:.2f} LPA expected salary"
    )
