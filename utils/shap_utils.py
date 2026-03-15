import pandas as pd
import shap


def get_feature_impacts(model, X_input: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    
    # For binary classification, use positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    impact_df = pd.DataFrame({
        "feature": X_input.columns,
        "impact": shap_values[0]
    })
    
    impact_df = impact_df.sort_values("impact", ascending=False)
    return impact_df


def get_top_weaknesses(impact_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    weaknesses = impact_df[impact_df["impact"] < 0].head(top_n)
    return weaknesses


def get_top_strengths(impact_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    strengths = impact_df[impact_df["impact"] > 0].tail(top_n)
    return strengths.sort_values("impact", ascending=False)
