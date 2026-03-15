import pandas as pd
from encoders import (
    gender_map,
    city_tier_map,
    ssc_board_map,
    hsc_board_map,
    hsc_stream_map,
    degree_field_map,
    FEATURE_ORDER
)


def preprocess_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])
    
    # Create engineered features (matching training)
    df["skills_score"] = (df["technical_skills_score"] + df["soft_skills_score"]) / 2
    df["academic_percentage"] = (df["hsc_percentage"] + df["degree_percentage"]) / 2
    
    # Encode categorical features
    df["gender"] = df["gender"].map(gender_map)
    df["city_tier"] = df["city_tier"].map(city_tier_map)
    df["ssc_board"] = df["ssc_board"].map(ssc_board_map)
    df["hsc_board"] = df["hsc_board"].map(hsc_board_map)
    df["hsc_stream"] = df["hsc_stream"].map(hsc_stream_map)
    df["degree_field"] = df["degree_field"].map(degree_field_map)
    
    # Drop raw columns that were used for engineering
    df = df.drop(columns=[
        "technical_skills_score",
        "soft_skills_score",
        "hsc_percentage",
        "degree_percentage"
    ])
    
    # Ensure correct column order
    df = df[FEATURE_ORDER]
    
    return df


def validate_input(user_input: dict) -> tuple[bool, str]:
    required_fields = [
        "gender", "city_tier", "ssc_board", "hsc_board", "hsc_stream",
        "degree_field", "hsc_percentage", "degree_percentage",
        "technical_skills_score", "soft_skills_score", "internships_count",
        "projects_count", "certifications_count",
        "aptitude_score", "work_experience_months", "leadership_roles",
        "extracurricular_activities", "backlogs"
    ]
    
    for field in required_fields:
        if field not in user_input:
            return False, f"Missing required field: {field}"
    
    # Validate categorical values
    if user_input["gender"] not in gender_map:
        return False, f"Invalid gender value: {user_input['gender']}"
    
    if user_input["city_tier"] not in city_tier_map:
        return False, f"Invalid city_tier value: {user_input['city_tier']}"
    
    if user_input["ssc_board"] not in ssc_board_map:
        return False, f"Invalid ssc_board value: {user_input['ssc_board']}"
    
    if user_input["hsc_board"] not in hsc_board_map:
        return False, f"Invalid hsc_board value: {user_input['hsc_board']}"
    
    if user_input["hsc_stream"] not in hsc_stream_map:
        return False, f"Invalid hsc_stream value: {user_input['hsc_stream']}"
    
    if user_input["degree_field"] not in degree_field_map:
        return False, f"Invalid degree_field value: {user_input['degree_field']}"
    
    return True, ""
