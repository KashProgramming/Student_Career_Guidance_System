"""
Test script to verify preprocessing logic matches training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.preprocessing import preprocess_input, validate_input
from encoders import FEATURE_ORDER


def test_preprocessing():
    """Test preprocessing with sample input."""
    
    # Sample student data
    sample_input = {
        "gender": "Male",
        "city_tier": "Tier 1",
        "ssc_board": "CBSE",
        "hsc_board": "CBSE",
        "hsc_stream": "Science",
        "degree_field": "Engineering",
        "hsc_percentage": 75.0,
        "degree_percentage": 70.0,
        "technical_skills_score": 7.0,
        "soft_skills_score": 6.5,
        "internships_count": 2,
        "projects_count": 3,
        "workshops_count": 1,
        "certifications_count": 2,
        "aptitude_score": 65.0,
        "work_experience_months": 6,
        "leadership_roles": 1,
        "extracurricular_activities": 3,
        "backlogs": 0
    }
    
    print("Testing input validation...")
    is_valid, error = validate_input(sample_input)
    print(f"Valid: {is_valid}")
    if not is_valid:
        print(f"Error: {error}")
        return
    
    print("\nTesting preprocessing...")
    X = preprocess_input(sample_input)
    
    print(f"\nOutput shape: {X.shape}")
    print(f"Expected shape: (1, {len(FEATURE_ORDER)})")
    
    print("\nFeature order matches:", list(X.columns) == FEATURE_ORDER)
    
    print("\nProcessed features:")
    for col in X.columns:
        print(f"  {col}: {X[col].values[0]}")
    
    print("\nEngineered features:")
    print(f"  skills_score = (7.0 + 6.5) / 2 = {X['skills_score'].values[0]:.2f}")
    print(f"  academic_percentage = (75.0 + 70.0) / 2 = {X['academic_percentage'].values[0]:.2f}")
    
    print("\nEncoded values:")
    print(f"  gender: Male -> {X['gender'].values[0]} (expected: 2)")
    print(f"  city_tier: Tier 1 -> {X['city_tier'].values[0]} (expected: 1)")
    print(f"  hsc_stream: Science -> {X['hsc_stream'].values[0]} (expected: 1)")
    
    print("\n✅ Preprocessing test completed!")


if __name__ == "__main__":
    test_preprocessing()
