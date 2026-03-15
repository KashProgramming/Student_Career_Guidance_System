from __future__ import annotations

from typing import Dict, List

import argparse

from ml_pipeline.data_ingestion import load_dataset

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "gender",
    "city_tier",
    "ssc_board",
    "hsc_percentage",
    "hsc_board",
    "hsc_stream",
    "degree_percentage",
    "degree_field",
    "internships_count",
    "projects_count",
    "certifications_count",
    "technical_skills_score",
    "soft_skills_score",
    "aptitude_score",
    "work_experience_months",
    "leadership_roles",
    "extracurricular_activities",
    "backlogs",
    "placed",
    "salary_lpa",
]


def validate_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        null_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isnull().any()].tolist()
        raise ValueError(f"Null values found in columns: {null_cols}")

    return {"missing": missing, "extra": extra}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset schema.")
    parser.add_argument("--data-version", default=None, help="Dataset version (e.g., v1)")
    args = parser.parse_args()

    df = load_dataset(args.data_version)
    validate_schema(df)
    print("Schema validation passed.")


if __name__ == "__main__":
    main()
