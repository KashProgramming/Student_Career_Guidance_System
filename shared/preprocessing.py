from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


GENDER_MAP = {
    "Male": 2,
    "Female": 1,
    "Other": 3,
}

CITY_TIER_MAP = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3,
}

BOARD_MAP = {
    "State": 1,
    "CBSE": 2,
    "ICSE": 3,
}

HSC_STREAM_MAP = {
    "Science": 1,
    "Commerce": 2,
    "Arts": 3,
}

DEGREE_FIELD_MAP = {
    "Engineering": 1,
    "Business": 2,
    "Other": 3,
    "Arts": 4,
    "Science": 5,
}

CATEGORICAL_MAPS: Dict[str, Dict[str, int]] = {
    "gender": GENDER_MAP,
    "city_tier": CITY_TIER_MAP,
    "ssc_board": BOARD_MAP,
    "hsc_board": BOARD_MAP,
    "hsc_stream": HSC_STREAM_MAP,
    "degree_field": DEGREE_FIELD_MAP,
}

RAW_INPUT_FIELDS: List[str] = [
    "gender",
    "city_tier",
    "ssc_board",
    "hsc_board",
    "hsc_stream",
    "degree_field",
    "hsc_percentage",
    "degree_percentage",
    "technical_skills_score",
    "soft_skills_score",
    "internships_count",
    "projects_count",
    "certifications_count",
    "aptitude_score",
    "work_experience_months",
    "leadership_roles",
    "extracurricular_activities",
    "backlogs",
]

FEATURE_ORDER: List[str] = [
    "gender",
    "city_tier",
    "ssc_board",
    "hsc_board",
    "hsc_stream",
    "degree_field",
    "internships_count",
    "projects_count",
    "certifications_count",
    "aptitude_score",
    "work_experience_months",
    "leadership_roles",
    "extracurricular_activities",
    "backlogs",
    "skills_score",
    "academic_percentage",
]

DROP_COLUMNS: List[str] = [
    "student_id",
    "student_name",
    "mba_percentage",
    "specialization",
    "age",
    "communication_score",
    "ssc_percentage",
]

RAW_DROP_COLUMNS: List[str] = [
    "technical_skills_score",
    "soft_skills_score",
    "hsc_percentage",
    "degree_percentage",
]


@dataclass
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features and drop unused columns."""

    drop_columns: List[str] = None
    raw_drop_columns: List[str] = None

    def __post_init__(self) -> None:
        if self.drop_columns is None:
            self.drop_columns = DROP_COLUMNS.copy()
        if self.raw_drop_columns is None:
            self.raw_drop_columns = RAW_DROP_COLUMNS.copy()

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        if {
            "technical_skills_score",
            "soft_skills_score",
        }.issubset(data.columns):
            data["skills_score"] = (
                data["technical_skills_score"] + data["soft_skills_score"]
            ) / 2

        if {"hsc_percentage", "degree_percentage"}.issubset(data.columns):
            data["academic_percentage"] = (
                data["hsc_percentage"] + data["degree_percentage"]
            ) / 2

        data = data.drop(columns=[col for col in self.drop_columns if col in data.columns])
        data = data.drop(columns=[col for col in self.raw_drop_columns if col in data.columns])

        return data


@dataclass
class CategoricalMapper(BaseEstimator, TransformerMixin):
    """Map categorical strings to ordinal integer values."""

    mappings: Dict[str, Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.mappings is None:
            self.mappings = CATEGORICAL_MAPS.copy()

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        for column, mapping in self.mappings.items():
            if column in data.columns:
                data[column] = data[column].map(mapping)
        return data


@dataclass
class ColumnOrderer(BaseEstimator, TransformerMixin):
    """Ensure columns follow the training feature order."""

    feature_order: List[str]

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [feature for feature in self.feature_order if feature not in X.columns]
        if missing:
            raise ValueError(f"Missing required engineered features: {missing}")
        return X[self.feature_order]
