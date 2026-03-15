from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    city_tier: Literal["Tier 1", "Tier 2", "Tier 3"]
    ssc_board: Literal["CBSE", "ICSE", "State"]
    hsc_board: Literal["CBSE", "ICSE", "State"]
    hsc_stream: Literal["Science", "Commerce", "Arts"]
    hsc_percentage: float = Field(..., ge=0.0, le=100.0)
    degree_field: Literal["Engineering", "Business", "Science", "Arts", "Other"]
    degree_percentage: float = Field(..., ge=0.0, le=100.0)
    backlogs: int = Field(..., ge=0)
    technical_skills_score: float = Field(..., ge=0.0, le=10.0)
    soft_skills_score: float = Field(..., ge=0.0, le=10.0)
    aptitude_score: float = Field(..., ge=0.0, le=100.0)
    internships_count: int = Field(..., ge=0)
    projects_count: int = Field(..., ge=0)
    certifications_count: int = Field(..., ge=0)
    work_experience_months: int = Field(..., ge=0)
    leadership_roles: int = Field(..., ge=0)
    extracurricular_activities: int = Field(..., ge=0)
    student_name: Optional[str] = None


class PredictionResponse(BaseModel):
    placement_probability: float
    expected_salary: float
    risk_tier: str
    model_version: Optional[str]


class FeatureImpact(BaseModel):
    feature: str
    impact: float


class ExplainResponse(BaseModel):
    feature_impacts: List[FeatureImpact]
    strengths: List[FeatureImpact]
    weaknesses: List[FeatureImpact]
    waterfall_plot_base64: Optional[str]
    bar_plot_base64: Optional[str]
    model_version: Optional[str]


class Recommendation(BaseModel):
    feature: str
    increment: float
    new_prob: float
    new_salary: float
    delta_prob: float
    delta_salary: float
    impact_score: float


class RecommendationsResponse(BaseModel):
    recommendations: List[Recommendation]
    model_version: Optional[str]


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str]
