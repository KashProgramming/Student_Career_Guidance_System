gender_map = {
    "Male": 2,
    "Female": 1,
    "Other": 3
}

city_tier_map = {
    "Tier 1": 1,
    "Tier 2": 2,
    "Tier 3": 3
}

ssc_board_map = {
    "State": 1,
    "CBSE": 2,
    "ICSE": 3
}

hsc_board_map = {
    "State": 1,
    "CBSE": 2,
    "ICSE": 3
}

hsc_stream_map = {
    "Science": 1,
    "Commerce": 2,
    "Arts": 3
}

degree_field_map = {
    "Engineering": 1,
    "Business": 2,
    "Other": 3,
    "Arts": 4,
    "Science": 5
}

# Feature order as used in training
FEATURE_ORDER = [
    'gender',
    'city_tier',
    'ssc_board',
    'hsc_board',
    'hsc_stream',
    'degree_field',
    'internships_count',
    'projects_count',
    'certifications_count',
    'aptitude_score',
    'work_experience_months',
    'leadership_roles',
    'extracurricular_activities',
    'backlogs',
    'skills_score',
    'academic_percentage'
]
