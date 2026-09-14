"""Shared paths and column definitions used across the pipeline."""
from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

REGRESSOR_PATH = os.path.join(MODELS_DIR, "score_regressor.joblib")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "pass_fail_classifier.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

NUMERIC_FEATURES = [
    "age",
    "study_hours_per_week",
    "previous_grade",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "school_type",
    "parental_education",
    "family_income_level",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "tutoring_support",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REGRESSION_TARGET = "final_score"
CLASSIFICATION_TARGET = "pass_fail"

# The accepted value set for each categorical feature — used to validate
# uploaded spreadsheets (an unrecognized value is still accepted by the
# model's one-hot encoder, just treated as "unknown", so we flag it
# instead of failing silently).
CATEGORY_VALUES = {
    "gender": ["Male", "Female"],
    "school_type": ["Public", "Private"],
    "parental_education": ["No Formal Education", "High School", "Bachelors", "Masters", "PhD"],
    "family_income_level": ["Low", "Medium", "High"],
    "internet_access": ["Yes", "No"],
    "extracurricular_activities": ["Yes", "No"],
    "part_time_job": ["Yes", "No"],
    "tutoring_support": ["Yes", "No"],
}

CATEGORY_BINS = [-0.1, 49.9, 64.9, 79.9, 100]
CATEGORY_LABELS = ["At Risk", "Average", "Good", "Excellent"]

GRADE_BINS = [-0.1, 39.9, 49.9, 59.9, 69.9, 100]
GRADE_LABELS = ["F", "D", "C", "B", "A"]

# Nigerian-polytechnic-style (ND/HND) result classification. These cutoffs
# are an illustrative default modelled loosely on the common NBTE
# Distinction/Credit/Pass banding — they have NOT been confirmed against
# Yabatech's actual grading policy. Adjust them here if you have the
# official cutoffs.
CLASSIFICATION_BINS = [-0.1, 44.9, 49.9, 59.9, 69.9, 100]
CLASSIFICATION_LABELS = ["Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"]

# Non-predictive identity/roster columns a batch spreadsheet may include.
# They're carried through to the results table and dashboard for context
# but are never fed to the model. Each key maps to the accepted column
# name aliases a spreadsheet might use (matched case-insensitively).
IDENTITY_COLUMN_ALIASES = {
    "name": ["name", "student_name", "full_name", "student"],
    "matric_no": ["matric_no", "matric_number", "reg_no", "registration_number", "student_id"],
    "department": ["department", "dept"],
    "level": ["level", "class_level", "programme_level"],
}
IDENTITY_COLUMNS = list(IDENTITY_COLUMN_ALIASES.keys())

MAX_BATCH_ROWS = 1000
