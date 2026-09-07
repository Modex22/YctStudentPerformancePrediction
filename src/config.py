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
    "attendance_percentage",
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

CATEGORY_BINS = [-0.1, 49.9, 64.9, 79.9, 100]
CATEGORY_LABELS = ["At Risk", "Average", "Good", "Excellent"]

GRADE_BINS = [-0.1, 39.9, 49.9, 59.9, 69.9, 100]
GRADE_LABELS = ["F", "D", "C", "B", "A"]
