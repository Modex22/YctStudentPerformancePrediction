"""Load the trained models and turn a single student's data into a prediction."""
from __future__ import annotations

import os
from typing import Any

import joblib
import pandas as pd

from src.config import (
    CATEGORY_BINS,
    CATEGORY_LABELS,
    CLASSIFIER_PATH,
    FEATURE_COLUMNS,
    GRADE_BINS,
    GRADE_LABELS,
    REGRESSOR_PATH,
)

_regressor = None
_classifier = None

RECOMMENDATION_RULES = [
    ("attendance_percentage", lambda v: v < 75,
     "Attendance is below 75%. Improving class attendance is one of the strongest levers for a better score."),
    ("study_hours_per_week", lambda v: v < 8,
     "Weekly study time is low. Aim for at least 8-10 focused hours per week."),
    ("sleep_hours", lambda v: v < 6 or v > 9,
     "Sleep is outside the 6.5-8.5 hour range associated with the best academic performance."),
    ("part_time_job", lambda v: v == "Yes",
     "A part-time job is competing with study time; consider reducing hours during exam periods."),
    ("tutoring_support", lambda v: v == "No",
     "No tutoring support in place. Extra tutoring correlates with meaningfully higher scores."),
]


def _ensure_models_trained() -> None:
    if not (os.path.exists(REGRESSOR_PATH) and os.path.exists(CLASSIFIER_PATH)):
        from src.train_model import main as train_main

        train_main()


def load_models():
    global _regressor, _classifier
    if _regressor is None or _classifier is None:
        _ensure_models_trained()
        _regressor = joblib.load(REGRESSOR_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _regressor, _classifier


def _to_dataframe(student: dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in student]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return pd.DataFrame([{col: student[col] for col in FEATURE_COLUMNS}])


def build_recommendations(student: dict[str, Any]) -> list[str]:
    notes = []
    for field, condition, message in RECOMMENDATION_RULES:
        try:
            if condition(student[field]):
                notes.append(message)
        except (TypeError, KeyError):
            continue
    if not notes:
        notes.append("All key indicators look healthy — keep up the current routine.")
    return notes


def predict_performance(student: dict[str, Any]) -> dict[str, Any]:
    """Predict a single student's final score, letter grade, category and pass/fail odds.

    `student` must contain every key in config.FEATURE_COLUMNS.
    """
    regressor, classifier = load_models()
    X = _to_dataframe(student)

    predicted_score = float(regressor.predict(X)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    category = pd.cut([predicted_score], bins=CATEGORY_BINS, labels=CATEGORY_LABELS)[0]
    grade = pd.cut([predicted_score], bins=GRADE_BINS, labels=GRADE_LABELS)[0]

    pass_fail_pred = classifier.predict(X)[0]
    proba = classifier.predict_proba(X)[0]
    pass_index = list(classifier.classes_).index("Pass")
    pass_probability = float(proba[pass_index])

    return {
        "predicted_score": round(predicted_score, 1),
        "grade": str(grade),
        "performance_category": str(category),
        "pass_fail": pass_fail_pred,
        "pass_probability": round(pass_probability * 100, 1),
        "recommendations": build_recommendations(student),
    }


if __name__ == "__main__":
    example_student = {
        "age": 18,
        "gender": "Female",
        "school_type": "Public",
        "study_hours_per_week": 6,
        "attendance_percentage": 68,
        "previous_grade": 55,
        "sleep_hours": 5.5,
        "parental_education": "High School",
        "family_income_level": "Low",
        "internet_access": "No",
        "extracurricular_activities": "No",
        "part_time_job": "Yes",
        "tutoring_support": "No",
    }
    result = predict_performance(example_student)
    print(result)
