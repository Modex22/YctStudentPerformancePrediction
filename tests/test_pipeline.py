import pandas as pd
import pytest

from data.generate_dataset import generate_dataset
from src.config import FEATURE_COLUMNS
from src.predict import predict_performance


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return generate_dataset(n_students=300, seed=1)


def test_dataset_shape_and_range(dataset):
    assert len(dataset) == 300
    assert dataset["final_score"].between(0, 100).all()
    assert set(dataset["pass_fail"].unique()) <= {"Pass", "Fail"}


def test_dataset_has_all_feature_columns(dataset):
    for col in FEATURE_COLUMNS:
        assert col in dataset.columns


SAMPLE_STUDENT = {
    "age": 19,
    "gender": "Male",
    "school_type": "Public",
    "study_hours_per_week": 20,
    "attendance_percentage": 92,
    "previous_grade": 78,
    "sleep_hours": 7.5,
    "parental_education": "Bachelors",
    "family_income_level": "Medium",
    "internet_access": "Yes",
    "extracurricular_activities": "Yes",
    "part_time_job": "No",
    "tutoring_support": "Yes",
}


def test_predict_performance_returns_expected_keys():
    result = predict_performance(SAMPLE_STUDENT)
    for key in ("predicted_score", "grade", "performance_category", "pass_fail", "pass_probability", "recommendations"):
        assert key in result
    assert 0 <= result["predicted_score"] <= 100
    assert result["pass_fail"] in {"Pass", "Fail"}
    assert isinstance(result["recommendations"], list) and result["recommendations"]


def test_strong_profile_scores_higher_than_weak_profile():
    weak_student = dict(SAMPLE_STUDENT)
    weak_student.update(
        study_hours_per_week=2,
        attendance_percentage=45,
        previous_grade=30,
        sleep_hours=3,
        part_time_job="Yes",
        tutoring_support="No",
    )
    strong_result = predict_performance(SAMPLE_STUDENT)
    weak_result = predict_performance(weak_student)
    assert strong_result["predicted_score"] > weak_result["predicted_score"]
