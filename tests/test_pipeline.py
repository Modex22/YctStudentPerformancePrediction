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
    "exam_score": 82,
    "test_score": 85,
    "assignment_score": 90,
    "practical_score": 88,
}


def test_predict_performance_returns_expected_keys():
    result = predict_performance(SAMPLE_STUDENT)
    for key in (
        "predicted_score", "grade", "classification", "performance_category",
        "pass_fail", "pass_probability", "recommendations", "counsellor_note",
    ):
        assert key in result
    assert 0 <= result["predicted_score"] <= 100
    assert result["pass_fail"] in {"Pass", "Fail"}
    assert isinstance(result["recommendations"], list) and result["recommendations"]


def test_strong_profile_scores_higher_than_weak_profile():
    weak_student = {
        "exam_score": 20, "test_score": 30, "assignment_score": 35, "practical_score": 25,
    }
    strong_result = predict_performance(SAMPLE_STUDENT)
    weak_result = predict_performance(weak_student)
    assert strong_result["predicted_score"] > weak_result["predicted_score"]


def test_exam_score_has_the_largest_effect_on_predicted_score():
    """exam_score is weighted highest (COMPONENT_WEIGHTS) so a drop there
    should move the prediction more than an equal drop in assignment_score."""
    base = {"exam_score": 80, "test_score": 80, "assignment_score": 80, "practical_score": 80}
    exam_drop = dict(base, exam_score=40)
    assignment_drop = dict(base, assignment_score=40)

    base_score = predict_performance(base)["predicted_score"]
    exam_drop_score = predict_performance(exam_drop)["predicted_score"]
    assignment_drop_score = predict_performance(assignment_drop)["predicted_score"]

    assert (base_score - exam_drop_score) > (base_score - assignment_drop_score)
