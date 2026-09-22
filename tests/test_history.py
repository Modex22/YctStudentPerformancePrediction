import os

import pytest

from src.history import build_trend_chart, delete_history, get_history, save_entry
from src.predict import predict_performance


@pytest.fixture(autouse=True)
def _isolated_history_db(tmp_path, monkeypatch):
    """Point HISTORY_DB_PATH at a throwaway file per test so tests never
    touch (or depend on) the real instance/history.db."""
    import src.config as config
    import src.history as history

    db_path = os.path.join(tmp_path, "history_test.db")
    monkeypatch.setattr(config, "HISTORY_DB_PATH", db_path)
    monkeypatch.setattr(history, "HISTORY_DB_PATH", db_path)
    yield


def test_save_and_get_history_round_trips():
    student = {"exam_score": 36, "test_score": 6, "assignment_score": 7, "practical_score": 13}
    result = predict_performance(student)
    save_entry("ND/CS/23/0001", student, result)

    entries = get_history("ND/CS/23/0001")
    assert len(entries) == 1
    assert entries[0]["exam_score"] == 36
    assert entries[0]["predicted_score"] == result["predicted_score"]


def test_save_entry_requires_matric_no():
    student = {"exam_score": 36, "test_score": 6, "assignment_score": 7, "practical_score": 13}
    result = predict_performance(student)
    with pytest.raises(ValueError):
        save_entry("", student, result)


def test_different_matric_numbers_are_isolated():
    student = {"exam_score": 20, "test_score": 4, "assignment_score": 4, "practical_score": 8}
    result = predict_performance(student)
    save_entry("ND/A/1", student, result)
    save_entry("ND/B/1", student, result)

    assert len(get_history("ND/A/1")) == 1
    assert len(get_history("ND/B/1")) == 1
    assert get_history("ND/NOBODY/1") == []


def test_delete_history_removes_only_that_matric_no():
    student = {"exam_score": 20, "test_score": 4, "assignment_score": 4, "practical_score": 8}
    result = predict_performance(student)
    save_entry("ND/A/1", student, result)
    save_entry("ND/B/1", student, result)

    deleted = delete_history("ND/A/1")
    assert deleted == 1
    assert get_history("ND/A/1") == []
    assert len(get_history("ND/B/1")) == 1


def test_build_trend_chart_none_when_empty():
    assert build_trend_chart([]) is None


def test_build_trend_chart_orders_points_left_to_right():
    entries = [
        {"predicted_score": 40.0, "saved_at": "2026-01-01T00:00:00+00:00"},
        {"predicted_score": 70.0, "saved_at": "2026-02-01T00:00:00+00:00"},
    ]
    chart = build_trend_chart(entries)
    assert chart is not None
    assert len(chart["points"]) == 2
    # Higher score -> smaller y (SVG y grows downward), and time moves left to right.
    assert chart["points"][0]["cx"] < chart["points"][1]["cx"]
    assert chart["points"][1]["cy"] < chart["points"][0]["cy"]
