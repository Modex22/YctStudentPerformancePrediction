import pandas as pd
import pytest

from src.batch import BatchError, build_template_csv, parse_upload, run_batch, validate_and_prepare
from src.predict import build_counsellor_note


def test_template_csv_round_trips_through_the_full_pipeline():
    csv_text = build_template_csv()
    df = parse_upload("template.csv", csv_text.encode("utf-8"))
    df, identity_map, missing = validate_and_prepare(df)

    assert missing == []
    assert set(identity_map) == {"name", "matric_no", "department", "level"}

    outcome = run_batch(df, identity_map)
    assert outcome["summary"]["n_students"] == 3
    assert outcome["warnings"] == []
    for row in outcome["results"]:
        assert 0 <= row["predicted_score"] <= 100
        assert row["classification"] in {"Fail", "Pass", "Lower Credit", "Upper Credit", "Distinction"}
        assert row["counsellor_note"]


def test_missing_feature_columns_are_reported():
    df = parse_upload("bad.csv", b"name,age\nJohn,18\n")
    df, identity_map, missing = validate_and_prepare(df)
    assert "study_hours_per_week" in missing
    assert len(missing) > 0


def test_unsupported_file_type_raises_batch_error():
    with pytest.raises(BatchError):
        parse_upload("notes.txt", b"hello")


def test_non_numeric_row_is_skipped_with_a_warning():
    csv_text = build_template_csv().replace(",20,90,74,", ",not-a-number,90,74,")

    parsed = parse_upload("roster.csv", csv_text.encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 2  # one row skipped
    assert any("not-a-number" in w for w in outcome["warnings"])


def test_unrecognized_categorical_value_is_flagged_but_not_fatal():
    csv_text = build_template_csv()
    df = pd.read_csv(__import__("io").StringIO(csv_text))
    df.loc[0, "gender"] = "Other"
    buf = __import__("io").StringIO()
    df.to_csv(buf, index=False)

    parsed = parse_upload("roster.csv", buf.getvalue().encode("utf-8"))
    parsed, identity_map, missing = validate_and_prepare(parsed)
    outcome = run_batch(parsed, identity_map)

    assert outcome["summary"]["n_students"] == 3  # still processed
    assert any("unrecognized 'gender'" in w for w in outcome["warnings"])


def test_counsellor_note_mentions_the_driving_factor_for_at_risk_students():
    student = {
        "attendance_percentage": 50, "study_hours_per_week": 20, "sleep_hours": 7,
        "part_time_job": "No", "tutoring_support": "Yes",
    }
    note = build_counsellor_note(student, "At Risk")
    assert "attendance" in note.lower()


def test_counsellor_note_is_positive_for_excellent_category():
    note = build_counsellor_note({}, "Excellent")
    assert "outstanding" in note.lower()
