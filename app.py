"""Flask web app for the Student Performance Prediction System."""
from __future__ import annotations

from flask import Flask, render_template, request, send_from_directory

from src.config import FIGURES_DIR
from src.predict import predict_performance

app = Flask(__name__)

FORM_FIELDS = [
    {"name": "age", "label": "Age", "type": "number", "min": 15, "max": 22, "default": 18},
    {"name": "gender", "label": "Gender", "type": "select", "options": ["Male", "Female"]},
    {"name": "school_type", "label": "School type", "type": "select", "options": ["Public", "Private"]},
    {"name": "study_hours_per_week", "label": "Study hours / week", "type": "number", "min": 0, "max": 40, "default": 15},
    {"name": "attendance_percentage", "label": "Attendance (%)", "type": "number", "min": 0, "max": 100, "default": 80},
    {"name": "previous_grade", "label": "Previous term grade (0-100)", "type": "number", "min": 0, "max": 100, "default": 60},
    {"name": "sleep_hours", "label": "Average sleep hours / night", "type": "number", "min": 0, "max": 12, "default": 7, "step": "0.5"},
    {"name": "parental_education", "label": "Parental education", "type": "select",
     "options": ["No Formal Education", "High School", "Bachelors", "Masters", "PhD"]},
    {"name": "family_income_level", "label": "Family income level", "type": "select", "options": ["Low", "Medium", "High"]},
    {"name": "internet_access", "label": "Internet access at home", "type": "select", "options": ["Yes", "No"]},
    {"name": "extracurricular_activities", "label": "Extracurricular activities", "type": "select", "options": ["Yes", "No"]},
    {"name": "part_time_job", "label": "Part-time job", "type": "select", "options": ["Yes", "No"]},
    {"name": "tutoring_support", "label": "Tutoring support", "type": "select", "options": ["Yes", "No"]},
]

NUMERIC_FIELD_NAMES = {f["name"] for f in FORM_FIELDS if f["type"] == "number"}


def parse_form(form) -> dict:
    student = {}
    for field in FORM_FIELDS:
        raw = form.get(field["name"])
        if field["name"] in NUMERIC_FIELD_NAMES:
            student[field["name"]] = float(raw)
        else:
            student[field["name"]] = raw
    return student


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    student = parse_form(request.form)
    result = predict_performance(student)
    return render_template("result.html", student=student, fields=FORM_FIELDS, result=result)


@app.route("/about", methods=["GET"])
def about():
    import json

    from src.config import METRICS_PATH

    metrics = {}
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        pass
    return render_template("about.html", metrics=metrics)


@app.route("/figures/<path:filename>", methods=["GET"])
def figures(filename):
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
