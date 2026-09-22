"""Flask web app for the Yabatech Student Performance Predictor."""
from __future__ import annotations

import json

from flask import Flask, Response, redirect, render_template, request, send_from_directory, url_for

from src.batch import BatchError, build_template_csv, parse_upload, run_batch, validate_and_prepare
from src.config import FEATURE_COLUMNS, FIGURES_DIR, METRICS_PATH
from src.history import build_trend_chart, delete_history, get_history, save_entry
from src.predict import predict_performance

app = Flask(__name__)

FORM_FIELDS = [
    {"name": "exam_score", "label": "Exam score", "type": "number", "min": 0, "max": 60, "default": 36,
     "help": "Out of 60."},
    {"name": "test_score", "label": "Test / CA score", "type": "number", "min": 0, "max": 10, "default": 6,
     "help": "Out of 10."},
    {"name": "assignment_score", "label": "Assignment score", "type": "number", "min": 0, "max": 10, "default": 7,
     "help": "Out of 10."},
    {"name": "practical_score", "label": "Practical score", "type": "number", "min": 0, "max": 20, "default": 13,
     "help": "Out of 20."},
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
    return render_template("batch_upload.html")


@app.route("/batch/template", methods=["GET"])
def batch_template():
    csv_text = build_template_csv()
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=yabatech_roster_template.csv"},
    )


@app.route("/batch/predict", methods=["POST"])
def batch_predict():
    upload = request.files.get("roster")
    if upload is None or upload.filename == "":
        return render_template("batch_error.html", message="No file was uploaded."), 400

    try:
        df = parse_upload(upload.filename, upload.read())
        df, identity_map, missing = validate_and_prepare(df)
    except BatchError as exc:
        return render_template("batch_error.html", message=str(exc)), 400

    if missing:
        return render_template(
            "batch_error.html",
            message=f"The file is missing {len(missing)} required column(s): {', '.join(missing)}.",
            missing=missing,
            all_columns=FEATURE_COLUMNS,
        ), 400

    outcome = run_batch(df, identity_map)
    return render_template("batch_results_fragment.html", outcome=outcome)


@app.route("/quick-check", methods=["GET"])
def quick_check():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/quick-check/predict", methods=["POST"])
def quick_check_predict():
    student = parse_form(request.form)
    result = predict_performance(student)

    save_to_history = request.form.get("save_to_history") == "on"
    matric_no = (request.form.get("matric_no") or "").strip()
    saved = False
    save_error = None
    if save_to_history:
        if matric_no:
            save_entry(matric_no, student, result)
            saved = True
        else:
            save_error = "Matric number is required to save to history."

    return render_template(
        "result.html",
        student=student,
        fields=FORM_FIELDS,
        result=result,
        matric_no=matric_no,
        saved=saved,
        save_error=save_error,
    )


@app.route("/history", methods=["GET"])
def history_lookup():
    return render_template("history_lookup.html")


@app.route("/history", methods=["POST"])
def history_lookup_submit():
    matric_no = (request.form.get("matric_no") or "").strip()
    if not matric_no:
        return render_template("history_lookup.html", error="Enter a matric number.")
    return redirect(url_for("history_view", matric_no=matric_no))


@app.route("/history/<path:matric_no>", methods=["GET"])
def history_view(matric_no: str):
    entries = get_history(matric_no)
    trend = build_trend_chart(entries)
    return render_template("history_view.html", matric_no=matric_no, entries=entries, trend=trend)


@app.route("/history/<path:matric_no>/delete", methods=["POST"])
def history_delete(matric_no: str):
    delete_history(matric_no)
    return redirect(url_for("history_view", matric_no=matric_no))


@app.route("/about", methods=["GET"])
def about():
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
