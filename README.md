# YABATECH Student Performance Predictor

A machine-learning system that predicts a student's likely final score, an
ND/HND-style classification, and pass/fail risk from academic and lifestyle
factors (study time, attendance, prior grades, sleep, family background,
etc.) — built for a class-roster workflow: upload a spreadsheet of students,
get a class report back.

**Not yet trained on real Yabatech data.** The model is trained on a
synthetic dataset (see below); the app is styled for Yabatech's context
(Department, Level, ND/HND classification) but the predictions themselves
are illustrative until it's retrained on real, anonymized records.

## How it works

1. **Dataset** (`data/generate_dataset.py`) — no labeled dataset was supplied,
   so a realistic synthetic dataset (2,000 students, 13 features) is generated
   from sensible distributions, with the target score built from a weighted
   combination of the features plus noise, mirroring patterns reported in
   education research (study time, attendance and prior grades are the
   strongest predictors; sleep has a sweet spot around 7-8 hours; a part-time
   job has a mild negative effect; etc.). Swap in a real dataset by replacing
   `data/student_performance.csv` with the same columns (see `src/config.py`).
2. **Preprocessing** (`src/data_preprocessing.py`) — numeric features are
   standardized, categorical features are one-hot encoded, via a
   `ColumnTransformer` that's part of the saved model pipeline (no separate
   scaler file to keep in sync).
3. **Training** (`src/train_model.py`) — trains and 5-fold cross-validates
   four regressors (Linear Regression, Ridge, Random Forest, Gradient
   Boosting) to predict the final score, keeps the best by test R², and
   separately trains a Random Forest classifier for pass/fail. Saves both
   pipelines, a `metrics.json`, and feature-importance / actual-vs-predicted
   plots.
4. **Prediction** (`src/predict.py`) — loads the saved pipelines (training
   automatically on first use if they don't exist yet) and turns one
   student's data into a score, grade, ND/HND classification, performance
   category, pass/fail call, rule-based recommendations, and a short
   "counsellor's note" (a one-line report-card-style remark naming the
   biggest driving factor).
5. **Batch upload** (`src/batch.py` + `/`) — the primary workflow: upload a
   `.csv`/`.xlsx` class roster, every row is validated and scored in one
   pass, and you get back a class dashboard (average score, pass rate,
   classification breakdown, score histogram, a ranked at-risk list) plus a
   full per-student results table. A scanning animation plays while the
   file is processed. `/batch/template` downloads a ready-to-fill CSV.
6. **Quick check** (`/quick-check`) — the original one-student form, for a
   single ad-hoc lookup instead of a roster.
7. **Model info** (`/about`) — how each candidate model performed, and the
   caveats on the ND/HND classification cutoffs and the synthetic training
   data.

Current results on the synthetic dataset: best regressor is Gradient
Boosting (test R² ≈ 0.76, MAE ≈ 5 points on a 0-100 scale); the pass/fail
classifier reaches ≈ 87% accuracy. Re-run training to regenerate these
numbers — see below.

## Project structure

```
├── app.py                     # Flask web app (batch upload is "/")
├── data/
│   ├── generate_dataset.py    # synthetic dataset generator
│   └── student_performance.csv
├── src/
│   ├── config.py              # paths, feature/target columns, classification bins
│   ├── data_preprocessing.py  # ColumnTransformer + data loading
│   ├── train_model.py         # trains, compares, saves models + plots
│   ├── predict.py             # loads models, predicts one student
│   ├── batch.py                # parses a roster spreadsheet, runs batch predictions
│   ├── export_web_model.py    # exports trained trees to JSON for a client-side demo
│   └── export_web_charts.py   # exports chart data for that demo
├── models/                    # saved pipelines + metrics.json (generated)
├── reports/figures/           # feature importance & actual-vs-predicted plots
├── templates/, static/        # Flask views (batch_upload, batch_results_fragment, ...)
└── tests/                     # pytest sanity checks
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the models

```bash
python data/generate_dataset.py   # (re)generate the dataset
python -m src.train_model         # train, compare, save models + plots
```

This writes `models/score_regressor.joblib`, `models/pass_fail_classifier.joblib`
and `models/metrics.json`. If you skip this step, the web app and CLI train
automatically the first time a prediction is requested.

## Run the web app

```bash
python app.py
```

Then open http://localhost:5000 — the batch upload page is the home page.
Click "Try an example roster" to see the full flow without your own file,
or download the CSV template and fill in a real class list. `/quick-check`
has the original single-student form, and `/about` has model metrics and
charts.

### Roster spreadsheet format

A `.csv` or `.xlsx` with one row per student. Required columns are the 13
feature columns in `src/config.py:FEATURE_COLUMNS` (age, study hours,
attendance, previous grade, sleep hours, gender, school type, parental
education, family income level, internet access, extracurriculars,
part-time job, tutoring support). Optional identity columns — `name`,
`matric_no`, `department`, `level` (aliases like `student_name` or `dept`
are also recognized) — are carried through to the report for context but
never fed to the model. Unrecognized categorical values or non-numeric
cells are flagged as warnings; rows with an unparseable number are skipped
rather than crashing the whole upload.

## Static client-side demo (no server)

`src/export_web_model.py` walks the fitted scikit-learn pipelines (every
tree's feature/threshold/children/value arrays, plus the scaler and one-hot
encoder parameters) into a JSON file, and `src/export_web_charts.py` exports
the feature-importance and actual-vs-predicted data used by the demo's
charts. A small hand-written JS predictor walks those same trees, so
predictions match the Python model bit-for-bit with nothing running
server-side:

```bash
python -m src.export_web_model web_demo_model.json
python -m src.export_web_charts web_demo_charts.json
```

Used to build the "Grade Forecast" artifact demo (single-student only —
the batch upload workflow is Flask-only for now).

## Predict from the command line

```bash
python -m src.predict
```

Edit the `example_student` dict at the bottom of `src/predict.py`, or import
`predict_performance()` from your own script — it takes a dict with the keys
listed in `src/config.py:FEATURE_COLUMNS` and returns the predicted score,
grade, ND/HND classification, category, pass/fail call, recommendations,
and a counsellor's note.

## Tests

```bash
pytest
```

## Notes on adapting this to real data

Replace `data/student_performance.csv` with real records that (a) keep the
same column names as `FEATURE_COLUMNS` in `src/config.py` (or update that
list) and (b) include a `final_score` (0-100) target and a `pass_fail`
(`Pass`/`Fail`) target, then re-run `python -m src.train_model`. Everything
downstream (preprocessing, training, the web app) works unchanged. The
`CLASSIFICATION_BINS`/`CLASSIFICATION_LABELS` cutoffs in `src/config.py` are
an illustrative default — update them to Yabatech's actual ND/HND grading
policy once known.
