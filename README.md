# YABATECH Student Performance Predictor

A machine-learning system that predicts a student's likely final score, an
ND/HND-style classification, and pass/fail risk from four assessment
components a department already has on record for every student:

- Exam score
- Test / CA score
- Assignment score
- Practical score

No lifestyle, demographic, or self-reported data (study habits, sleep,
family background, attendance, etc.) — just the numbers already sitting in
a results spreadsheet. Built for a class-roster workflow: upload that
spreadsheet, get a class report back.

**Not yet trained on real Yabatech data.** The model is trained on a
synthetic dataset (see below); the app is styled for Yabatech's context
(Department, Level, ND/HND classification) but the predictions themselves
are illustrative until it's retrained on real, anonymized records.

## How it works

1. **Dataset** (`data/generate_dataset.py`) — no labeled dataset was
   supplied, so a synthetic one (2,000 students, 4 features) is generated:
   each component score is sampled from its own realistic distribution
   (exams run harder and more spread out than coursework; assignments are
   the most forgiving), and `final_score` is a weighted combination of the
   four (see `COMPONENT_WEIGHTS` in `src/config.py` — illustrative, not
   confirmed against Yabatech's actual continuous-assessment policy) plus a
   little noise, representing real-world effects a fixed formula wouldn't
   capture exactly (moderation, rounding, marker variation). Swap in a real
   dataset by replacing `data/student_performance.csv` with the same
   columns (see `src/config.py`).
2. **Preprocessing** (`src/data_preprocessing.py`) — the four scores are
   standardized via a `ColumnTransformer` that's part of the saved model
   pipeline (no separate scaler file to keep in sync). There are currently
   no categorical features, but the pipeline still has an (empty)
   categorical branch so one can be added back without restructuring
   anything.
3. **Training** (`src/train_model.py`) — trains and 5-fold cross-validates
   four regressors (Linear Regression, Ridge, Random Forest, Gradient
   Boosting) to predict the final score, keeps the best by test R², and
   separately trains a Random Forest classifier for pass/fail. Saves both
   pipelines, a `metrics.json`, and feature-importance / actual-vs-predicted
   plots. (Feature importance is read from `feature_importances_` for a
   tree ensemble or `|coefficient|` for a linear model — whichever type
   wins a given retrain.)
4. **Prediction** (`src/predict.py`) — loads the saved pipelines (training
   automatically on first use if they don't exist yet) and turns one
   student's four scores into a final score, grade, ND/HND classification,
   performance category, pass/fail call, rule-based recommendations, and a
   short "counsellor's note" (a one-line report-card-style remark naming
   the weakest component(s)).
5. **Batch upload** (`src/batch.py` + `/`) — the primary workflow: upload a
   `.csv`/`.xlsx` class roster, every row is validated and scored in one
   pass, and you get back a class dashboard (average score, pass rate,
   classification breakdown, score histogram, a ranked at-risk list) plus a
   full per-student results table. A scanning animation plays while the
   file is processed. `/batch/template` downloads a ready-to-fill CSV.
6. **Check my score** (`/quick-check`) — the same prediction for one
   student filling in their own four scores, instead of a roster.
7. **Model info** (`/about`) — how each candidate model performed, and the
   caveats on the ND/HND classification cutoffs and the synthetic training
   data.

Current results on the synthetic dataset: best regressor is Linear
Regression (test R² ≈ 0.90, MAE ≈ 2.6 points on a 0-100 scale — high
because `final_score` really is close to a weighted sum of the four inputs
by construction); the pass/fail classifier reaches ≈ 95% accuracy. Re-run
training to regenerate these numbers on real data — see below.

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
│   ├── export_web_model.py    # exports the trained model to JSON for a client-side demo
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
lets one student check their own score, and `/about` has model metrics and
charts.

## Deploying it publicly

**Not Vercel** — this app's dependencies (numpy + pandas + scipy + scikit-learn)
total ~283 MB uncompressed, over Vercel's ~250 MB serverless function limit,
and batch upload needs a real running Python process (pandas parsing an
uploaded file), not a stateless function. Vercel *is* a great fit for the
"Grade Forecast" client-side demo (`src/export_web_model.py`) since that's
pure static HTML/JS with no backend at all.

For the full app (including batch upload), any host that runs a persistent
Python process works — this repo is ready for **Render**
(`render.yaml` + `Procfile`, both already here):

1. Push this repo to your own GitHub account (or use this one directly if
   you have push access).
2. On [render.com](https://render.com): New → Blueprint → connect the repo.
   It reads `render.yaml` automatically — free tier, no config needed.
3. Or manually: New → Web Service → connect the repo → Render detects
   Python, build command `pip install -r requirements.txt`, start command
   `gunicorn app:app` (already set in `render.yaml`).

Railway, Fly.io, PythonAnywhere, or a plain VM all work the same way — the
only requirement is running `gunicorn app:app` (or equivalent) instead of
`python app.py`'s development server. Verified locally before recommending
this: `gunicorn app:app` serves every route (including a full predict
request) identically to the dev server.

### Roster spreadsheet format

A `.csv` or `.xlsx` with one row per student. Required columns are the 4
feature columns in `src/config.py:FEATURE_COLUMNS` — `exam_score`,
`test_score`, `assignment_score`, `practical_score`, each 0-100. Optional
identity columns — `name`, `matric_no`, `department`, `level` (aliases like
`student_name` or `dept` are also recognized) — are carried through to the
report for context but never fed to the model. Non-numeric cells are
flagged as warnings and that row is skipped rather than crashing the whole
upload.

## Static client-side demo (no server)

`src/export_web_model.py` exports the fitted regressor + classifier
pipelines to a JSON file a plain-JS predictor can walk with no Python/Flask
backend required, and `src/export_web_charts.py` exports the
feature-importance and actual-vs-predicted data used by the demo's charts.
The regressor's shape depends on which model type won training — a tree
ensemble is exported as its trees (feature/threshold/children/value per
node), a linear model as its coefficients + intercept — so the exporter
(and the JS that reads its output) branches on `payload.regressor.type`
rather than assuming one shape:

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
listed in `src/config.py:FEATURE_COLUMNS` (`exam_score`, `test_score`,
`assignment_score`, `practical_score`) and returns the predicted score,
grade, ND/HND classification, category, pass/fail call, recommendations,
and a counsellor's note.

## Tests

```bash
pytest
```

## Notes on adapting this to real data

Replace `data/student_performance.csv` with real records that (a) keep the
same column names as `FEATURE_COLUMNS` in `src/config.py` (or update that
list, and `COMPONENT_WEIGHTS`, if the real assessment structure differs)
and (b) include a `final_score` (0-100) target and a `pass_fail`
(`Pass`/`Fail`) target, then re-run `python -m src.train_model`. Everything
downstream (preprocessing, training, the web app) works unchanged. The
`CLASSIFICATION_BINS`/`CLASSIFICATION_LABELS` cutoffs in `src/config.py` are
an illustrative default — update them to Yabatech's actual ND/HND grading
policy once known.
