# Student Performance Prediction System

A machine-learning system that predicts a student's likely final exam score,
letter grade and pass/fail risk from academic and lifestyle factors
(study time, attendance, prior grades, sleep, family background, etc.), and
gives a small set of actionable recommendations. Includes a Flask web app
for interactive predictions.

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
   student's data into a score, grade, performance category, pass/fail call,
   and rule-based recommendations.
5. **Web app** (`app.py` + `templates/`) — a form to enter a student's
   details, a result page with the prediction, and a model-info page showing
   how each candidate model performed.

Current results on the synthetic dataset: best regressor is Gradient
Boosting (test R² ≈ 0.76, MAE ≈ 5 points on a 0-100 scale); the pass/fail
classifier reaches ≈ 87% accuracy. Re-run training to regenerate these
numbers — see below.

## Project structure

```
├── app.py                     # Flask web app
├── data/
│   ├── generate_dataset.py    # synthetic dataset generator
│   └── student_performance.csv
├── src/
│   ├── config.py              # paths, feature/target column lists
│   ├── data_preprocessing.py  # ColumnTransformer + data loading
│   ├── train_model.py         # trains, compares, saves models + plots
│   └── predict.py             # loads models, predicts one student
├── models/                    # saved pipelines + metrics.json (generated)
├── reports/figures/           # feature importance & actual-vs-predicted plots
├── templates/, static/        # Flask views
└── tests/test_pipeline.py     # pytest sanity checks
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

Then open http://localhost:5000, fill in the form, and submit to see the
prediction. Visit `/about` for model comparison metrics and charts.

## Predict from the command line

```bash
python -m src.predict
```

Edit the `example_student` dict at the bottom of `src/predict.py`, or import
`predict_performance()` from your own script — it takes a dict with the keys
listed in `src/config.py:FEATURE_COLUMNS` and returns the predicted score,
grade, category, pass/fail call and recommendations.

## Tests

```bash
pytest
```

## Notes on adapting this to real data

Replace `data/student_performance.csv` with real records that (a) keep the
same column names as `FEATURE_COLUMNS` in `src/config.py` (or update that
list) and (b) include a `final_score` (0-100) target and a `pass_fail`
(`Pass`/`Fail`) target, then re-run `python -m src.train_model`. Everything
downstream (preprocessing, training, the web app) works unchanged.
