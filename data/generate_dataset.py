"""
Synthetic student performance dataset generator.

No public labeled dataset was supplied for this project, so this script
builds a realistic synthetic one: features are sampled from sensible
distributions and the target (final exam score) is produced from a
weighted combination of those features plus random noise, mirroring
patterns reported in education-research literature (study time,
attendance and prior grades are the strongest predictors; sleep has a
sweet spot; a part-time job has a mild negative effect; etc.).

Run directly to (re)write data/student_performance.csv:
    python data/generate_dataset.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 2000

PARENTAL_EDUCATION_LEVELS = ["No Formal Education", "High School", "Bachelors", "Masters", "PhD"]
PARENTAL_EDUCATION_BONUS = {
    "No Formal Education": -4,
    "High School": -1,
    "Bachelors": 2,
    "Masters": 4,
    "PhD": 6,
}
FAMILY_INCOME_LEVELS = ["Low", "Medium", "High"]
FAMILY_INCOME_BONUS = {"Low": -3, "Medium": 0, "High": 3}


def generate_dataset(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(15, 23, size=n_students)
    gender = rng.choice(["Male", "Female"], size=n_students)
    school_type = rng.choice(["Public", "Private"], size=n_students, p=[0.65, 0.35])

    study_hours_per_week = np.clip(rng.normal(15, 7, n_students), 0, 40)
    attendance_percentage = np.clip(rng.normal(80, 12, n_students), 30, 100)
    previous_grade = np.clip(rng.normal(62, 15, n_students), 0, 100)
    sleep_hours = np.clip(rng.normal(6.8, 1.3, n_students), 3, 10)

    parental_education = rng.choice(
        PARENTAL_EDUCATION_LEVELS, size=n_students, p=[0.1, 0.35, 0.3, 0.18, 0.07]
    )
    family_income = rng.choice(FAMILY_INCOME_LEVELS, size=n_students, p=[0.35, 0.45, 0.2])
    internet_access = rng.choice(["Yes", "No"], size=n_students, p=[0.78, 0.22])
    extracurricular = rng.choice(["Yes", "No"], size=n_students, p=[0.4, 0.6])
    part_time_job = rng.choice(["Yes", "No"], size=n_students, p=[0.3, 0.7])
    tutoring_support = rng.choice(["Yes", "No"], size=n_students, p=[0.25, 0.75])

    # --- Build the target from a weighted combination of the features ---
    parental_bonus = np.array([PARENTAL_EDUCATION_BONUS[p] for p in parental_education])
    income_bonus = np.array([FAMILY_INCOME_BONUS[i] for i in family_income])
    internet_bonus = np.where(internet_access == "Yes", 2.5, -2.5)
    extracurricular_bonus = np.where(extracurricular == "Yes", 1.5, 0)
    part_time_penalty = np.where(part_time_job == "Yes", -3.0, 0)
    tutoring_bonus = np.where(tutoring_support == "Yes", 3.0, 0)
    # Sleep has a sweet spot around 7-8 hours; deviating either way hurts.
    sleep_penalty = -1.4 * (sleep_hours - 7.5) ** 2

    noise = rng.normal(0, 6, n_students)

    final_score = (
        0.30 * previous_grade
        + 0.22 * attendance_percentage
        + 1.15 * study_hours_per_week
        + parental_bonus
        + income_bonus
        + internet_bonus
        + extracurricular_bonus
        + part_time_penalty
        + tutoring_bonus
        + sleep_penalty
        + noise
        + 12  # recentring constant so the mean lands near a realistic ~60-65
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "school_type": school_type,
            "study_hours_per_week": study_hours_per_week.round(1),
            "attendance_percentage": attendance_percentage.round(1),
            "previous_grade": previous_grade.round(1),
            "sleep_hours": sleep_hours.round(1),
            "parental_education": parental_education,
            "family_income_level": family_income,
            "internet_access": internet_access,
            "extracurricular_activities": extracurricular,
            "part_time_job": part_time_job,
            "tutoring_support": tutoring_support,
            "final_score": final_score,
        }
    )

    df["performance_category"] = pd.cut(
        df["final_score"],
        bins=[-0.1, 49.9, 64.9, 79.9, 100],
        labels=["At Risk", "Average", "Good", "Excellent"],
    )
    df["pass_fail"] = np.where(df["final_score"] >= 50, "Pass", "Fail")

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    out_path = "data/student_performance.csv"
    dataset.to_csv(out_path, index=False)
    print(f"Wrote {len(dataset)} rows to {out_path}")
    print(dataset.describe(include="all").transpose())
