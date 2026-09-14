"""
Synthetic student performance dataset generator.

No public labeled dataset was supplied for this project, so this script
builds a realistic synthetic one, using only four inputs a department
already has on record for every student — no self-reported or hard-to-
collect data (attendance, lifestyle, family background, etc.):

    exam_score, test_score, assignment_score, practical_score

final_score is a weighted combination of those four components (see
COMPONENT_WEIGHTS in src/config.py — illustrative, not confirmed against
Yabatech's actual continuous-assessment policy) plus a small amount of
noise, representing real-world effects a fixed formula wouldn't capture
exactly (moderation, rounding, marker variation). The noise is what gives
the model an actual job to do rather than just re-deriving a known
formula.

Run directly to (re)write data/student_performance.csv:
    python data/generate_dataset.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow `python data/generate_dataset.py` to work directly (not just
# `python -m data.generate_dataset`) by ensuring the project root is on
# sys.path before importing the src package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import COMPONENT_WEIGHTS  # noqa: E402

RANDOM_SEED = 42
N_STUDENTS = 2000


def generate_dataset(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Distinct, realistic distributions per component: exams run harder and
    # more spread out than coursework; assignments are the most forgiving.
    exam_score = np.clip(rng.normal(58, 18, n_students), 0, 100)
    test_score = np.clip(rng.normal(65, 15, n_students), 0, 100)
    assignment_score = np.clip(rng.normal(72, 12, n_students), 0, 100)
    practical_score = np.clip(rng.normal(68, 14, n_students), 0, 100)

    noise = rng.normal(0, 3, n_students)

    final_score = (
        COMPONENT_WEIGHTS["exam_score"] * exam_score
        + COMPONENT_WEIGHTS["test_score"] * test_score
        + COMPONENT_WEIGHTS["assignment_score"] * assignment_score
        + COMPONENT_WEIGHTS["practical_score"] * practical_score
        + noise
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame(
        {
            "exam_score": exam_score.round(1),
            "test_score": test_score.round(1),
            "assignment_score": assignment_score.round(1),
            "practical_score": practical_score.round(1),
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
