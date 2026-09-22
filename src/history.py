"""
Opt-in history: a student (or lecturer, on their behalf) can choose to save
a "Check My Score" result so it's possible to see a trend across terms
instead of one static snapshot. This is explicitly NOT automatic — nothing
here runs unless save_entry() is called from a request that ticked the
"save to history" box (see /quick-check/predict in app.py). Batch uploads
are never saved.

Storage is a small SQLite file. On a host with an ephemeral filesystem
(e.g. Render's free tier without a persistent disk), this resets on every
redeploy — history isn't durable there without adding a paid persistent
disk or an external database. Documented in the README.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.config import HISTORY_DB_PATH


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(HISTORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            matric_no TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            exam_score REAL NOT NULL,
            test_score REAL NOT NULL,
            assignment_score REAL NOT NULL,
            practical_score REAL NOT NULL,
            predicted_score REAL NOT NULL,
            classification TEXT NOT NULL,
            pass_fail TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_history_matric ON history(matric_no)")
    return conn


def save_entry(matric_no: str, student: dict[str, Any], result: dict[str, Any]) -> None:
    """Record one saved check for a matric number. Called only when the
    person submitting explicitly opted in."""
    matric_no = matric_no.strip()
    if not matric_no:
        raise ValueError("matric_no is required to save to history")

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO history
                (matric_no, saved_at, exam_score, test_score, assignment_score,
                 practical_score, predicted_score, classification, pass_fail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                matric_no,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                student["exam_score"],
                student["test_score"],
                student["assignment_score"],
                student["practical_score"],
                result["predicted_score"],
                result["classification"],
                result["pass_fail"],
            ),
        )


def get_history(matric_no: str) -> list[dict[str, Any]]:
    """All saved entries for a matric number, oldest first."""
    matric_no = matric_no.strip()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE matric_no = ? ORDER BY saved_at ASC",
            (matric_no,),
        ).fetchall()
    return [dict(row) for row in rows]


def build_trend_chart(
    entries: list[dict[str, Any]],
    width: int = 480,
    height: int = 170,
    pad_x: int = 24,
    pad_y: int = 18,
) -> dict[str, Any] | None:
    """Pixel coordinates + SVG path strings for a simple line-and-area trend
    chart of predicted_score over saved entries, oldest to newest. Computed
    server-side (this app has no chart JS anywhere else either) so the
    template just drops the paths into an inline <svg>. None if there's
    nothing to plot."""
    if not entries:
        return None

    n = len(entries)
    plot_w = width - 2 * pad_x
    plot_h = height - 2 * pad_y

    def x_at(i: int) -> float:
        return pad_x if n == 1 else pad_x + (i / (n - 1)) * plot_w

    def y_at(score: float) -> float:
        return pad_y + plot_h - (max(0.0, min(100.0, score)) / 100) * plot_h

    points = [
        {
            "cx": round(x_at(i), 1),
            "cy": round(y_at(e["predicted_score"]), 1),
            "score": e["predicted_score"],
            "date": e["saved_at"][:10],
        }
        for i, e in enumerate(entries)
    ]

    baseline_y = pad_y + plot_h
    if n == 1:
        # A single point has no line to draw; still show the dot.
        line_path = ""
        area_path = ""
    else:
        line_path = "M " + " L ".join(f"{p['cx']},{p['cy']}" for p in points)
        area_path = (
            line_path
            + f" L {points[-1]['cx']},{baseline_y} L {points[0]['cx']},{baseline_y} Z"
        )

    return {
        "width": width,
        "height": height,
        "points": points,
        "line_path": line_path,
        "area_path": area_path,
        "baseline_y": baseline_y,
    }


def delete_history(matric_no: str) -> int:
    """Remove every saved entry for a matric number. Returns rows deleted —
    lets a student clear their own record."""
    matric_no = matric_no.strip()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM history WHERE matric_no = ?", (matric_no,))
        return cursor.rowcount
