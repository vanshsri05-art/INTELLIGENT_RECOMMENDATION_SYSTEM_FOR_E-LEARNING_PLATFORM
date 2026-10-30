"""Build dashboard/index.html from the project's CSV files.

Usage (from the project root):
    python dashboard/build_dashboard.py

The page is a single self-contained HTML file. Open it in a browser, or let
the Streamlit app embed it on the Analytics tab. Re-run after the data or
data/model_metrics.json changes.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent


def rnd(x, n=2):
    return round(float(x), n)


def build_payload():
    courses = pd.read_csv(DATA / "courses.csv")
    students = pd.read_csv(DATA / "students.csv")
    inter = pd.read_csv(DATA / "interactions.csv")
    quiz = pd.read_csv(DATA / "quiz_results.csv")
    hist = pd.read_csv(DATA / "learning_history.csv")
    metrics = json.loads((DATA / "model_metrics.json").read_text())

    inter["done"] = (inter["completed"] == "Yes").astype(int)
    m = inter.merge(courses, on="course_id", suffixes=("", "_c")).merge(
        students[["student_id", "skill_level"]], on="student_id"
    )

    def group(key):
        g = m.groupby(key).agg(
            n=("rating", "size"), rating=("rating", "mean"),
            comp=("done", "mean"), prog=("progress", "mean"),
            time=("time_spent", "mean"),
        ).reset_index()
        return [
            dict(name=r[key], n=int(r["n"]), rating=rnd(r["rating"]),
                 comp=rnd(r["comp"] * 100, 1), prog=rnd(r["prog"], 1),
                 time=rnd(r["time"], 1))
            for _, r in g.iterrows()
        ]

    def histogram(series, lo, hi, step):
        out = []
        for b in range(lo, hi, step):
            last = b + step >= hi
            mask = series >= b if last else (series >= b) & (series < b + step)
            out.append(int(mask.sum()))
        return out

    cg = m.groupby(["course_id", "course_name", "category", "level"]).agg(
        n=("rating", "size"), rating=("rating", "mean"),
        comp=("done", "mean"), prog=("progress", "mean"),
    ).reset_index()

    hist["mo"] = hist["last_accessed"].str[:7]
    mo = hist.groupby("mo").agg(
        n=("hours_spent", "size"), h=("hours_spent", "sum"),
        st=("student_id", "nunique"),
    ).reset_index()

    qa = quiz.groupby("attempts")["quiz_score"].agg(["mean", "size"]).reset_index()

    return {
        "asof": pd.to_datetime(hist["last_accessed"].max()).strftime("%b %Y"),
        "models": metrics["models"],
        "kpi": dict(
            students=int(students["student_id"].nunique()),
            courses=int(len(courses)),
            interactions=int(len(inter)),
            rating=rnd(inter["rating"].mean()),
            completion=rnd(inter["done"].mean() * 100, 1),
            progress=rnd(inter["progress"].mean(), 1),
            quiz=rnd(quiz["quiz_score"].mean(), 1),
            hours=rnd(hist["hours_spent"].sum(), 0),
        ),
        "cat": sorted(group("category"), key=lambda d: -d["n"]),
        "lvl": group("level"),
        "skill": group("skill_level"),
        "courses": [
            dict(id=r.course_id, name=r.course_name, cat=r.category, lvl=r.level,
                 n=int(r.n), rating=rnd(r.rating), comp=rnd(r.comp * 100, 1),
                 prog=rnd(r.prog, 1))
            for r in cg.itertuples()
        ],
        "month": [dict(mo=r.mo, n=int(r.n), h=rnd(r.h, 0), st=int(r.st))
                  for r in mo.itertuples()],
        "rdist": [int((inter["rating"] == k).sum()) for k in range(1, 6)],
        "quizatt": [dict(a=int(r.attempts), n=int(r.size), score=rnd(r.mean, 1))
                    for r in qa.itertuples()],
        "qhist": histogram(quiz["quiz_score"], 30, 100, 10),
        "pdist": histogram(inter["progress"], 0, 100, 10),
        "skillcount": students["skill_level"].value_counts().to_dict(),
        "interests": students["interests"].str.split(", ").explode().value_counts().to_dict(),
    }


def main():
    payload = build_payload()
    template = (HERE / "template.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
