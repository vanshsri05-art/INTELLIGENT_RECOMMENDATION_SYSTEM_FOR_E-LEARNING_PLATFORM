"""E-Learning Recommendation System - Streamlit app.

Run from the project root:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_HTML = BASE_DIR / "dashboard" / "index.html"
sys.path.insert(0, str(BASE_DIR))

DEMO_PASSWORD = "1234"  # demo only: replace with real authentication before deploying
PROGRESS_STEPS = [0, 25, 50, 75, 100]

# Newer Streamlit versions replaced use_container_width and components.html.
NEW_ST = hasattr(st, "iframe")
STRETCH = {"width": "stretch"} if NEW_ST else {"use_container_width": True}

st.set_page_config(page_title="E-Learning Recommendation System", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- data helpers
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def load_enrollments() -> pd.DataFrame:
    path = DATA_DIR / "enrollments.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["student_id", "course_id", "progress", "status"])


def save_enrollments(df: pd.DataFrame) -> None:
    df.to_csv(DATA_DIR / "enrollments.csv", index=False)


def start_course(student_id: str, course_id: str) -> bool:
    """Enroll a student. Returns False if they already started the course."""
    df = load_enrollments()
    if ((df["student_id"] == student_id) & (df["course_id"] == course_id)).any():
        return False
    row = pd.DataFrame([{"student_id": student_id, "course_id": course_id,
                         "progress": 0, "status": "Started"}])
    save_enrollments(pd.concat([df, row], ignore_index=True))
    return True


def update_course_progress(student_id: str, course_id: str, progress: int) -> bool:
    df = load_enrollments()
    mask = (df["student_id"] == student_id) & (df["course_id"] == course_id)
    if not mask.any():
        return False
    df.loc[mask, "progress"] = progress
    df.loc[mask, "status"] = "Completed" if progress >= 100 else "In Progress"
    save_enrollments(df)
    return True


@st.cache_resource(show_spinner="Loading recommendation models...")
def load_models():
    from recommendation.explanation import explain_recommendation
    from recommendation.hybrid import hybrid_recommendations
    return hybrid_recommendations, explain_recommendation


# ---------------------------------------------------------------------- state
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("student_id", "")


def logout():
    st.session_state.logged_in = False
    st.session_state.student_id = ""


# ---------------------------------------------------------------------- login
def login_page():
    students = load_csv("students.csv")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.title("🎓 E-Learning Recommendation System")
        st.write("Personalized learning recommendations for students")
        st.divider()
        with st.form("login"):
            student_id = st.text_input("Student ID", placeholder="Example: STU0001")
            password = st.text_input("Password", type="password", placeholder="Demo password: 1234")
            submitted = st.form_submit_button("Login", **STRETCH)
        if not submitted:
            return
        student_id = student_id.strip().upper()
        if not student_id or not password:
            st.warning("Please enter Student ID and Password.")
        elif password != DEMO_PASSWORD:
            st.error("Incorrect password.")
        elif student_id not in set(students["student_id"]):
            st.error("Student ID not found.")
        else:
            st.session_state.logged_in = True
            st.session_state.student_id = student_id
            st.rerun()


# ---------------------------------------------------------------------- tabs
def tab_recommendations(student_id: str):
    hybrid_recommendations, explain_recommendation = load_models()
    recs = hybrid_recommendations(student_id, top_n=5)
    st.subheader("Recommended for you")
    if recs.empty:
        st.info("No recommendations available for this student yet.")
        return
    for _, row in recs.iterrows():
        with st.container(border=True):
            st.markdown(f"#### 🎓 {row['course_name']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**Category:** {row['category']}")
            c2.write(f"**Level:** {row['level']}")
            c3.write(f"**Rating:** ⭐ {row['rating']}")
            c4.write(f"**Hybrid score:** {row['hybrid_score']:.3f}")
            st.info(f"💡 **Why this course?**\n\n{explain_recommendation(student_id, row)}")
            if st.button("▶ Start course", key=f"rec_start_{row['course_id']}"):
                if start_course(student_id, row["course_id"]):
                    st.success(f"You started {row['course_name']} 🎉")
                else:
                    st.info("You have already started this course.")


def tab_explore(student_id: str):
    courses = load_csv("courses.csv")
    st.subheader("Explore courses")
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search by course name", placeholder="Python, Machine Learning, SQL")
    category = c2.selectbox("Category", ["All"] + sorted(courses["category"].dropna().unique()))
    level = c3.selectbox("Level", ["All", "Beginner", "Intermediate", "Advanced"])

    found = courses
    if query.strip():
        found = found[found["course_name"].str.contains(query, case=False, na=False)]
    if category != "All":
        found = found[found["category"] == category]
    if level != "All":
        found = found[found["level"] == level]

    st.caption(f"{len(found)} course(s) found. Showing the first 12.")
    for _, c in found.head(12).iterrows():
        with st.expander(f"🎓 {c['course_name']}  ·  {c['category']}  ·  {c['level']}"):
            st.write(f"**Rating:** ⭐ {c['rating']}")
            st.write(c["description"])
            if st.button("▶ Start course", key=f"start_{c['course_id']}"):
                if start_course(student_id, c["course_id"]):
                    st.success(f"You started {c['course_name']} 🎉")
                else:
                    st.info("You have already started this course.")


def tab_my_courses(student_id: str):
    st.subheader("My courses")
    enrollments = load_enrollments()
    mine = enrollments[enrollments["student_id"] == student_id].merge(
        load_csv("courses.csv")[["course_id", "course_name", "category", "level"]],
        on="course_id", how="left")
    if mine.empty:
        st.info("You haven't started any courses yet. Use Explore or Recommended to start one.")
        return
    for _, c in mine.iterrows():
        with st.container(border=True):
            st.markdown(f"#### 🎓 {c['course_name']}")
            a, b, s = st.columns(3)
            a.write(f"**Category:** {c['category']}")
            b.write(f"**Level:** {c['level']}")
            s.write(f"**Status:** {c['status']}")
            current = int(c["progress"])
            st.progress(current / 100, text=f"{current}% complete")
            options = PROGRESS_STEPS if current in PROGRESS_STEPS else sorted(PROGRESS_STEPS + [current])
            col1, col2 = st.columns([3, 1])
            chosen = col1.selectbox("Update progress", options, index=options.index(current),
                                    format_func=lambda x: f"{x}%", key=f"progress_{c['course_id']}")
            col2.write("")
            col2.write("")
            if col2.button("💾 Save", key=f"save_{c['course_id']}"):
                if update_course_progress(student_id, c["course_id"], chosen):
                    st.rerun()


def tab_analytics():
    st.subheader("Model performance")
    metrics = json.loads((DATA_DIR / "model_metrics.json").read_text())
    table = pd.DataFrame(metrics["models"]).rename(columns={
        "name": "Model", "precision": f"Precision@{metrics['k']}", "recall": f"Recall@{metrics['k']}"})
    left, right = st.columns([1, 1])
    left.dataframe(table, hide_index=True, **STRETCH)
    best = table.sort_values(f"Precision@{metrics['k']}", ascending=False).iloc[0]
    right.success(f"Best model: **{best['Model']}** "
                  f"(Precision@{metrics['k']} = {best.iloc[1]:.3f}, Recall@{metrics['k']} = {best.iloc[2]:.4f})")
    st.divider()
    st.subheader("Platform dashboard")
    if DASHBOARD_HTML.exists():
        if NEW_ST:
            st.iframe(DASHBOARD_HTML, height=1800)
        else:
            components.html(DASHBOARD_HTML.read_text(encoding="utf-8"), height=1800, scrolling=True)
    else:
        st.warning("Dashboard not built yet. Run `python dashboard/build_dashboard.py`.")


# ------------------------------------------------------------------- main app
def main_page():
    student_id = st.session_state.student_id
    features = load_csv("student_features.csv")
    row = features[features["student_id"] == student_id]

    with st.sidebar:
        st.markdown(f"### 👤 {student_id}")
        if not row.empty:
            info = row.iloc[0]
            st.caption(f"{info['skill_level']} · interests: {info['interests']}")
        st.button("🚪 Logout", on_click=logout, **STRETCH)

    st.title("🎯 Your Personalized Learning Dashboard")

    if not row.empty:
        info = row.iloc[0]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Skill level", info["skill_level"])
        m2.metric("Avg quiz score", f"{info['average_quiz_score']:.1f}%")
        m3.metric("Avg progress", f"{info['average_progress']:.1f}%")
        m4.metric("Courses completed", int(info["courses_completed"]))
        m5.metric("Learning hours", f"{info['total_learning_hours']:.0f}")

    t1, t2, t3, t4 = st.tabs(["Recommended", "Explore", "My courses", "Analytics"])
    with t1:
        tab_recommendations(student_id)
    with t2:
        tab_explore(student_id)
    with t3:
        tab_my_courses(student_id)
    with t4:
        tab_analytics()


if st.session_state.logged_in:
    main_page()
else:
    login_page()
