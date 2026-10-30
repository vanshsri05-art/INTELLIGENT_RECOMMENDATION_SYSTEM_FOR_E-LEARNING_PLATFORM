from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .content_based import (
    courses,
    similarity_matrix
)

from .collaborative import (
    student_course_matrix,
    student_similarity
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


students = pd.read_csv(
    DATA_DIR / "students.csv"
)

interactions = pd.read_csv(
    DATA_DIR / "interactions.csv"
)

student_features = pd.read_csv(
    DATA_DIR / "student_features.csv"
)

enrollments_path = DATA_DIR / "enrollments.csv"


def hybrid_recommendations(
    student_id,
    top_n=5
):

    if student_id not in students[
        "student_id"
    ].values:
        return pd.DataFrame()

    student_courses = interactions[
        interactions["student_id"] == student_id
    ]["course_id"].unique()

    if enrollments_path.exists():

        enrollments = pd.read_csv(
            enrollments_path
        )

        enrolled_courses = enrollments[
            enrollments["student_id"] == student_id
        ]["course_id"].unique()

        if len(enrolled_courses) > 0:

            student_courses = np.unique(
                np.concatenate(
                    [
                        student_courses,
                        enrolled_courses
                    ]
                )
            )

    # -----------------------------
    # Content-based score
    # -----------------------------

    if len(student_courses) > 0:

        course_indices = courses[
            courses["course_id"].isin(
                student_courses
            )
        ].index.tolist()

        if course_indices:

            content_scores = (
                similarity_matrix[
                    course_indices
                ].mean(axis=0)
            )

        else:
            content_scores = np.zeros(
                len(courses)
            )

    else:
        content_scores = np.zeros(
            len(courses)
        )

    # -----------------------------
    # Collaborative score
    # -----------------------------

    collaborative_scores = np.zeros(
        len(courses)
    )

    if student_id in student_course_matrix.index:

        student_index = (
            student_course_matrix.index.get_loc(
                student_id
            )
        )

        similarity_scores = student_similarity[
            student_index
        ]

        similar_students = np.argsort(
            similarity_scores
        )[::-1]

        similar_students = [
            index
            for index in similar_students
            if index != student_index
        ][:20]

        for similar_index in similar_students:

            similarity = similarity_scores[
                similar_index
            ]

            similar_student_id = (
                student_course_matrix.index[
                    similar_index
                ]
            )

            ratings = student_course_matrix.loc[
                similar_student_id
            ]

            for course_id, rating in ratings.items():

                if rating <= 0:
                    continue

                course_index = courses.index[
                    courses["course_id"] == course_id
                ]

                if len(course_index) == 0:
                    continue

                index = course_index[0]

                collaborative_scores[index] += (
                    similarity * rating
                )

    # -----------------------------
    # Normalize scores
    # -----------------------------

    scaler = MinMaxScaler()

    content_normalized = (
        scaler.fit_transform(
            content_scores.reshape(-1, 1)
        ).flatten()
    )

    collaborative_normalized = (
        scaler.fit_transform(
            collaborative_scores.reshape(-1, 1)
        ).flatten()
    )

    # -----------------------------
    # Student performance
    # -----------------------------

    performance_score = np.zeros(
        len(courses)
    )

    student_data = student_features[
        student_features["student_id"] == student_id
    ]

    if not student_data.empty:

        student_info = student_data.iloc[0]

        quiz_score = float(
            student_info.get(
                "average_quiz_score",
                0
            )
        )

        progress = float(
            student_info.get(
                "average_progress",
                0
            )
        )

        performance = (
            (quiz_score / 100.0) * 0.60
            +
            (progress / 100.0) * 0.40
        )

        performance_score = np.full(
            len(courses),
            performance
        )

    # -----------------------------
    # Original hybrid formula
    # -----------------------------

    final_score = (
        0.45 * content_normalized
        +
        0.35 * collaborative_normalized
        +
        0.20 * performance_score
    )

    # -----------------------------
    # Result
    # -----------------------------

    recommendations = courses.copy()

    recommendations["content_score"] = (
        content_normalized
    )

    recommendations["collaborative_score"] = (
        collaborative_normalized
    )

    recommendations["performance_score"] = (
        performance_score
    )

    recommendations["hybrid_score"] = (
        final_score
    )

    recommendations = recommendations[
        ~recommendations["course_id"].isin(
            student_courses
        )
    ]

    recommendations = recommendations.sort_values(
        "hybrid_score",
        ascending=False
    )

    return recommendations[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "rating",
            "content_score",
            "collaborative_score",
            "performance_score",
            "hybrid_score"
        ]
    ].head(top_n)