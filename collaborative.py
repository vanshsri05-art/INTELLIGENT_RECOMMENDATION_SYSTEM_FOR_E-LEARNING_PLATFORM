from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .content_based import recommend_for_student


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

interactions = pd.read_csv(
    DATA_DIR / "interactions.csv"
)

courses = pd.read_csv(
    DATA_DIR / "courses.csv"
)


student_course_matrix = interactions.pivot_table(
    index="student_id",
    columns="course_id",
    values="rating",
    aggfunc="mean",
    fill_value=0
)


student_similarity = cosine_similarity(
    student_course_matrix
)


def collaborative_recommendations(
    student_id,
    top_n=5
):
    if student_id not in student_course_matrix.index:
        return pd.DataFrame(
            columns=[
                "course_id",
                "collaborative_score"
            ]
        )

    student_index = student_course_matrix.index.get_loc(
        student_id
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
    ]

    similar_students = similar_students[:20]

    student_ratings = student_course_matrix.loc[
        student_id
    ]

    existing_courses = set(
        student_ratings[
            student_ratings > 0
        ].index
    )

    course_scores = {}

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

            if course_id in existing_courses:
                continue

            course_scores[course_id] = (
                course_scores.get(course_id, 0)
                + similarity * rating
            )

    if not course_scores:
        return pd.DataFrame(
            columns=[
                "course_id",
                "collaborative_score"
            ]
        )

    ranked = sorted(
        course_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return pd.DataFrame(
        ranked,
        columns=[
            "course_id",
            "collaborative_score"
        ]
    )


def get_course_details(
    recommendations
):
    if recommendations.empty:
        return recommendations

    result = recommendations.merge(
        courses[
            [
                "course_id",
                "course_name",
                "category",
                "level",
                "rating"
            ]
        ],
        on="course_id",
        how="left"
    )

    return result[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "rating",
            "collaborative_score"
        ]
    ]


def get_collaborative_recommendations(
    student_id,
    top_n=5
):
    recommendations = collaborative_recommendations(
        student_id,
        top_n
    )

    return get_course_details(
        recommendations
    )