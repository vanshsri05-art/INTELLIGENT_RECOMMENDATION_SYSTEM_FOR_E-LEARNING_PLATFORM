from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# =========================================================
# LOAD DATA
# =========================================================

courses = pd.read_csv(
    DATA_DIR / "courses.csv"
)

interactions = pd.read_csv(
    DATA_DIR / "interactions.csv"
)


# =========================================================
# COURSE CONTENT
# =========================================================

courses["content"] = (
    courses["course_name"].fillna("")
    + " "
    + courses["category"].fillna("")
    + " "
    + courses["level"].fillna("")
    + " "
    + courses["description"].fillna("")
)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    courses["content"]
)

course_similarity = cosine_similarity(
    tfidf_matrix
)


# =========================================================
# CREATE TRAIN / TEST DATA
# =========================================================

np.random.seed(42)

train_parts = []
test_parts = []

for student_id, group in interactions.groupby("student_id"):

    group = group.sample(
        frac=1,
        random_state=42
    )

    if len(group) < 3:
        continue

    test_size = max(
        1,
        int(len(group) * 0.2)
    )

    test_parts.append(
        group.iloc[:test_size]
    )

    train_parts.append(
        group.iloc[test_size:]
    )


train = pd.concat(
    train_parts,
    ignore_index=True
)

test = pd.concat(
    test_parts,
    ignore_index=True
)


# =========================================================
# STUDENT COURSE MATRIX
# =========================================================

matrix = train.pivot_table(
    index="student_id",
    columns="course_id",
    values="rating",
    aggfunc="mean",
    fill_value=0
)

student_similarity = cosine_similarity(
    matrix
)


# =========================================================
# CONTENT RECOMMENDER
# =========================================================

def content_recommendations(
    student_id,
    k=5
):

    if student_id not in matrix.index:
        return []

    known = matrix.loc[
        student_id
    ]

    known_courses = known[
        known > 0
    ].index.tolist()

    if not known_courses:
        return []

    indices = courses[
        courses["course_id"].isin(
            known_courses
        )
    ].index.tolist()

    scores = course_similarity[
        indices
    ].mean(axis=0)

    ranked = np.argsort(
        scores
    )[::-1]

    result = []

    for idx in ranked:

        course_id = courses.iloc[
            idx
        ]["course_id"]

        if course_id in known_courses:
            continue

        result.append(course_id)

        if len(result) == k:
            break

    return result


# =========================================================
# COLLABORATIVE RECOMMENDER
# =========================================================

def collaborative_recommendations(
    student_id,
    k=5
):

    if student_id not in matrix.index:
        return []

    student_index = matrix.index.get_loc(
        student_id
    )

    similarities = student_similarity[
        student_index
    ]

    similar_students = np.argsort(
        similarities
    )[::-1]

    known_courses = set(
        matrix.loc[
            student_id
        ][
            matrix.loc[
                student_id
            ] > 0
        ].index
    )

    scores = {}

    for idx in similar_students[1:11]:

        similarity = similarities[idx]

        other_student = matrix.index[idx]

        ratings = matrix.loc[
            other_student
        ]

        for course_id, rating in ratings.items():

            if rating <= 0:
                continue

            if course_id in known_courses:
                continue

            scores[course_id] = (
                scores.get(course_id, 0)
                + similarity * rating
            )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        course_id
        for course_id, score in ranked[:k]
    ]


# =========================================================
# HYBRID RECOMMENDER
# =========================================================

def hybrid_recommendations(
    student_id,
    k=5
):

    content_result = content_recommendations(
        student_id,
        k=10
    )

    collaborative_result = (
        collaborative_recommendations(
            student_id,
            k=10
        )
    )

    # Combine candidates
    candidates = list(
        dict.fromkeys(
            content_result
            + collaborative_result
        )
    )

    if not candidates:
        return []

    # Give points based on ranking
    scores = {}

    for rank, course_id in enumerate(
        content_result
    ):
        scores[course_id] = (
            scores.get(course_id, 0)
            + (10 - rank) * 0.55
        )

    for rank, course_id in enumerate(
        collaborative_result
    ):
        scores[course_id] = (
            scores.get(course_id, 0)
            + (10 - rank) * 0.45
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        course_id
        for course_id, score in ranked[:k]
    ]


# =========================================================
# METRICS
# =========================================================

def precision_at_5(
    recommended,
    actual
):

    if not recommended:
        return 0

    hits = len(
        set(recommended)
        & set(actual)
    )

    return hits / 5


def recall_at_5(
    recommended,
    actual
):

    if not actual:
        return 0

    hits = len(
        set(recommended)
        & set(actual)
    )

    return hits / len(actual)


# =========================================================
# SAMPLE STUDENTS
# =========================================================

test_students = test[
    "student_id"
].unique()

# Use at most 50 students for fast evaluation
test_students = test_students[:50]

print()
print("Evaluating", len(test_students), "students...")
print()


# =========================================================
# RUN EVALUATION
# =========================================================

content_p = []
content_r = []

collab_p = []
collab_r = []

hybrid_p = []
hybrid_r = []


for count, student_id in enumerate(
    test_students,
    start=1
):

    actual = test[
        test["student_id"] == student_id
    ]["course_id"].unique().tolist()

    # Content
    content = content_recommendations(
        student_id
    )

    content_p.append(
        precision_at_5(
            content,
            actual
        )
    )

    content_r.append(
        recall_at_5(
            content,
            actual
        )
    )

    # Collaborative
    collab = collaborative_recommendations(
        student_id
    )

    collab_p.append(
        precision_at_5(
            collab,
            actual
        )
    )

    collab_r.append(
        recall_at_5(
            collab,
            actual
        )
    )

    # Hybrid
    hybrid = hybrid_recommendations(
        student_id
    )

    hybrid_p.append(
        precision_at_5(
            hybrid,
            actual
        )
    )

    hybrid_r.append(
        recall_at_5(
            hybrid,
            actual
        )
    )

    if count % 10 == 0:
        print(
            f"Processed {count}/"
            f"{len(test_students)} students"
        )


# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame({
    "Model": [
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid"
    ],
    "Precision@5": [
        np.mean(content_p),
        np.mean(collab_p),
        np.mean(hybrid_p)
    ],
    "Recall@5": [
        np.mean(content_r),
        np.mean(collab_r),
        np.mean(hybrid_r)
    ]
})


results["Precision@5"] = results[
    "Precision@5"
].round(4)

results["Recall@5"] = results[
    "Recall@5"
].round(4)


print()
print("=" * 60)
print("RECOMMENDATION SYSTEM EVALUATION")
print("=" * 60)
print()
print(
    results.to_string(index=False)
)
print()

best_model = results.loc[
    results["Precision@5"].idxmax(),
    "Model"
]

print(
    "Best model based on Precision@5:",
    best_model
)