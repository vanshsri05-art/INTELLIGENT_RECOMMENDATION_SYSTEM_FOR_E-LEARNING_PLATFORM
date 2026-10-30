from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

courses = pd.read_csv(
    BASE_DIR / "data" / "courses.csv"
)

courses["content"] = (
    courses["course_name"].fillna("") + " " +
    courses["category"].fillna("") + " " +
    courses["level"].fillna("") + " " +
    courses["description"].fillna("")
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    courses["content"]
)

from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend_similar_courses(course_id, top_n=5):
    """
    Recommend courses similar to a given course.
    """

    if course_id not in courses["course_id"].values:
        return pd.DataFrame()

    course_index = courses[
        courses["course_id"] == course_id
    ].index[0]

    similarity_scores = similarity_matrix[course_index]

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = [
        i for i in similar_indices
        if i != course_index
    ][:top_n]

    recommendations = courses.iloc[
        similar_indices
    ][
        ["course_id", "course_name", "category", "level", "rating"]
    ].copy()

    recommendations["similarity_score"] = [
        similarity_scores[i]
        for i in similar_indices
    ]

    return recommendations

interactions = pd.read_csv(
    BASE_DIR / "data" / "interactions.csv"
)

def recommend_for_student(student_id, top_n=5):
    """
    Generate personalized course recommendations
    based on courses the student has interacted with.
    """

    # Check whether student exists
    if student_id not in interactions["student_id"].values:
        return pd.DataFrame()

    # Get courses already interacted with
    student_courses = interactions[
        interactions["student_id"] == student_id
    ]["course_id"].unique()

    if len(student_courses) == 0:
        return pd.DataFrame()

    # Find indexes of student's courses
    course_indices = courses[
        courses["course_id"].isin(student_courses)
    ].index.tolist()

    # Calculate average similarity to student's courses
    scores = similarity_matrix[course_indices].mean(axis=0)

    # Create recommendation dataframe
    recommendations = courses.copy()

    recommendations["similarity_score"] = scores

    # Remove courses the student has already interacted with
    recommendations = recommendations[
        ~recommendations["course_id"].isin(student_courses)
    ]

    # Sort by similarity
    recommendations = recommendations.sort_values(
        "similarity_score",
        ascending=False
    )

    return recommendations[
        [
            "course_id",
            "course_name",
            "category",
            "level",
            "rating",
            "similarity_score"
        ]
    ].head(top_n)

if __name__ == "__main__":
    student_id = "STU0001"

    recommendations = recommend_for_student(
        student_id,
        top_n=5
    )

    print("Recommendations for:", student_id)
    print(recommendations)

    for student_id in ["STU0001", "STU0002", "STU0003"]:
    
        print("\n" + "=" * 50)
        print("Student:", student_id)
        print("=" * 50)
    
        print(
            recommend_for_student(
                student_id,
                top_n=5
            )
        )
