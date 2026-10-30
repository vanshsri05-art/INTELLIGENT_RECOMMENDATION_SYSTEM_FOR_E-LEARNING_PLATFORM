import pandas as pd

from .hybrid import hybrid_recommendations, student_features


def explain_recommendation(student_id, recommendation):

    student_data = student_features[
        student_features["student_id"] == student_id
    ]

    if student_data.empty:
        return "Recommended based on your learning activity."

    student = student_data.iloc[0]

    reasons = []

    # Content similarity
    content_score = float(
        recommendation["content_score"]
    )

    if content_score >= 0.6:
        reasons.append(
            "it is highly similar to courses you have studied"
        )
    elif content_score >= 0.3:
        reasons.append(
            "it is related to courses you have studied"
        )

    # Collaborative filtering
    collaborative_score = float(
        recommendation["collaborative_score"]
    )

    if collaborative_score >= 0.6:
        reasons.append(
            "students with similar learning patterns preferred it"
        )
    elif collaborative_score >= 0.3:
        reasons.append(
            "students with similar learning patterns explored it"
        )

    # Skill level
    skill_level = str(
        student.get("skill_level", "")
    ).lower()

    course_level = str(
        recommendation["level"]
    ).lower()

    if skill_level == course_level:
        reasons.append(
            f"it matches your {recommendation['level']} skill level"
        )

    # Learning performance
    quiz_score = float(
        student.get("average_quiz_score", 0)
    )

    progress = float(
        student.get("average_progress", 0)
    )

    if quiz_score >= 80 and progress >= 70:
        reasons.append(
            "your strong learning performance suggests you are ready for it"
        )
    elif quiz_score >= 60 or progress >= 50:
        reasons.append(
            "your current learning progress supports it"
        )
    else:
        reasons.append(
            "it can help strengthen your learning skills"
        )

    if not reasons:
        return (
            "Recommended based on your learning history "
            "and preferences."
        )

    return (
        "Recommended because "
        + ", ".join(reasons[:3])
        + "."
    )


def add_explanations(student_id, recommendations):

    recommendations = recommendations.copy()

    recommendations["explanation"] = recommendations.apply(
        lambda row: explain_recommendation(
            student_id,
            row
        ),
        axis=1
    )

    return recommendations


if __name__ == "__main__":

    student_id = "STU0001"

    recommendations = hybrid_recommendations(
        student_id,
        top_n=5
    )

    recommendations = add_explanations(
        student_id,
        recommendations
    )

    print()
    print("===== EXPLAINABLE RECOMMENDATIONS =====")
    print()

    for _, row in recommendations.iterrows():

        print("----------------------------------------")

        print(
            "Course:",
            row["course_name"]
        )

        print(
            "Category:",
            row["category"]
        )

        print(
            "Level:",
            row["level"]
        )

        print(
            "Rating:",
            row["rating"]
        )

        print(
            "Hybrid Score:",
            round(row["hybrid_score"], 3)
        )

        print(
            "Why recommended:",
            row["explanation"]
        )

    print("----------------------------------------")