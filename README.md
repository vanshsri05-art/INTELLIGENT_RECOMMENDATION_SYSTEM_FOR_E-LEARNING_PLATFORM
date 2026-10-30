# E-Learning Recommendation System

A hybrid course recommender for an online learning platform, with a Streamlit app for students and an analytics dashboard for the platform team.

![Dashboard](docs/dashboard.png)

## What it does

- **Recommends courses** to each student by blending content-based filtering (TF-IDF over course text) with collaborative filtering (similar students), and explains why each course was picked.
- **Lets students act on it**: browse and filter the catalog, start a course, and track progress.
- **Shows platform analytics**: engagement by category, monthly learning hours, progress, ratings, quiz results, learner profile, a sortable course table, and model accuracy.

## Model accuracy (Top-5, held-out interactions)

| Model | Precision@5 | Recall@5 |
|---|---|---|
| Content-Based | 0.424 | 0.2807 |
| Collaborative Filtering | 0.436 | 0.2876 |
| **Hybrid** | **0.440** | **0.2879** |

These values live in `data/model_metrics.json`, which both the app and the dashboard read. Running `python recommendation/evaluation.py` reproduces them.

## Quick start

```bash
pip install -r requirements.txt
python dashboard/build_dashboard.py     # regenerate the dashboard from the CSVs
streamlit run app/app.py                # open the app
```

Demo login: any Student ID from `data/students.csv` (for example `STU0001`) with password `1234`. This password is for demonstration only; replace it with real authentication before deploying.

To view the dashboard on its own, open `dashboard/index.html` in a browser.

## Project layout

```
app/app.py                    Streamlit app: login, recommendations, explore, my courses, analytics
dashboard/build_dashboard.py  Reads data/*.csv and writes dashboard/index.html
dashboard/template.html       Dashboard page template (HTML, CSS, JS, no build step)
recommendation/
  content_based.py            TF-IDF course similarity
  collaborative.py            Student-student similarity
  hybrid.py                   Blended scoring used by the app
  explanation.py              Plain-language reason for each recommendation
  evaluation.py               Precision@5 / Recall@5 evaluation
  evaluation_chart.py         Bar charts of the evaluation results (saved to docs/)
data/                         students, courses, interactions, quiz results, learning history,
                              enrollments (written by the app), model_metrics.json
notebooks/01_EDA.ipynb        Exploratory analysis
```

## Updating the dashboard

After changing any CSV or `data/model_metrics.json`, run `python dashboard/build_dashboard.py`. The app embeds the generated file on its Analytics tab.

## Data notes

The data looks synthetic. Completion rate (12.9% to 16.1%), ratings and progress are nearly identical across categories, levels and skill groups, so differences between segments are small. Treat segment comparisons with caution.

`enrollments.csv` is written by the app when a student starts a course or saves progress.
