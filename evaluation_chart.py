from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


results = pd.DataFrame({
    "Model": [
        "Content-Based",
        "Collaborative Filtering",
        "Hybrid"
    ],
    "Precision@5": [
        0.424,
        0.436,
        0.440
    ],
    "Recall@5": [
        0.2807,
        0.2876,
        0.2879
    ]
})


# Precision@5
plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["Precision@5"]
)

plt.title("Recommendation Models - Precision@5")
plt.xlabel("Model")
plt.ylabel("Precision@5")
plt.ylim(0, 0.5)

for i, value in enumerate(
    results["Precision@5"]
):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    str(Path(__file__).resolve().parent.parent / "docs" / "precision_at_5.png"),
    dpi=300
)

plt.show()


# Recall@5
plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["Recall@5"]
)

plt.title("Recommendation Models - Recall@5")
plt.xlabel("Model")
plt.ylabel("Recall@5")
plt.ylim(0, 0.35)

for i, value in enumerate(
    results["Recall@5"]
):
    plt.text(
        i,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    str(Path(__file__).resolve().parent.parent / "docs" / "recall_at_5.png"),
    dpi=300
)

plt.show()