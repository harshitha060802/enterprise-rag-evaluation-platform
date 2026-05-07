from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHART_PATH = PROJECT_ROOT / "evals" / "retrieval_score.png"


def main():
    score = 1.00
    threshold = 0.80

    labels = ["Retrieval Score", "Minimum Threshold"]
    values = [score, threshold]
    colors = ["#2563eb", "#dc2626"]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=colors)

    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Retrieval Evaluation Score")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH)
    print(f"Chart written to {CHART_PATH}")


if __name__ == "__main__":
    main()