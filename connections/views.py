"""
Connections output and plotting helpers.
"""

import matplotlib.pyplot as plt

CLUSTERS = 4
COLORS = {0: "🟨", 1: "🟩", 2: "🟦", 3: "🟪"}


def print_final_summary(
    game_id: int,
    guess_history: list[tuple[list[str], int]],
    word_to_level: dict[str, int],
) -> None:
    print("Connections")
    print(f"Puzzle #{game_id}")
    for guess, _ in guess_history:
        row = "".join([COLORS[word_to_level[word]] for word in guess])
        print(row)
    print(f"\nTotal mistakes: {len(guess_history) - CLUSTERS}")


def plot_results(guess_history: list[tuple[list[str], int]]) -> None:
    cluster_counts = [sum(1 for _, fb in guess_history if fb == i) for i in range(3)]
    labels = ["Misses (0)", "Close (1)", "Correct (2)"]
    colors = ["gold", "lightblue", "green"]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pie(
        cluster_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140
    )
    plt.title("Connections Guess Distribution")

    plt.subplot(1, 2, 2)
    attempts = list(range(1, len(guess_history) + 1))
    feedback_values = [fb for _, fb in guess_history]
    plt.bar(attempts, feedback_values, color="blue")
    plt.xlabel("Attempt Number")
    plt.ylabel("Feedback Type (0=Miss, 1=Close, 2=Correct)")
    plt.title("Guess Feedback Over Time")

    plt.tight_layout()
    plt.show()
