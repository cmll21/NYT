"""
Wordle UI and plotting helpers.
"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

MAX_GUESSES: int = 6
UPDATE_FREQ: int = 1
COLORS: dict[int, str] = {0: "⬜", 1: "🟨", 2: "🟩"}


class Dashboard:
    def __init__(self) -> None:
        sys.stdout.write("\033[H\033[J")

    @staticmethod
    def draw_dashboard(
        version: str | None = None,
        target: str | None = None,
        guesses: list[str] | None = None,
        feedbacks: list[tuple[int, ...]] | None = None,
        distribution: list[int] | None = None,
        remaining: list[int] | None = None,
        total: int | None = None,
        start_time: float | None = None,
        best_guess: str | None = None,
        completed: int | None = None,
    ) -> None:
        games_completed = (
            completed if completed is not None else sum(distribution or [])
        )
        if games_completed % UPDATE_FREQ != 0:
            return

        sys.stdout.write("\033[H\033[J")

        if version is not None:
            sys.stdout.write(f"Version: {version}\n")
        if feedbacks is not None:
            sys.stdout.write(f"Score: {len(feedbacks)}\n")
        if target is not None:
            sys.stdout.write(f"Target: {target}\n")
        if guesses is not None:
            sys.stdout.write(f"Guesses: {guesses}\n")
        if remaining is not None:
            sys.stdout.write(f"Remaining: {remaining}\n")

        if feedbacks is not None:
            rows = 0
            for fb in feedbacks:
                sys.stdout.write("".join(COLORS[col] for col in fb) + "\n")
                rows += 1
            sys.stdout.write("\n" * (MAX_GUESSES - rows))

        if best_guess is not None:
            sys.stdout.write(f"Best Guess: {best_guess}\n")

        if distribution is not None:
            sys.stdout.write(f"Distribution: {distribution}\n")
            sys.stdout.write(f"Average: {Dashboard.get_average(distribution):.2f}\n\n")
        if start_time is not None and total is not None and distribution is not None:
            Dashboard.progress_bar(
                current=games_completed, total=total, start_time=start_time
            )

    @staticmethod
    def clear_screen() -> None:
        sys.stdout.write("\033[2J\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(
        current: int, total: int, bar_length: int = 25, start_time: float | None = None
    ) -> None:
        total_units = bar_length * 8
        filled_units = int((current / total) * total_units)
        bar = ""

        for _ in range(bar_length):
            if filled_units >= 8:
                bar += "█"
                filled_units -= 8
            else:
                partials = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
                bar += partials[filled_units]
                filled_units = 0

        fraction = current / total
        if current > 0 and start_time is not None:
            elapsed_time = time.perf_counter() - start_time
            estimated_total = elapsed_time / fraction
            time_remaining = estimated_total - elapsed_time
            time_remaining_str = f"{time_remaining:.1f}s remaining"
        else:
            time_remaining_str = "Calculating..."

        sys.stdout.write(f"\r[{bar}] {fraction * 100:.1f}% | {time_remaining_str}")
        sys.stdout.flush()

    @staticmethod
    def get_average(distribution: list[int]) -> float:
        total_games = sum(distribution)
        if total_games == 0:
            return 0.0
        total_guesses = sum(count * (i + 1) for i, count in enumerate(distribution))
        return total_guesses / total_games


def plot_results(remaining_counts_per_game: list[list[int]], distribution: list[int]):
    max_guesses = max(len(game_counts) for game_counts in remaining_counts_per_game)

    avg_remaining_candidates = []
    for guess_num in range(max_guesses):
        guess_counts = [
            game_counts[guess_num]
            for game_counts in remaining_counts_per_game
            if len(game_counts) > guess_num
        ]
        avg_remaining_candidates.append(np.mean(guess_counts) if guess_counts else 0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].bar(
        range(1, MAX_GUESSES + 1), distribution, color="skyblue", edgecolor="black"
    )
    axs[0].set_title("Wordle Guess Distribution")
    axs[0].set_xlabel("Number of Guesses")
    axs[0].set_ylabel("Frequency")

    axs[1].plot(
        range(1, len(avg_remaining_candidates) + 1),
        avg_remaining_candidates,
        marker="o",
        linestyle="-",
        color="red",
    )
    axs[1].set_title("Average Remaining Candidates per Guess")
    axs[1].set_xlabel("Guess Number")
    axs[1].set_ylabel("Average Candidates Remaining")
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.show()
