"""
Wordle Solver

This program simulates a Wordle solver using different strategies.
It displays a dashboard showing the progress of the simulation.
"""

import sys
import time
from functools import cache
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Global Constants
INITIAL_GUESS: str | None = None
WLEN: int = 5
MAX_GUESSES: int = 6
MISS, CLOSE, HIT = 0, 1, 2
COLORS: dict[int, str] = {MISS: "⬜", CLOSE: "🟨", HIT: "🟩"}
CANDIDATES_FILE: str = "wordle/answers-alphabetical.txt"
WORDS_FILE: str = "wordle/allowed-guesses.txt"
UPDATE_FREQ: int = 1
THRESH: int = 3000  # Switch to word list when candidates fall below this threshold
M, C = 3 / 20, 3 / 2


# =============================================================================
# Dashboard Class
# =============================================================================
class Dashboard:
    """Handles drawing and updating the simulation dashboard in the terminal."""

    def __init__(self) -> None:
        # Move the cursor to the top-left and clear the screen
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
        """
        Draws the Wordle dashboard with the given simulation information.
        Only updates every UPDATE_FREQ games.
        """
        games_completed = (
            completed if completed is not None else sum(distribution or [])
        )
        if games_completed % UPDATE_FREQ != 0:
            return

        # Clear screen and reposition cursor
        sys.stdout.write("\033[H\033[J")

        # Write header and game info
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

        # Display feedback rows with colors
        if feedbacks is not None:
            rows = 0
            for fb in feedbacks:
                sys.stdout.write("".join(COLORS[col] for col in fb) + "\n")
                rows += 1
            sys.stdout.write("\n" * (MAX_GUESSES - rows))

        if best_guess is not None:
            sys.stdout.write(f"Best Guess: {best_guess}\n")

        # Display game distribution and average score
        if distribution is not None:
            sys.stdout.write(f"Distribution: {distribution}\n")
            sys.stdout.write(f"Average: {Dashboard.get_average(distribution):.2f}\n\n")
        if start_time is not None and total is not None and distribution is not None:
            Dashboard.progress_bar(
                current=games_completed, total=total, start_time=start_time
            )

    @staticmethod
    def clear_screen() -> None:
        """Clears the terminal screen and hides the cursor."""
        sys.stdout.write("\033[2J\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(
        current: int, total: int, bar_length: int = 25, start_time: float | None = None
    ) -> None:
        """
        Displays a progress bar representing the simulation progress.
        """
        total_units = bar_length * 8
        filled_units = int((current / total) * total_units)
        bar = ""

        for _ in range(bar_length):
            if filled_units >= 8:
                bar += "█"  # Full block
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
        """Calculates the average number of guesses used to solve the game."""
        total_games = sum(distribution)
        if total_games == 0:
            return 0.0
        total_guesses = sum(count * (i + 1) for i, count in enumerate(distribution))
        return total_guesses / total_games


# =============================================================================
# WordleGame Class
# =============================================================================
class WordleGame:
    """Represents a single Wordle game with a target word."""

    def __init__(self, target: str) -> None:
        self.target: str = target.lower()

    @staticmethod
    @cache
    def score_guess(guess: str, target: str) -> tuple[int, ...]:
        """
        Scores a guess against the target word.
        Returns a tuple where:
          - 0: letter not in target (gray)
          - 1: letter in target but in a different position (yellow)
          - 2: letter in the correct position (green)
        """
        target_chars: list[str | None] = list(target)
        score = [MISS] * WLEN

        # First pass: mark greens and mark letters as used.
        for i in range(WLEN):
            if guess[i] == target[i]:
                score[i] = HIT
                target_chars[i] = None

        # Second pass: mark yellows.
        for i in range(WLEN):
            if score[i] == 0 and guess[i] in target_chars:
                score[i] = CLOSE
                target_chars[target_chars.index(guess[i])] = None

        return tuple(score)

    def make_guess(self, guess: str) -> tuple[int, ...]:
        """Scores a guess against this game's target word."""
        return WordleGame.score_guess(guess, self.target)


# =============================================================================
# WordleSolver Class
# =============================================================================
class WordleSolver:
    """
    Solves a Wordle game using one of several strategies.
    Strategies available: 'frequency', 'entropy', and 'hybrid'.
    """

    def __init__(
        self, all_candidates: set[str], all_words: set[str], version: str = "frequency"
    ) -> None:
        self.version: str = version.lower()
        self.strategy = {
            "frequency": self.select_guess_frequency,
            "entropy": self.select_guess_entropy,
            "hybrid": self.select_guess_hybrid,
        }[self.version]
        # Working copies of candidate and allowed words.
        self.candidates: set[str] = all_candidates.copy()
        self.words: set[str] = all_words.copy()

    def __str__(self) -> str:
        return self.version

    def filter_candidates(self, guess: str, feedback: tuple[int, ...]) -> None:
        """
        Filters candidate words based on feedback from a guess.
        """
        self.candidates = {
            word
            for word in self.candidates
            if WordleGame.score_guess(guess, word) == feedback
        }

    def switch_list(self) -> None:
        if len(self.candidates) <= THRESH and self.words != self.candidates:
            self.words = self.candidates

    def select_guess_frequency(self) -> str:
        """
        Chooses a guess based on letter frequency across candidate words.
        Each candidate is scored by summing frequencies for its unique letters.
        """
        self.switch_list()
        letters = "abcdefghijklmnopqrstuvwxyz"
        freq: dict[str, int] = {
            letter: sum(word.count(letter) for word in self.candidates)
            for letter in letters
        }
        best_guess = max(
            self.words,
            key=lambda word: sum(freq.get(letter, 0) for letter in set(word)),
        )
        self.words.remove(best_guess)
        return best_guess

    def select_guess_entropy(self) -> str:
        """
        Chooses a guess using an entropy-based strategy.
        Returns a candidate when there are only a couple of possibilities left.
        """
        self.switch_list()
        max_entropy = -1.0
        best_guess = next(iter(self.candidates))
        for guess in self.words:
            entropy = self.expected_information_gain(guess)
            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = guess
        self.words.remove(best_guess)
        return best_guess

    def select_guess_hybrid(self) -> str:
        """
        Chooses a guess using a hybrid strategy that minimizes expected guesses.
        """
        self.switch_list()
        min_expected = float("inf")
        best_guess = next(iter(self.candidates))
        for guess in self.words:
            expected = self.expected_guesses(guess)
            if expected < min_expected:
                min_expected = expected
                best_guess = guess
        self.words.remove(best_guess)
        return best_guess

    def preview_next_guess(self) -> str:
        """
        Returns the next recommended guess without permanently removing it
        from the available guess pool.
        """
        try:
            suggestion = self.strategy()
        except (ValueError, StopIteration):
            return ""

        self.words.add(suggestion)
        return suggestion

    def expected_information_gain(self, guess: str) -> float:
        """
        Calculates the expected information gain from a guess.
        Uses the entropy of the feedback distribution over the candidate list.
        """
        feedback_counts = defaultdict(int)
        for target in self.candidates:
            fb = WordleGame.score_guess(guess, target)
            feedback_counts[fb] += 1

        total = len(self.candidates)
        return -sum(
            (count / total) * np.log2(count / total)
            for count in feedback_counts.values()
        )

    def entropy_to_guesses(self, entropy: float) -> float:
        """Converts an entropy value to an estimated number of guesses using constants M and C."""
        return entropy * M + C

    def expected_guesses(self, guess: str) -> float:
        """
        Estimates the expected number of guesses remaining if the given guess is made.
        """

        total_candidates = len(self.candidates)
        if total_candidates <= 1:
            return 0.0

        p = self.probability_of_guess(guess)
        current_uncertainty = np.log2(total_candidates)
        remaining_entropy = max(
            current_uncertainty - self.expected_information_gain(guess), 0.0
        )
        expected = (1 - p) * self.entropy_to_guesses(remaining_entropy)
        return expected

    def probability_of_guess(self, guess: str) -> float:
        """Returns the probability of a guess being correct among the current candidates."""

        if guess not in self.candidates:
            return 0.0
        return 1 / len(self.candidates)

    def solve_wordle(
        self, game: WordleGame, max_guesses: int = MAX_GUESSES, guess: str | None = None
    ) -> tuple[list[str], list[tuple[int, ...]], list[int], bool]:
        """
        Solves the Wordle game by iteratively making guesses.
        Returns a tuple: (list of guesses, list of feedback tuples, list of candidate counts after each guess).
        """
        guesses: list[str] = []
        feedbacks: list[tuple[int, ...]] = []
        remaining: list[int] = []

        # Use the provided initial guess or select one using the chosen strategy.
        if guess is None:
            guess = self.strategy()
        else:
            self.words.discard(guess)

        solved = False
        for _ in range(max_guesses):
            guesses.append(guess)
            feedback = game.make_guess(guess)
            feedbacks.append(feedback)

            if feedback == (HIT,) * WLEN:
                solved = True
                break

            self.filter_candidates(guess, feedback)
            remaining.append(len(self.candidates))
            if not self.candidates:
                print("No candidates left to guess!")
                break

            guess = self.strategy()

        return guesses, feedbacks, remaining, solved


# =============================================================================
# SolutionTester Class
# =============================================================================
class SolutionTester:
    """Tests the WordleSolver against all target words and tracks statistics."""

    def __init__(
        self,
        all_candidates: set[str],
        all_words: set[str],
        dashboard,
        version: str,
        visualize: bool = False,
    ):
        self.dashboard = dashboard
        self.all_candidates = all_candidates  # Target words
        self.all_words = all_words  # Allowed guesses
        self.distribution: list[int] = [0] * MAX_GUESSES
        self.failures: int = 0
        self.version = version
        self.visualize = visualize

        # Track remaining candidate counts per guess
        self.remaining_counts_per_game = []

    @contextmanager
    def hidden_cursor(self):
        """Hides the terminal cursor during testing."""
        print("\033[?25l", end="")  # Hide cursor
        try:
            yield
        finally:
            print("\033[?25h", end="")  # Show cursor

    def test_solver(self, initial_guess: str | None = None):
        """Tests the solver on all target words and optionally visualizes results."""
        self.start_time = time.perf_counter()
        with self.hidden_cursor():
            for target in self.all_candidates:
                self.test_target(target, initial_guess)

        elapsed = time.perf_counter() - self.start_time
        print(
            f"\n{len(self.all_candidates)} games played, "
            f"{elapsed:.2f}s elapsed, failures: {self.failures}."
        )

        if self.visualize:
            plot_results(self.remaining_counts_per_game, self.distribution)

    def test_target(self, target: str, initial_guess: str | None = None):
        """
        Tests the solver on a single target word.
        Updates the distribution of guesses and the dashboard.
        """
        solver = WordleSolver(self.all_candidates, self.all_words, version=self.version)
        game = WordleGame(target)
        guesses, feedbacks, remaining_candidates, solved = solver.solve_wordle(
            game, guess=initial_guess
        )

        if solved:
            self.distribution[len(guesses) - 1] += 1
        else:
            self.failures += 1
        self.remaining_counts_per_game.append(remaining_candidates)

        # Ensure remaining list has at least one element for dashboard formatting.
        dashboard_remaining = remaining_candidates + [0]
        self.dashboard.draw_dashboard(
            str(solver),
            target,
            guesses,
            feedbacks,
            self.distribution,
            dashboard_remaining,
            len(self.all_candidates),
            self.start_time,
            completed=sum(self.distribution) + self.failures,
        )


# =============================================================================
# Output Functions
# =============================================================================
def plot_results(remaining_counts_per_game: list[list[int]], distribution: list[int]):
    """Plot Wordle solver performance metrics using Matplotlib."""

    # Determine the maximum number of guesses taken in any game
    max_guesses = max(len(game_counts) for game_counts in remaining_counts_per_game)

    # Compute the average remaining candidates at each guess
    avg_remaining_candidates = []
    for guess_num in range(max_guesses):
        guess_counts = [
            game_counts[guess_num]
            for game_counts in remaining_counts_per_game
            if len(game_counts) > guess_num
        ]
        avg_remaining_candidates.append(np.mean(guess_counts) if guess_counts else 0)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram: Distribution of guesses needed
    axs[0].bar(
        range(1, MAX_GUESSES + 1), distribution, color="skyblue", edgecolor="black"
    )
    axs[0].set_title("Wordle Guess Distribution")
    axs[0].set_xlabel("Number of Guesses")
    axs[0].set_ylabel("Frequency")

    # Line chart: Average remaining candidates per guess
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
    axs[1].set_yscale("log")  # Log scale for better visualization

    plt.tight_layout()
    plt.show()


# =============================================================================
# Simulation Functions
# =============================================================================
def simulate(
    all_candidates_file: str = CANDIDATES_FILE,
    all_words_file: str = WORDS_FILE,
    version: str = "frequency",
    initial_guess: str | None = INITIAL_GUESS,
    visualize: bool = False,
) -> None:
    """Loads words from files and runs the simulation."""
    if initial_guess is not None and len(initial_guess) != WLEN:
        initial_guess = None

    # Load and filter target words (one word per line with the correct length)
    with open(all_candidates_file, "r") as f:
        all_candidates: set[str] = {
            line.strip().lower() for line in f if len(line.strip()) == WLEN
        }

    # Load and filter allowed guesses.
    with open(all_words_file, "r") as f:
        all_words = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    dashboard = Dashboard()
    tester = SolutionTester(
        all_candidates, all_words, dashboard, version=version, visualize=visualize
    )
    tester.test_solver(initial_guess)


def manual(
    all_candidates_file: str = CANDIDATES_FILE,
    all_words_file: str = WORDS_FILE,
    version: str = "frequency",
) -> None:
    """Loads words from files and runs solver on a single target word."""
    # Load and filter target words (one word per line with the correct length)
    with open(all_candidates_file, "r") as f:
        all_candidates = {
            line.strip().lower() for line in f if len(line.strip()) == WLEN
        }

    # Load and filter allowed guesses.
    with open(all_words_file, "r") as f:
        all_words = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    dashboard = Dashboard()
    solver = WordleSolver(all_candidates, all_words, version=version)

    guess: str = ""
    feedbacks: list[tuple[int, ...]] = []
    feedback: tuple[int, ...] = ()

    dashboard.draw_dashboard(
        version=str(solver), feedbacks=feedbacks, best_guess=solver.preview_next_guess()
    )

    while feedback != (HIT,) * WLEN:
        guess = ""
        feedback = ()
        while True:
            guess = input("Guess: ").lower()
            if len(guess) == WLEN and guess.isalpha():
                break
            print("Invalid input.")

        while True:
            try:
                feedback: tuple[int, ...] = tuple([int(c) for c in input("Feedback: ")])
            except ValueError:
                feedback: tuple[int, ...] = ()
            if len(feedback) == WLEN and all([c in (COLORS.keys()) for c in feedback]):
                break
            print("Invalid input")

        if feedback == (HIT,) * WLEN:
            break

        feedbacks.append(feedback)
        solver.words.discard(guess)
        solver.filter_candidates(guess, feedback)
        dashboard.draw_dashboard(
            feedbacks=feedbacks, best_guess=solver.preview_next_guess()
        )


# =============================================================================
# Main Execution
# =============================================================================


def main() -> None:
    manual(version="hybrid")


if __name__ == "__main__":
    main()
