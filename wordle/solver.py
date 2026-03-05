"""
Wordle core solver logic.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from functools import cache

import numpy as np

from .data import load_words
from .views import Dashboard, plot_results

INITIAL_GUESS: str | None = None
WLEN: int = 5
MAX_GUESSES: int = 6
MISS, CLOSE, HIT = 0, 1, 2
COLORS: dict[int, str] = {MISS: "⬜", CLOSE: "🟨", HIT: "🟩"}
CANDIDATES_FILE: str = "wordle/answers-alphabetical.txt"
WORDS_FILE: str = "wordle/allowed-guesses.txt"
UPDATE_FREQ: int = 1
THRESH: int = 3000
M, C = 3 / 20, 3 / 2


class WordleGame:
    def __init__(self, target: str) -> None:
        self.target: str = target.lower()

    @staticmethod
    @cache
    def score_guess(guess: str, target: str) -> tuple[int, ...]:
        target_chars: list[str | None] = list(target)
        score = [MISS] * WLEN

        for i in range(WLEN):
            if guess[i] == target[i]:
                score[i] = HIT
                target_chars[i] = None

        for i in range(WLEN):
            if score[i] == 0 and guess[i] in target_chars:
                score[i] = CLOSE
                target_chars[target_chars.index(guess[i])] = None

        return tuple(score)

    def make_guess(self, guess: str) -> tuple[int, ...]:
        return WordleGame.score_guess(guess, self.target)


class WordleSolver:
    def __init__(
        self, all_candidates: set[str], all_words: set[str], version: str = "frequency"
    ) -> None:
        self.version: str = version.lower()
        self.strategy = {
            "frequency": self.select_guess_frequency,
            "entropy": self.select_guess_entropy,
            "hybrid": self.select_guess_hybrid,
        }[self.version]
        self.candidates: set[str] = all_candidates.copy()
        self.words: set[str] = all_words.copy()

    def __str__(self) -> str:
        return self.version

    def filter_candidates(self, guess: str, feedback: tuple[int, ...]) -> None:
        self.candidates = {
            word
            for word in self.candidates
            if WordleGame.score_guess(guess, word) == feedback
        }

    def switch_list(self) -> None:
        if len(self.candidates) <= THRESH and self.words != self.candidates:
            self.words = self.candidates

    def select_guess_frequency(self) -> str:
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
        try:
            suggestion = self.strategy()
        except (ValueError, StopIteration):
            return ""

        self.words.add(suggestion)
        return suggestion

    def expected_information_gain(self, guess: str) -> float:
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
        return entropy * M + C

    def expected_guesses(self, guess: str) -> float:
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
        if guess not in self.candidates:
            return 0.0
        return 1 / len(self.candidates)

    def solve_wordle(
        self, game: WordleGame, max_guesses: int = MAX_GUESSES, guess: str | None = None
    ) -> tuple[list[str], list[tuple[int, ...]], list[int], bool]:
        guesses: list[str] = []
        feedbacks: list[tuple[int, ...]] = []
        remaining: list[int] = []

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


class SolutionTester:
    def __init__(
        self,
        all_candidates: set[str],
        all_words: set[str],
        dashboard,
        version: str,
        visualize: bool = False,
    ):
        self.dashboard = dashboard
        self.all_candidates = all_candidates
        self.all_words = all_words
        self.distribution: list[int] = [0] * MAX_GUESSES
        self.failures: int = 0
        self.version = version
        self.visualize = visualize
        self.remaining_counts_per_game = []

    @contextmanager
    def hidden_cursor(self):
        print("\033[?25l", end="")
        try:
            yield
        finally:
            print("\033[?25h", end="")

    def test_solver(self, initial_guess: str | None = None):
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


def simulate(
    all_candidates_file: str = CANDIDATES_FILE,
    all_words_file: str = WORDS_FILE,
    version: str = "frequency",
    initial_guess: str | None = INITIAL_GUESS,
    visualize: bool = False,
) -> None:
    if initial_guess is not None and len(initial_guess) != WLEN:
        initial_guess = None

    all_candidates: set[str] = load_words(all_candidates_file)
    all_words = load_words(all_words_file)

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
    all_candidates = load_words(all_candidates_file)
    all_words = load_words(all_words_file)

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
                feedback = tuple([int(c) for c in input("Feedback: ")])
            except ValueError:
                feedback = ()
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


def main() -> None:
    manual(version="hybrid")


if __name__ == "__main__":
    main()
