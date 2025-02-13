"""
Wordle Solver

This program simulates a Wordle solver using different strategies.
It displays a dashboard showing the progress of the simulation.
"""

import sys
import time
from functools import lru_cache
from contextlib import contextmanager
from collections import defaultdict
from typing import List, Tuple, Dict, Generator
import pandas as pd
import numpy as np

# Global Constants
INITIAL_GUESS: str = None
WLEN: int = 5
MISS, CLOSE, HIT = np.uint8(0), np.uint8(1), np.uint8(2)
COLOURS: Dict[int, str] = {
    MISS: "⬜", 
    CLOSE: "🟨", 
    HIT: "🟩"
    }
CANDIDATES_FILE: str = "wordle/answers-alphabetical.txt"
WORDS_FILE: str = "wordle/allowed-guesses.txt"
UPDATE_FREQ: int = 1
THRESH: int = 3000 # Switch to word list when candidates fall below this threshold
M, C = 3/20, 3/2


# =============================================================================
# Dashboard Class
# =============================================================================
class Dashboard:
    """Handles drawing and updating the simulation dashboard in the terminal."""

    def __init__(self) -> None:
        # Move the cursor to the top-left and clear the screen
        sys.stdout.write("\033[H\033[J")

    @staticmethod
    def draw_dashboard(version: str = None,
                       target: str = None,
                       guesses: List[str] = None,
                       feedbacks: List[Tuple[int, ...]] = None,
                       distribution: List[int] = None,
                       remaining: List[int] = None,
                       total: int = None,
                       start_time: float = None,
                       best_guess: str = None) -> None:
        """
        Draws the Wordle dashboard with the given simulation information.
        Only updates every UPDATE_FREQ games.
        """
        if distribution is not None and sum(distribution) % UPDATE_FREQ != 0:
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

        # Display feedback rows with colours
        if feedbacks is not None:
            rows = 0
            for fb in feedbacks:
                sys.stdout.write("".join(COLOURS[col] for col in fb) + "\n")
                rows += 1
            sys.stdout.write("\n" * (6 - rows))

        if best_guess is not None:
            sys.stdout.write(f"Best Guess: {best_guess}\n")

        # Display game distribution and average score
        if distribution is not None:
            sys.stdout.write(f"Distribution: {distribution}\n")
            sys.stdout.write(f"Average: {Dashboard.get_average(distribution):.2f}\n\n")
        if start_time is not None:
            Dashboard.progress_bar(sum(distribution), total, start_time=start_time)
        

    @staticmethod
    def clear_screen() -> None:
        """Clears the terminal screen and hides the cursor."""
        sys.stdout.write("\033[2J\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(current: int,
                     total: int,
                     bar_length: int = 25,
                     start_time: float = None) -> None:
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

        sys.stdout.write(f'\r[{bar}] {fraction*100:.1f}% | {time_remaining_str}')
        sys.stdout.flush()

    @staticmethod
    def get_average(distribution: List[int]) -> float:
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
    @lru_cache(maxsize=None)
    def score_guess(guess: str, target: str) -> Tuple[int, ...]:
        """
        Scores a guess against the target word.
        Returns a tuple where:
          - 0: letter not in target (grey)
          - 1: letter in target but in a different position (yellow)
          - 2: letter in the correct position (green)
        """
        target_chars = list(target)
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

    def make_guess(self, guess: str) -> Tuple[int, ...]:
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

    def __init__(self, all_candidates: set[str], all_words: set[str], version: str = "frequency") -> None:
        self.version: str = version.lower()
        self.strategy = {
            "frequency": self.select_guess_frequency,
            "entropy": self.select_guess_entropy,
            "hybrid": self.select_guess_hybrid
        }[self.version]
        # Working copies of candidate and allowed words.
        self.candidates: set[str] = all_candidates.copy()
        self.words: set[str] = all_words.copy()

    def __str__(self) -> str:
        return self.version

    def filter_candidates(self, guess: str, feedback: Tuple[int, ...]) -> None:
        """
        Filters candidate words based on feedback from a guess.
        """
        self.candidates = {word for word in self.candidates
                           if WordleGame.score_guess(guess, word) == feedback}

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
        freq = {letter: sum(word.count(letter) for word in self.candidates)
                for letter in letters}
        best_guess = max(self.words,
                         key=lambda word: sum(freq.get(letter, 0) for letter in set(word)))
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
        entropy = 0.0
        for count in feedback_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    def entropy_to_guesses(self, entropy: float) -> float:
        """Converts an entropy value to an estimated number of guesses using constants M and C."""
        return entropy * M + C

    def expected_guesses(self, guess: str) -> float:
        """
        Estimates the expected number of guesses remaining if the given guess is made.
        """

        p = self.probability_of_guess(guess)
        current_uncertainty = np.log2(1 / len(self.candidates))
        expected = (1 - p) * self.entropy_to_guesses(current_uncertainty - self.expected_information_gain(guess))
        return expected

    def probability_of_guess(self, guess: str) -> float:
        """Returns the probability of a guess being correct among the current candidates."""

        if guess not in self.candidates:
            return 0.0
        return 1 / len(self.candidates)
    
    def solve_wordle(self, game: WordleGame, max_guesses: int = 6, guess: str = None) -> Tuple[List[str], List[Tuple[int, ...]], List[int]]:
        """
        Solves the Wordle game by iteratively making guesses.
        Returns a tuple: (list of guesses, list of feedback tuples, list of candidate counts after each guess).
        """
        guesses: List[str] = []
        feedbacks: List[Tuple[int, ...]] = []
        remaining: List[int] = []

        # Use the provided initial guess or select one using the chosen strategy.
        if guess is None:
            guess = self.strategy()

        for _ in range(max_guesses):
            guesses.append(guess)
            feedback = game.make_guess(guess)
            feedbacks.append(feedback)

            if feedback == (HIT,) * WLEN:
                break

            self.filter_candidates(guess, feedback)
            remaining.append(len(self.candidates))
            if not self.candidates:
                print("No candidates left to guess!")
                break

            guess = self.strategy()

        return guesses, feedbacks, remaining


# =============================================================================
# SolutionTester Class
# =============================================================================
class SolutionTester:
    """Tests the WordleSolver against all target words and updates the dashboard."""

    def __init__(self, all_candidates: set[str], all_words: set[str], dashboard: Dashboard, version: str) -> None:
        self.dashboard = dashboard
        self.all_candidates = all_candidates  # Target words
        self.all_words = all_words            # Allowed guesses
        self.distribution: List[int] = [0] * 6
        self.version = version

    @contextmanager
    def hidden_cursor(self) -> Generator[None, None, None]:
        """Hides the terminal cursor during testing."""
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        try:
            yield
        finally:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    def test_solver(self, initial_guess: str = None) -> None:
        """Tests the solver on all target words."""
        self.start_time = time.perf_counter()
        with self.hidden_cursor():
            for target in self.all_candidates:
                self.test_target(target, initial_guess)
        elapsed = time.perf_counter() - self.start_time
        print(f"\n{len(self.all_candidates)} games played, {elapsed:.2f}s elapsed.")

    def test_target(self, target: str, initial_guess: str = None) -> None:
        """
        Tests the solver on a single target word.
        Updates the distribution of guesses and the dashboard.
        """
        solver = WordleSolver(self.all_candidates, self.all_words, version=self.version)
        game = WordleGame(target)
        guesses, feedbacks, remaining = solver.solve_wordle(game, guess=initial_guess)
        self.distribution[len(guesses) - 1] += 1

        # Ensure remaining list has at least one element for dashboard formatting.
        dashboard_remaining = remaining + [0]  
        self.dashboard.draw_dashboard(str(solver), target, guesses, feedbacks,
                                      self.distribution, dashboard_remaining,
                                      len(self.all_candidates), self.start_time)


# =============================================================================
# Simulation Functions
# =============================================================================
def simulate(all_candidates_file: str = CANDIDATES_FILE, all_words_file: str = WORDS_FILE, 
             version: str = "frequency", initial_guess: str = INITIAL_GUESS) -> None:
    """Loads words from files and runs the simulation."""
    if initial_guess is not None and len(initial_guess) != WLEN:
        initial_guess = None

    # Load and filter target words (one word per line with the correct length)
    with open(all_candidates_file, "r") as f:
        all_candidates = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    # Load and filter allowed guesses.
    with open(all_words_file, "r") as f:
        all_words = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    dashboard = Dashboard()
    tester = SolutionTester(all_candidates, all_words, dashboard, version=version)
    tester.test_solver(initial_guess)

def manual(all_candidates_file: str = CANDIDATES_FILE, all_words_file: str = WORDS_FILE, version: str = "frequency") -> None:
    """Loads words from files and runs solver on a single target word."""
     # Load and filter target words (one word per line with the correct length)
    with open(all_candidates_file, "r") as f:
        all_candidates = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    # Load and filter allowed guesses.
    with open(all_words_file, "r") as f:
        all_words = {line.strip().lower() for line in f if len(line.strip()) == WLEN}

    dashboard = Dashboard()
    solver = WordleSolver(all_candidates, all_words, version=version)
    
    guess, feedbacks, feedback = "", [], ()
    
    dashboard.draw_dashboard(version=str(solver), feedbacks=feedbacks, best_guess=solver.strategy())

    while feedback != (HIT,) * WLEN:
        guess, feedback = "", []
        while True:
            guess = input("Guess: ").lower()
            if len(guess) == WLEN and guess.isalpha():
                break
            print("Invalid input.")
            
        while True:
            try:
                feedback = tuple([int(c) for c in input("Feedback: ")])
            except:
                feedback = []
            if len(feedback) == WLEN and all([c in (COLOURS.keys()) for c in feedback]):
                break
            print("Invalid input")
            
        feedbacks.append(feedback)
        solver.filter_candidates(guess, feedback)
        dashboard.draw_dashboard(feedbacks=feedbacks, best_guess=solver.strategy())


def main() -> None:
    manual(version="hybrid")


if __name__ == "__main__":
    main()
