INITIAL_GUESS = "slate"
WLEN = 5
COLOURS = {0: "⬜", 1: "🟨", 2: "🟩"}
CANDIDATES_FILE = "answers-alphabetical.txt"
WORDS_FILE = "allowed-guesses.txt"
UPDATE_FREQ = 1
M, C = 3/20, 3/2

import sys
import time
from functools import lru_cache
from contextlib import contextmanager
from typing import Callable as function
from collections import defaultdict
import pandas as pd
import numpy as np

class Dashboard:
    def __init__(self):
        # Move the cursor to the top-left of the terminal
        sys.stdout.write("\033[H")
        # Clear the screen from cursor down
        sys.stdout.write("\033[J")

    @staticmethod
    def draw_dashboard(target: str, guesses: list, feedbacks: list, 
                       distribution: list, remaining: list, total: int, start_time: float):
        """
        Draws the Wordle dashboard with the given information.
        """
        # Only update the dashboard every UPDATE_FREQ games.
        if sum(distribution) % UPDATE_FREQ != 0:
            return

        # Move the cursor to the top-left of the terminal and clear screen.
        sys.stdout.write("\033[H")
        sys.stdout.write("\033[J")
        
        # Dashboard content.
        sys.stdout.write(f"Score: {len(feedbacks)}\n")
        sys.stdout.write(f"Target: {target}\n")
        sys.stdout.write(f"Guesses: {guesses}\n")
        sys.stdout.write(f"Remaining: {remaining}\n")

        # Display the feedback for each guess.
        rows = 0
        for feedback in feedbacks:
            sys.stdout.write("".join(COLOURS[colour] for colour in feedback) + "\n")
            rows += 1
        sys.stdout.write("\n" * (6 - rows))

        sys.stdout.write(f"Distribution: {distribution}\n")
        sys.stdout.write(f"Average: {Dashboard.get_average(distribution):.2f}\n\n")

        Dashboard.progress_bar(sum(distribution), total, start_time=start_time)

    @staticmethod
    def clear_screen():
        """
        Clears the terminal screen and hides the cursor.
        """
        sys.stdout.write("\033[2J")
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(current: int, total: int, bar_length: int = 25, start_time: float = None):
        """
        Displays a progress bar for the current progress out of the total.
        """
        # Total units available (each character can have 8 levels of fill)
        total_units = bar_length * 8
        # Calculate how many units should be filled based on progress.
        filled_units = int((current / total) * total_units)
        
        bar = ""
        for _ in range(bar_length):
            if filled_units >= 8:
                # Full block if 8 or more units are available for this slot.
                bar += "█"  # U+2588
                filled_units -= 8
            else:
                # Mapping: 0->" " (empty), 1->"▏", 2->"▎", 3->"▍", 4->"▌",
                #          5->"▋", 6->"▊", 7->"▉"
                partials = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
                bar += partials[filled_units]
                filled_units = 0
        fraction = current / total
        
        # Calculate estimated time remaining.
        if current > 0 and start_time is not None:
            elapsed_time = time.perf_counter() - start_time
            estimated_total_time = elapsed_time / fraction
            time_remaining = estimated_total_time - elapsed_time
            time_remaining_str = f"{time_remaining:.1f}s remaining"
        else:
            time_remaining_str = "Calculating..."

        # Build and write the progress bar line.
        sys.stdout.write(f'\r[{bar}] {fraction*100:.1f}% | {time_remaining_str}')
        sys.stdout.flush()

    @staticmethod
    def get_average(distribution: list) -> float:
        """
        Returns the average number of guesses needed to solve the game.
        """
        total_games = sum(distribution)
        if total_games == 0:
            return 0
        total_guesses = sum(count * (i + 1) for i, count in enumerate(distribution))
        return total_guesses / total_games

class WordleGame:
    def __init__(self, target: str):
        """
        Initializes the game with a target word.
        """
        self.target = target.lower()
    
    @staticmethod
    @lru_cache(maxsize=None)
    def score_guess(guess: str, target: str) -> tuple:
        """
        Scores a guess against a given target word.
        Returns a tuple where:
          - 0 means the letter is not in the target (grey),
          - 1 means the letter is in the target but in a different position (yellow),
          - 2 means the letter is in the correct position (green).
        """
        target_chars = list(target)
        score = [0] * WLEN
        
        # First pass: mark greens.
        for i in range(WLEN):
            if guess[i] == target[i]:
                score[i] = 2
                target_chars[i] = None  # mark as used
        
        # Second pass: mark yellows.
        for i in range(WLEN):
            if score[i] == 0 and guess[i] in target_chars:
                score[i] = 1
                target_chars[target_chars.index(guess[i])] = None  # mark as used
        
        return tuple(score)
    
    def make_guess(self, guess: str) -> tuple:
        """
        Scores a guess against this game's target word.
        """
        return WordleGame.score_guess(guess, self.target)

class WordleSolver:
    def __init__(self, all_candidates: list, all_words: list):
        """
        Initializes the solver by loading the list of target words and allowed guesses.
        """
        # Create copies of the word lists.
        self.candidates = all_candidates[:]  
        self.words = all_words[:]
    
    def filter_candidates(self, guess: str, feedback: tuple):
        """
        Filters the candidate words based on the feedback from a guess.
        """
        self.candidates = [word for word in self.candidates 
                           if WordleGame.score_guess(guess, word) == feedback]
    
    def select_guess_frequency(self) -> str:
        """
        Selects the best next guess from the candidate words.
        This version uses letter frequency over the candidates and only counts unique letters.
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        # Build frequency dictionary over the candidate list.
        freq = {letter: sum(word.count(letter) for word in self.candidates) for letter in letters}
        # Calculate score for each candidate: sum frequency for each unique letter.
        scores = [sum(freq.get(letter, 0) for letter in set(word)) for word in self.candidates]
        best_index = scores.index(max(scores))
        return self.candidates[best_index]
    
    def select_guess_entropy(self) -> str:
        """
        Selects the best next guess from the candidate words.
        """
        if len(self.candidates) <= 2:
            return best_guess
        max_entropy = -1
        best_guess = self.candidates[0]
        for guess in self.words:
            entropy = self.expected_information_gain(guess)
            if entropy > max_entropy:
                max_entropy = entropy
                best_guess = guess
        self.words.remove(best_guess)
        return best_guess
    
    def select_guess_hybrid(self) -> str:
        """
        Selects the best next guess from the candidate words.
        """
        min_guesses = float("inf")
        best_guess = self.candidates[0]
        if len(self.candidates) <= 1:
            return best_guess
        for guess in self.words:
            expected = self.expected_guesses(guess)
            if expected < min_guesses:
                min_guesses = expected
                best_guess = guess
        self.words.remove(best_guess)
        return best_guess
    
    def expected_information_gain(self, guess: str) -> float:
        """
        Calculates the expected information gain from making a guess.
        """
        # Count the frequency of each feedback for the candidate list.
        feedback_counts = defaultdict(int)
        for target in self.candidates:
            feedback = WordleGame.score_guess(guess, target)
            feedback_counts[feedback] += 1
        
        # Calculate the entropy of the feedback distribution.
        total = len(self.candidates)
        entropy = 0
        for count in feedback_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        return entropy
    
    def entropy_to_guesses(self, entropy: float) -> float:
        return entropy * M + C
    
    def expected_guesses(self, guess: str) -> float:
        """
        Calculates the expected number of guesses needed to solve the game.
        """
        uncertainty = sum([self.probability_of_guess(candidate) * np.log2(self.probability_of_guess(candidate)) for candidate in self.candidates])
        p = self.probability_of_guess(guess)
        expected = (1-p)*self.entropy_to_guesses(uncertainty - self.expected_information_gain(guess))
        return expected
        

    def probability_of_guess(self, guess: str) -> float:
        # return len([word for word in self.candidates if guess in word]) / len(self.candidates)
        return 1/len(self.candidates)


    def solve_wordle(self, game: WordleGame, strategy: function, max_guesses: int = 6, guess: str = None) -> tuple:
        """
        Solves the Wordle game by interacting with a WordleGame instance.
        Returns a tuple containing the list of guesses made, their feedback, and remaining candidate counts.
        """
        guesses = []
        feedbacks = []
        remaining = []
        
        # Make an initial guess from the candidate list.
        if guess is None:
            guess = strategy()
        
        for _ in range(max_guesses):
            guesses.append(guess)
            feedback = game.make_guess(guess)
            feedbacks.append(feedback)
            
            if feedback == (2,) * WLEN:
                break
            
            self.filter_candidates(guess, feedback)
            remaining.append(len(self.candidates))
            if not self.candidates:
                print("No candidates left to guess!")
                break
            
            guess = strategy()
        
        return guesses, feedbacks, remaining

class SolutionTester:
    def __init__(self, all_candidates: list, all_words: list, dashboard: Dashboard):
        """
        Initializes the tester with a solver and game instance.
        """
        self.dashboard = dashboard
        self.all_candidates = all_candidates  # List of target words
        self.all_words = all_words            # List of allowed guesses
        self.distribution = [0] * 6
        
    @contextmanager
    def hidden_cursor(self):
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        try:
            yield
        finally:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

    def test_solver(self, initial_guess: str = None):
        """
        Tests the solver on all target words.
        """
        self.start_time = time.perf_counter()
        with self.hidden_cursor():
            for target in self.all_candidates:
                self.test_target(target, initial_guess=initial_guess)
        
        elapsed = time.perf_counter() - self.start_time
        print(f"\n{len(self.all_candidates)} games played, {elapsed:.2f}s elapsed.")

    def test_target(self, target: str, initial_guess: str = None):
        """
        Tests the solver on a specific target word.
        """
        solver = WordleSolver(self.all_candidates, self.all_words)
        game = WordleGame(target)
        guesses, feedbacks, remaining = solver.solve_wordle(game, strategy = solver.select_guess_hybrid, guess=initial_guess)
        self.distribution[len(guesses) - 1] += 1
        # Pass the remaining candidate counts plus a trailing 0 to the dashboard.
        self.dashboard.draw_dashboard(target, guesses, feedbacks, self.distribution, remaining + [0],
                                      len(self.all_candidates), self.start_time)

def simulate(all_candidates_file: str = CANDIDATES_FILE, all_words_file: str = WORDS_FILE):

    initial_guess = INITIAL_GUESS if len(INITIAL_GUESS) == WLEN else None

    # Load and clean the target words from file.
    with open(all_candidates_file, "r") as f:        all_candidates = [line.strip().lower() for line in f if len(line.strip()) == WLEN]

    # Load and clean the allowed guesses from file.
    with open(all_words_file, "r") as f:
        all_words = [line.strip().lower() for line in f if len(line.strip()) == WLEN]

    dashboard = Dashboard()
    tester = SolutionTester(all_candidates, all_words, dashboard)
    tester.test_solver(initial_guess=initial_guess)

def main():
    simulate()

if __name__ == "__main__":
    main()
