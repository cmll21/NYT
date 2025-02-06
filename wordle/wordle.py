WLEN = 5
COLOURS = {0: "⬜", 1: "🟨", 2: "🟩"}

import numpy as np
import pandas as pd
import sys
import time

class Dashboard:
    def __init__(self):
        # Move the cursor to the top-left of the terminal
        sys.stdout.write("\033[H")
        # Clear the screen from cursor down
        sys.stdout.write("\033[J")

    @staticmethod
    def draw_dashboard(target: str, guesses: list, feedbacks: list, distribution: list, total: int, start_time: float):
        """
        Draws the Wordle dashboard with the given information.
        """
        # Move the cursor to the top-left of the terminal
        sys.stdout.write("\033[H")
        # Clear the screen from cursor down
        sys.stdout.write("\033[J")
        
        # Dashboard content
        sys.stdout.write(f"Score: {len(feedbacks)}\n")
        sys.stdout.write(f"Target: {target}\n")
        sys.stdout.write(f"Guesses: {guesses}\n")

        # Display the feedback for each guess
        rows = 0
        for feedback in feedbacks:
            sys.stdout.write("".join(COLOURS[colour] for colour in feedback) + "\n")
            rows += 1
        sys.stdout.write("\n" * (6 - rows))

        sys.stdout.write(f"Distribution: {distribution}\n")
        sys.stdout.write(f"Average: {Dashboard.get_average(distribution):.2f}\n\n")

        Dashboard.progress_bar(sum(distribution), total,start_time= start_time)

    @staticmethod
    def clear_screen():
        """
        Clears the terminal screen and hides the cursor.
        """
        sys.stdout.write("\033[2J")
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(current: int, total: int, bar_length: int = 50, start_time: float = None):
        """
        Displays a progress bar for the current progress out of the total.
        """
        fraction = current / total
        arrow_count = int(fraction * bar_length)
        space_count = bar_length - arrow_count
        bar = '[' + '#' * arrow_count + ' ' * space_count + ']'
        percent = int(fraction * 100)
        
        # Calculate estimated time remaining
        if current > 0 and start_time is not None:
            elapsed_time = time.perf_counter() - start_time
            # Avoid division by zero; fraction > 0 since current > 0
            estimated_total_time = elapsed_time / fraction
            time_remaining = estimated_total_time - elapsed_time
            time_remaining_str = f"{time_remaining:.1f}s remaining"
        else:
            time_remaining_str = "Calculating..."

        # Build and write the progress bar line
        sys.stdout.write(f'\r{bar} {percent}% | {time_remaining_str}')
        sys.stdout.flush()

    @staticmethod
    def get_average(distribution: list) -> float:
        """
        Returns the average number of guesses needed to solve the game.
        """
        total_guesses = sum(count * (i + 1) for i, count in enumerate(distribution))
        total_games = sum(distribution)
        return total_guesses / total_games

class WordleGame:
    def __init__(self, target: str):
        """
        Initializes the game with a target word.
        """
        self.target = target.lower()
    
    @staticmethod
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
    def __init__(self, all_candidates: pd.Series, all_words: pd.Series):
        """
        Initializes the solver by loading the list of target words and allowed guesses.
        """
        self.candidates = all_candidates
        self.words = all_words
    
    def filter_candidates(self, guess: str, feedback: tuple):
        """
        Filters the candidate words based on the feedback from a guess.
        """
        self.candidates = self.candidates[self.candidates.apply(lambda x: WordleGame.score_guess(guess, x) == feedback)]
        
    
    def select_best_guess(self) -> str:
        """
        Selects the best next guess from the candidate words.
        This version uses letter frequency over the candidates and only counts unique letters.
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        freq = {letter: int(self.candidates.str.count(letter).sum()) for letter in letters}

        def word_score(word):
            # Only count each letter once.
            return sum(freq.get(letter, 0) for letter in set(word))
        
        best_guess = self.candidates[self.candidates.apply(word_score).idxmax()]
        return best_guess
    
    def solve_wordle(self, game: WordleGame, max_guesses: int = 6) -> tuple:
        """
        Solves the Wordle game by interacting with a WordleGame instance.
        Returns a tuple containing the list of guesses made and their feedback.
        """
        # Start with all possible target words.
        guesses = []
        feedbacks = []
        
        # Make an initial guess from the candidate list.
        guess = self.select_best_guess()
        
        for _ in range(max_guesses):
            guesses.append(guess)
            # Use the game to get feedback.
            feedback = game.make_guess(guess)
            feedbacks.append(feedback)
            
            # Check if the guess is correct.
            if feedback == (2,) * WLEN:
                break
            
            # Filter the candidates based on the feedback.
            self.filter_candidates(guess, feedback)
            if self.candidates.empty:
                print("No candidates left to guess!")
                break
            
            # Select the next guess from the filtered candidates.
            guess = self.select_best_guess()
        
        return guesses, feedbacks

class SolutionTester:
    def __init__(self, all_candidates: pd.Series, all_words: pd.Series, dashboard: Dashboard):
        """
        Initializes the tester with a solver and game instance.
        """
        self.dashboard = dashboard
        self.all_candidates = all_candidates
        self.all_words = all_words
        self.distribution = [0] * 6
        

    def test_solver(self):
        """
        Tests the solver on all target words.
        """
        self.start_time = time.perf_counter()
        for target in self.all_candidates:
            self.test_target(target)
        
        print("\n\nAll games completed!")

    def test_target(self, target: str):
        """
        Tests the solver on a specific target word.
        """
        solver = WordleSolver(self.all_candidates, self.all_words)
        game = WordleGame(target)
        guesses, feedbacks = solver.solve_wordle(game)
        self.distribution[len(guesses) - 1] += 1
        self.dashboard.draw_dashboard(target, guesses, feedbacks, self.distribution, len(self.all_candidates), self.start_time)



def main(all_candidates_file: str = "answers-alphabetical.txt", all_words_file: str = "allowed-guesses.txt"):

    # Load and clean the target words.
    all_candidates = pd.read_csv(all_candidates_file, header=None, names=["word"])
    all_candidates = all_candidates["word"].str.strip().str.lower()
    all_candidates = all_candidates[all_candidates.str.len() == WLEN].reset_index(drop=True)
    
    # Load and clean the allowed guesses.
    all_words = pd.read_csv(all_words_file, header=None, names=["word"])
    all_words = all_words["word"].str.strip().str.lower()
    all_words = all_words[all_words.str.len() == WLEN].reset_index(drop=True)

    dashboard = Dashboard
    tester = SolutionTester(all_candidates, all_words, dashboard)
    tester.test_solver()

# Main function to run the game and solver.
if __name__ == "__main__":
    
    main()

