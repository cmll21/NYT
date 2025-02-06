WLEN = 5

import numpy as np
import pandas as pd
import curses

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
    def __init__(self, target_list_file: str = "answers-alphabetical.txt",
                 guesses_list_file: str = "allowed-guesses.txt"):
        """
        Initializes the solver by loading the list of target words and allowed guesses.
        """
        # Load and clean the target words.
        words = pd.read_csv(target_list_file, header=None, names=["word"])
        words = words["word"].str.strip().str.lower()
        
        # Load and clean the allowed guesses.
        all_words = pd.read_csv(guesses_list_file, header=None, names=["word"])
        all_words = all_words["word"].str.strip().str.lower()
        
        # Filter to ensure words of the correct length.
        self.target_words = words[words.str.len() == WLEN].reset_index(drop=True)
        self.all_words = all_words[all_words.str.len() == WLEN].reset_index(drop=True)
    
    def filter_candidates(self, candidates: pd.Series, guess: str, feedback: tuple) -> pd.Series:
        """
        Filters the candidate words based on the feedback from a guess.
        """
        filtered = candidates[candidates.apply(lambda x: WordleGame.score_guess(guess, x) == feedback)]
        return filtered
    
    def select_best_guess(self, candidates: pd.Series) -> str:
        """
        Selects the best next guess from the candidate words.
        This version uses letter frequency over the candidates and only counts unique letters.
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        freq = {letter: int(candidates.str.count(letter).sum()) for letter in letters}

        def word_score(word):
            # Only count each letter once.
            return sum(freq.get(letter, 0) for letter in set(word))
        
        best_guess = candidates[candidates.apply(word_score).idxmax()]
        return best_guess
    
    def solve_wordle(self, game: WordleGame, max_guesses: int = 6) -> tuple:
        """
        Solves the Wordle game by interacting with a WordleGame instance.
        Returns a tuple containing the list of guesses made and their feedback.
        """
        # Start with all possible target words.
        candidates = self.target_words.copy()
        guesses = []
        feedbacks = []
        
        # Make an initial guess from the candidate list.
        guess = self.select_best_guess(candidates)
        
        for _ in range(max_guesses):
            guesses.append(guess)
            # Use the game to get feedback.
            feedback = game.make_guess(guess)
            feedbacks.append(feedback)
            
            # Check if the guess is correct.
            if feedback == (2,) * WLEN:
                break
            
            # Filter the candidates based on the feedback.
            candidates = self.filter_candidates(candidates, guess, feedback)
            if candidates.empty:
                print("No candidates left to guess!")
                break
            
            # Select the next guess from the filtered candidates.
            guess = self.select_best_guess(candidates)
        
        return guesses, feedbacks


def main(stdscr):
    stdscr.clear()


# Main function to run the game and solver.
if __name__ == "__main__":
    # Define the target word for the game.
    target = "crane"  # You can change this as needed.
    
    # Create a game instance with the target.
    game = WordleGame(target)
    
    # Create a solver instance.
    solver = WordleSolver()
    
    # Solve the game and print the result.
    guesses, feedbacks = solver.solve_wordle(game)
    print("Guesses:", guesses)
    print("Feedbacks:", feedbacks)

    curses.wrapper(main)
