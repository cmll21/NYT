WLEN = 5

import numpy as np
import pandas as pd

class WordleSolver:
    def __init__(self, target_list_file: str = "answers-alphabetical.txt", guesses_list_file: str = "allowed-guesses.txt"):
        """
        Initialises WordleSolver by importing and filtering the word list
        """

        # Input words into Panda series and make all lowercase
        words = pd.read_csv(target_list_file, header = None, names = ["word"])
        words = words["word"].str.strip().str.lower()

        all_words = pd.read_csv(guesses_list_file, header = None, names = ["word"])
        all_words = all_words["word"].str.strip().str.lower()

        # Ensure five-letter words are being used
        self.target_words = words[words.str.len() == WLEN].reset_index(drop=True)
        self.all_words = all_words[all_words.str.len() == WLEN].reset_index(drop=True)

    @staticmethod
    def score_guess(guess: str, target: str) -> tuple:
        """
        Scores guesses (0: Grey, 1; Yellow, 2: Green) by comparing word to target
        """

        # Initialises lists to track score and scored characters
        target_chars = list(target)
        score = [0] * WLEN
        
        # Check green letters
        for i in range(WLEN):
            if guess[i] == target[i]:
                score[i] = 2
                # Mark character as used
                target_chars[i] = None
        
        # Check yellow letters
        for i in range(WLEN):
            if score[i] == 0 and guess[i] in target_chars:
                score[i] = 1
                # Mark character as used
                target_chars[target_chars.index(guess[i])] = None
        
        return tuple(score)
    
    def filter_candidates(self, candidates: pd.Series, guess: str, feedback: tuple) -> pd.Series:
        """
        Filters down possible answers based off feedback
        """
    
        # Filter candidates by score
        candidates = candidates[candidates.apply(lambda x: self.score_guess(guess, x) == feedback)]
        
        return candidates
    
    def select_best_guess(self, candidates: pd.Series) -> str:
        """
        Selects the best guess from the remaining candidates
        """
        letters = "abcdefghijklmnopqrstuvwxyz"

        freq = {letter: int(candidates.str.count(letter).sum()) for letter in letters}

        def word_score(word):
            return sum([freq.get(letter, 0) for letter in word])
        
        best_guess = candidates[candidates.apply(word_score).idxmax()]
        return best_guess
    
    def solve_wordle(self, target: str, max_guesses: int = 6) -> tuple:
        """
        Solves a wordle puzzle
        """
        
        # Initialises candidates to all words
        candidates = self.target_words.copy()

        # Initialises guesses and feedback
        guesses = []
        feedbacks = []



        # Loop through guesses
        for _ in range(max_guesses):
            # Select best guess
            guess = self.select_best_guess(candidates)
            # Append guess
            guesses.append(guess)

            # Score guess
            feedback = self.score_guess(guess, target)
            feedbacks.append(feedback)

            # Filter candidates
            candidates = self.filter_candidates(candidates, guess, feedback)

            # Check if guess is correct
            if guess == target:
                break
        
        return guesses, feedbacks
    
# Main function
if __name__ == "__main__":
    ws = WordleSolver()

    print(ws.solve_wordle("jaunt"))