"""
Connections Solver

Uses SpaCy and KMeans clustering to solve the Connections game one cluster at a time.
At the end, it prints a summary of all guesses using colored squares:
Level 0 = Yellow (🟨)
Level 1 = Green  (🟩)
Level 2 = Blue   (🟦)
Level 3 = Purple (🟪)
"""

import spacy
import json
import random
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Set, Dict, Tuple, Any, Deque
from functools import lru_cache
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# Configuration & Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)  # Disable logging

# =============================================================================
# Constants & Colour Map
# =============================================================================
WORDS = 4        # Number of words per cluster (a guess)
CLUSTERS = 4     # Total clusters in the puzzle (16 words total)
MISS = 0       # ≤2 words correct
CLOSE = 1      # Exactly 3 words correct
HIT = 2        # Exact match to a solution cluster
ANSWERS_FILE = "connections/answers.json"

COLOURS = {
    0: "🟨",  # Level 0 = Yellow
    1: "🟩",  # Level 1 = Green
    2: "🟦",  # Level 2 = Blue
    3: "🟪"   # Level 3 = Purple
}

# =============================================================================
# JSON Loading & Helper Functions
# =============================================================================
def load_games(filename: str) -> List[Dict[str, Any]]:
    """Load the JSON file containing games."""
    try:
        with open(filename, "r") as f:
            games = json.load(f)
        return games
    except Exception as e:
        logger.error(f"Error loading file {filename}: {e}")
        raise

def pick_random_game(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Randomly select one game from the list."""
    return random.choice(games)

def extract_solution(game: Dict[str, Any]) -> List[Set[str]]:
    """Extract the solution groups from the game."""
    return [set(answer["members"]) for answer in game["answers"]]

def build_word_level_mapping(game: Dict[str, Any]) -> Dict[str, int]:
    """Build a dictionary mapping each word to its level."""
    mapping = {}
    for answer in game["answers"]:
        level = answer["level"]
        for word in answer["members"]:
            mapping[word] = level
    return mapping

# =============================================================================
# Game & Solver Class Definitions
# =============================================================================
class ConnectionsGame:
    def __init__(self, solution: List[Set[str]], all_words: Set[str]) -> None:
        self.solution = solution
        self.all_words = all_words  # Pool of all words

    def valid_guess(self, guess: List[str]) -> bool:
        """A guess must be 4 words and a subset of the overall word pool."""
        return set(guess).issubset(self.all_words) and len(guess) == WORDS

    def check_guess(self, guess: List[str]) -> int:
        """
        Check a candidate guess against the solution groups.
        Returns:
            HIT (2) if the guess exactly matches a solution cluster,
            CLOSE (1) if it differs by exactly one word,
            MISS (0) otherwise.
        """
        guess_set = set(guess)
        if guess_set in self.solution:
            return HIT
        if any(len(guess_set.symmetric_difference(cluster)) == 2 for cluster in self.solution):
            return CLOSE
        return MISS

class ConnectionsSolver:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        self.guess_history: List[Tuple[List[str], int]] = []
        self.locked_words: Deque[str] = deque(maxlen=3)  # Up to 3 locked words (in order)
        self.tried_words_per_cluster: Dict[str, Set[str]] = {}  # Tracks tried words per cluster
        self.reached_close = False  # Tracks if a CLOSE guess was hit
        self.feedback = MISS  # Most recent feedback

    def get_spacy_vectors(self, words: List[str]) -> np.ndarray:
        return np.array([self.nlp(word).vector for word in words])

    def cluster_words(self, words: List[str], n_clusters: int) -> List[Tuple[int, List[str], float]]:
        word_vectors = normalize(self.get_spacy_vectors(words))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(word_vectors)

        # Group words by cluster
        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(words[idx])

        centroids = kmeans.cluster_centers_
        distances = [np.linalg.norm(word_vectors[i] - centroids[labels[i]]) for i in range(len(words))]
        cluster_confidence = {
            i: np.mean([distances[j] for j in range(len(words)) if labels[j] == i])
            for i in range(n_clusters)
        }

        # Sort clusters by confidence (lower average distance first)
        sorted_clusters = sorted(clusters.items(), key=lambda item: cluster_confidence[item[0]])

        # Prepare candidate clusters: first WORDS words per cluster
        new_clusters = {c: words[:WORDS] for c, words in sorted_clusters}
        # Redistribute overflow words into clusters with less than WORDS words
        overflow = []
        for c, words_in_cluster in clusters.items():
            if len(words_in_cluster) > WORDS:
                overflow.extend(words_in_cluster[WORDS:])
        for word in overflow:
            available = [c for c in new_clusters if len(new_clusters[c]) < WORDS]
            if available:
                new_clusters[random.choice(available)].append(word)

        return [(c, new_clusters[c], cluster_confidence[c]) for c in new_clusters]

    @lru_cache(maxsize=None)
    def word_similarity(self, word1: str, word2: str) -> float:
        # Compute spaCy vectors
        doc1 = self.nlp(word1)
        doc2 = self.nlp(word2)
        
        # If one of the vectors is empty, fallback to TF-IDF similarity.
        if doc1.vector_norm == 0 or doc2.vector_norm == 0:
            return self.tfidf_similarity(word1, word2)
        
        return doc1.similarity(doc2)

    @lru_cache(maxsize=None)
    def tfidf_similarity(self, word1: str, word2: str) -> float:
        # Create a TF-IDF vectoriser for just the two words.
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([word1, word2])
        
        # Compute cosine similarity between the two TF-IDF vectors.
        sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return sim[0][0]

    def average_similarity(self, word: str, group: List[str]) -> float:
        sims = [self.word_similarity(word, other) for other in group if other != word]
        return sum(sims) / len(sims) if sims else 0.0

    def find_best_replacement(self, pool: Set[str], used_words: Set[str], candidate: List[str]) -> str:
        """Return the most similar unused word from the pool."""
        unused = list(pool - used_words - set(candidate))
        if not unused:
            # Fallback if all words have been tried
            return random.choice(list(pool - set(candidate)))
        unused.sort(key=lambda w: self.average_similarity(w, candidate), reverse=True)
        return unused[0]

    def modify_candidate(self, candidate: List[str], pool: Set[str], cluster_id: str) -> List[str]:
        """
        Modify the candidate guess while preserving locked words.
        Replaces the least fitting non-locked word with a new candidate.
        """
        candidate_copy = candidate.copy()

        # Ensure locked words remain
        for word in self.locked_words:
            if word not in candidate_copy:
                if len(candidate_copy) < WORDS:
                    candidate_copy.append(word)
                else:
                    non_locked = [w for w in candidate_copy if w not in self.locked_words]
                    if non_locked:
                        worst = min(non_locked, key=lambda w: self.average_similarity(w, candidate_copy))
                        candidate_copy.remove(worst)
                        candidate_copy.append(word)

        # Replace one of the non-locked words
        non_locked = [w for w in candidate_copy if w not in self.locked_words]
        if non_locked:
            worst = min(non_locked, key=lambda w: self.average_similarity(w, candidate_copy))
            candidate_copy.remove(worst)
            if self.reached_close:
                self.locked_words.append(worst)
            elif self.feedback == MISS and len(non_locked) > 1:
                # Optionally remove another low-similarity word if still in MISS mode
                next_worst = min([w for w in non_locked if w != worst],
                                 key=lambda w: self.average_similarity(w, candidate_copy))
                candidate_copy.remove(next_worst)

        # Track tried words for this cluster
        if cluster_id not in self.tried_words_per_cluster:
            self.tried_words_per_cluster[cluster_id] = set()

        # Refill candidate until it has WORDS items
        while len(candidate_copy) < WORDS:
            new_word = self.find_best_replacement(pool, self.tried_words_per_cluster[cluster_id], candidate_copy)
            candidate_copy.append(new_word)
            self.tried_words_per_cluster[cluster_id].add(new_word)

        return candidate_copy

    def solve_cluster(self, game: ConnectionsGame, cluster_id: str) -> List[str]:
        pool = set(game.all_words)
        tried = set()

        # Determine number of clusters based on pool size
        n_clusters = max(1, len(pool) // WORDS)
        clusters = self.cluster_words(list(pool), n_clusters)
        candidate = clusters[0][1][:WORDS]

        while True:
            candidate_tuple = tuple(candidate)
            if candidate_tuple in tried:
                candidate = self.modify_candidate(candidate, pool, cluster_id)
                continue

            tried.add(candidate_tuple)
            self.feedback = game.check_guess(candidate)
            self.guess_history.append((candidate.copy(), self.feedback))
            logger.info(f"Guess: {candidate} | Feedback: {self.feedback} | Locked: {list(self.locked_words)}")

            if self.feedback == CLOSE and not self.reached_close:
                logger.info("Reached CLOSE!")
                self.reached_close = True

            # If feedback is CLOSE and there are locked words, remove the most recent lock
            if self.feedback == CLOSE and self.locked_words:
                removed = self.locked_words.pop()
                logger.info(f"Removing locked word: {removed}")

            if self.feedback == HIT:
                logger.info("Cluster Solved!")
                self.locked_words.clear()
                self.reached_close = False
                return candidate
            else:
                candidate = self.modify_candidate(candidate, pool, cluster_id)

    def solve_game(self, game: ConnectionsGame) -> List[List[str]]:
        solved_clusters = []
        cluster_ids = list(range(CLUSTERS))
        while game.all_words:
            solved = self.solve_cluster(game, str(cluster_ids[len(solved_clusters)]))
            solved_clusters.append(solved)
            game.all_words -= set(solved)
        return solved_clusters

# =============================================================================
# Output Functions
# =============================================================================
def print_final_summary(game_id: int, guess_history: List[Tuple[List[str], int]], word_to_level: Dict[str, int]) -> None:
    """Print summary of all guess attempts using coloured squares."""
    print("Connections")
    print(f"Puzzle #{game_id}")
    for guess, _ in guess_history:
        row = "".join([COLOURS[word_to_level[word]] for word in guess])
        print(row)
    print(f"\nTotal mistakes: {len(guess_history) - CLUSTERS}")

def plot_results(guess_history: List[Tuple[List[str], int]]) -> None:
    """Plot solver results using Matplotlib."""
    cluster_counts = [sum(1 for _, fb in guess_history if fb == i) for i in range(3)]
    labels = ["Misses (0)", "Close (1)", "Correct (2)"]
    colors = ["gold", "lightblue", "green"]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.pie(cluster_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
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

# =============================================================================
# Simulation Functions
# =============================================================================
def select_game(filename: str = ANSWERS_FILE, game_id: int = None):
    games = load_games(filename)
    if game_id is not None:
        selected_game = next((game for game in games if game["id"] == game_id), None)
    else:
        selected_game = pick_random_game(games)

    if not selected_game:
        logger.error("No game found with the provided ID.")
        return

    return selected_game

def simulate(filename: str = ANSWERS_FILE, game_id: int = None, visualise: bool = False) -> None:
    selected_game = select_game(filename, game_id)
    game_id = selected_game["id"]
    solution = extract_solution(selected_game)
    all_words = {word for cluster in solution for word in cluster}
    word_to_level = build_word_level_mapping(selected_game)

    game = ConnectionsGame(solution, all_words)
    solver = ConnectionsSolver()
    solution_clusters = solver.solve_game(game)

    print("\nPuzzle Solved")
    print(f"Solution: {solution_clusters}")
    print_final_summary(game_id, solver.guess_history, word_to_level)
    if visualise:
        plot_results(solver.guess_history)

def manual(filename: str = "answers.json", game_id: int = None) -> None:
    selected_game = select_game(filename, game_id)
    game_id = selected_game["id"]
    solution = extract_solution(selected_game)
    all_words = {word for cluster in solution for word in cluster}
    word_to_level = build_word_level_mapping(selected_game)

    game = ConnectionsGame(solution, all_words)
    solver = ConnectionsSolver()

    while game.all_words:
        print("\nRemaining words:", ", ".join(game.all_words))
        guess = input("\nEnter guess (comma-separated): ").upper().strip().split(",")
        guess = [word.strip() for word in guess]
        if not game.valid_guess(guess):
            print("Invalid guess.")
            continue

        feedback = game.check_guess(guess)
        solver.guess_history.append((guess, feedback))
        print("Feedback:", "".join([COLOURS[word_to_level[word]] for word in guess]))
        if feedback == HIT:
            game.all_words -= set(guess)

    print("\nPuzzle Solved")
    print_final_summary(game_id, solver.guess_history, word_to_level)

# =============================================================================
# Main Execution
# =============================================================================
def main() -> None:
    simulate()

if __name__ == "__main__":
    main()