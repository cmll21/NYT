"""
Connections Solver

Uses WordNet and KMeans clustering to solve the Connections game one cluster at a time.
At the end, it prints a summary of all guesses using colored squares:
Level 0 = Yellow (🟨)
Level 1 = Green  (🟩)
Level 2 = Blue   (🟦)
Level 3 = Purple (🟪)
"""

import json
import random
import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Set, Dict, Tuple, Any

# ----------------------
# Constants & Emoji Map
# ----------------------
WORDS = 4        # number of words per cluster (a guess)
CLUSTERS = 4     # total clusters in the puzzle (16 words total)
MISS = 0         # guess has ≤2 words correct
CLOSE = 1        # guess is one off (exactly 3 words correct)
HIT = 2          # guess exactly matches a solution cluster

COLOURS = {
    0: "🟨",  # Level 0 = Yellow
    1: "🟩",  # Level 1 = Green
    2: "🟦",  # Level 2 = Blue
    3: "🟪"   # Level 3 = Purple
}

# -------------------------------------------
# JSON Loading & Game/Mapping Helper Functions
# -------------------------------------------
def load_games(filename: str) -> List[Dict[str, Any]]:
    """Load the JSON file containing games."""
    with open(filename, 'r') as f:
        games = json.load(f)
    return games

def pick_random_game(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Randomly select one game from the list."""
    return random.choice(games)

def extract_solution(game: Dict[str, Any]) -> List[Set[str]]:
    """
    Extract the solution groups from the game.
    Each solution group is represented as a set of words.
    """
    return [set(answer['members']) for answer in game['answers']]

def build_word_level_mapping(game: Dict[str, Any]) -> Dict[str, int]:
    """
    Build a dictionary mapping each word (from the game answers)
    to its level.
    """
    mapping = {}
    for answer in game['answers']:
        level = answer['level']
        for word in answer['members']:
            mapping[word] = level
            
    return mapping

# ---------------------------
# Game & Solver Class Definitions
# ---------------------------
class ConnectionsGame:
    def __init__(self, solution: List[Set[str]], all_words: Set[str]):
        self.solution = solution
        self.all_words = all_words  # pool of all words

    def valid_guess(self, guess: List[str]) -> bool:
        """A guess must be 4 words and a subset of the overall word pool."""
        return set(guess).issubset(self.all_words) and len(guess) == WORDS

    def check_guess(self, guess: List[str]) -> int:
        """
        Check the candidate guess (list of 4 words) against the solution groups.
        Returns:
            HIT (2) if the guess exactly matches one solution cluster,
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
    def __init__(self):
        # Download necessary NLTK data (quietly)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('brown', quiet=True)
        # Will hold every guess (list of words) made across clusters.
        self.guess_history: List[List[str]] = []

    def get_word_profile(self, word: str) -> str:
        """Build a richer profile for the word using WordNet definitions, examples, and synonyms."""
        synsets = wn.synsets(word)
        if synsets:
            definitions = [s.definition() for s in synsets]
            examples = [ex for s in synsets for ex in s.examples()]
            synonyms = [l.name() for s in synsets for l in s.lemmas()]
            return " ".join(definitions + examples + synonyms)
        return word

    def cluster_words(self, words: List[str], n_clusters: int) -> List[Tuple[int, List[str], float]]:
        """
        Cluster the words using TF–IDF (on their enriched profiles) and KMeans.
        Returns a list of tuples: (cluster_label, list_of_words, cluster_confidence)
        """
        profiles = [self.get_word_profile(word) for word in words]
        vectoriser = TfidfVectorizer(stop_words='english')
        X = vectoriser.fit_transform(profiles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(X)

        X_dense = X.toarray()
        centroids = kmeans.cluster_centers_

        distances = [np.linalg.norm(X_dense[i] - centroids[labels[i]]) for i in range(len(words))]
        clusters_indices = {label: [] for label in range(n_clusters)}

        for i, label in enumerate(labels):
            clusters_indices[label].append(i)
        for label in clusters_indices:
            clusters_indices[label].sort(key=lambda i: distances[i])

        new_clusters = {label: clusters_indices[label][:WORDS] for label in clusters_indices}
        overflow = []

        for label in clusters_indices:
            if len(clusters_indices[label]) > WORDS:
                overflow.extend(clusters_indices[label][WORDS:])

        for i in overflow:
            word_vec = X_dense[i]
            dists_to_centroids = [np.linalg.norm(word_vec - centroid) for centroid in centroids]
            sorted_centroids = np.argsort(dists_to_centroids)
            for c in sorted_centroids:
                if len(new_clusters[c]) < WORDS:
                    new_clusters[c].append(i)
                    break

        final_clusters = {c: [words[i] for i in indices] for c, indices in new_clusters.items()}
        cluster_confidence = {}

        for c, indices in new_clusters.items():
            if indices:
                centroid = np.mean([X_dense[i] for i in indices], axis=0)
                dists = [np.linalg.norm(X_dense[i] - centroid) for i in indices]
                cluster_confidence[c] = np.mean(dists)
            else:
                cluster_confidence[c] = float('inf')

        sorted_clusters = sorted(final_clusters.items(), key=lambda item: cluster_confidence[item[0]])

        return [(label, words_list, cluster_confidence[label])
                for label, words_list in sorted_clusters]

    @staticmethod
    def word_similarity(word1: str, word2: str) -> float:
        """Compute the Wu–Palmer similarity between two words using WordNet."""
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)

        if synsets1 and synsets2:
            sim = synsets1[0].wup_similarity(synsets2[0])
            return sim if sim is not None else 0.0
        return 0.0

    def average_similarity(self, word: str, group: List[str]) -> float:
        """Compute the average similarity of 'word' with all the other words in 'group'."""
        sims = [self.word_similarity(word, other) for other in group if other != word]
        return sum(sims) / len(sims) if sims else 0.0

    def get_initial_candidate(self, pool: List[str]) -> List[str]:
        """
        Generate an initial candidate guess (list of 4 words) from the available pool,
        using clustering.
        """
        n_clusters = max(1, len(pool) // WORDS)
        clusters = self.cluster_words(pool, n_clusters)
        candidate = clusters[0][1][:WORDS]
        if len(candidate) < WORDS:
            candidate += random.sample(list(set(pool) - set(candidate)), WORDS - len(candidate))
        return candidate

    def modify_candidate(self, candidate: List[str], pool: Set[str], change_one: bool = True) -> List[str]:
        """
        Modify the candidate guess.
        If change_one is True (feedback was CLOSE) then swap out the word with the lowest average similarity;
        otherwise (MISS) do a random swap.
        """
        candidate_copy = candidate.copy()
        if change_one:
            worst_word = min(candidate_copy, key=lambda w: self.average_similarity(w, candidate_copy))
        else:
            worst_word = random.choice(candidate_copy)

        candidate_copy.remove(worst_word)
        possible = list(pool - set(candidate_copy))

        if possible:
            new_word = random.choice(possible)
            candidate_copy.append(new_word)
        else:
            candidate_copy.append(worst_word)

        return candidate_copy

    def solve_cluster(self, game: ConnectionsGame) -> List[str]:
        """
        Solve one cluster (i.e. a candidate guess of 4 words) from the current pool.
        This method repeatedly generates candidate guesses (storing each in guess_history)
        until one receives a HIT.
        """
        pool = set(game.all_words)
        tried = set()
        candidate = self.get_initial_candidate(list(pool))

        while True:
            candidate_tuple = tuple(candidate)

            if candidate_tuple in tried:
                candidate = self.modify_candidate(candidate, pool, change_one=False)
                continue

            tried.add(candidate_tuple)
            # Record the guess (order preserved)
            self.guess_history.append(candidate.copy())
            feedback = game.check_guess(candidate)
            print("Guess:", candidate, "Feedback:", feedback)

            if feedback == HIT:
                return candidate
            elif feedback == CLOSE:
                candidate = self.modify_candidate(candidate, pool, change_one=True)
            else:
                candidate = self.modify_candidate(candidate, pool, change_one=False)

    def solve_game(self, game: ConnectionsGame) -> List[List[str]]:
        """
        Solve the entire game one cluster at a time.
        After each cluster is solved, remove its words from the overall pool.
        """
        solved_clusters = []
        while game.all_words:
            solved = self.solve_cluster(game)
            solved_clusters.append(solved)
            game.all_words -= set(solved)

        return solved_clusters

# ---------------------------
# Final Summary Output Function
# ---------------------------
def print_final_summary(game_id: int, guess_history: List[List[str]], word_to_level: Dict[str, int]):
    """
    Print a summary of all guess attempts.
    Each row corresponds to one guess.
    Each square (emoji) represents the level of the word (based on the original game mapping).
    """
    print("Connections")
    print(f"Puzzle #{game_id}")

    for guess in guess_history:
        # For each word in the guess, look up its level and map it to the corresponding emoji.
        row = "".join([COLOURS[word_to_level[word]] for word in guess])
        print(row)
        
    print(f"\nTotal guesses: {len(guess_history)}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Load games from the JSON file.
    games = load_games("answers.json")
    random_game = pick_random_game(games)
    game_id = random_game["id"]
    solution = extract_solution(random_game)
    all_words = {word for cluster in solution for word in cluster}
    word_to_level = build_word_level_mapping(random_game)

    # Create the game and solver objects.
    game = ConnectionsGame(solution, all_words)
    solver = ConnectionsSolver()

    # Solve the game (one cluster at a time). All guess attempts are stored in solver.guess_history.
    solver.solve_game(game)

    # Print the final summary.
    print("\n")
    print_final_summary(game_id, solver.guess_history, word_to_level)
