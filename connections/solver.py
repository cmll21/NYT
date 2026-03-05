"""
Connections core solver logic.
"""

import random
import logging
from collections import deque
from functools import cache

import numpy as np
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .data import build_word_level_mapping, extract_solution, select_game
from .views import plot_results, print_final_summary

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

WORDS = 4
CLUSTERS = 4
MISS = 0
CLOSE = 1
HIT = 2
ANSWERS_FILE = "connections/answers.json"

COLORS = {
    0: "🟨",
    1: "🟩",
    2: "🟦",
    3: "🟪",
}
FEEDBACK_LABELS = {
    MISS: "MISS",
    CLOSE: "CLOSE",
    HIT: "HIT",
}


class ConnectionsGame:
    def __init__(self, solution: list[set[str]], all_words: set[str]) -> None:
        self.solution = solution
        self.all_words = all_words

    def valid_guess(self, guess: list[str]) -> bool:
        return set(guess).issubset(self.all_words) and len(guess) == WORDS

    def check_guess(self, guess: list[str]) -> int:
        guess_set = set(guess)
        if guess_set in self.solution:
            return HIT
        if any(
            len(guess_set.symmetric_difference(cluster)) == 2
            for cluster in self.solution
        ):
            return CLOSE
        return MISS


class ConnectionsSolver:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")
        self.guess_history: list[tuple[list[str], int]] = []
        self.locked_words: deque[str] = deque(maxlen=3)
        self.tried_words_per_cluster: dict[str, set[str]] = {}
        self.reached_close = False
        self.feedback = MISS

    def get_spacy_vectors(self, words: list[str]) -> np.ndarray:
        return np.array([self.nlp(word).vector for word in words])

    def cluster_words(
        self, words: list[str], n_clusters: int
    ) -> list[tuple[int, list[str], float]]:
        word_vectors = normalize(self.get_spacy_vectors(words))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(word_vectors)

        clusters = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clusters[label].append(words[idx])

        centroids = kmeans.cluster_centers_
        distances = [
            np.linalg.norm(word_vectors[i] - centroids[labels[i]])
            for i in range(len(words))
        ]
        cluster_confidence = {
            i: np.mean([distances[j] for j in range(len(words)) if labels[j] == i])
            for i in range(n_clusters)
        }

        sorted_clusters = sorted(
            clusters.items(), key=lambda item: cluster_confidence[item[0]]
        )

        new_clusters = {c: words[:WORDS] for c, words in sorted_clusters}
        overflow = []
        for c, words_in_cluster in clusters.items():
            if len(words_in_cluster) > WORDS:
                overflow.extend(words_in_cluster[WORDS:])
        for word in overflow:
            available = [c for c in new_clusters if len(new_clusters[c]) < WORDS]
            if available:
                new_clusters[random.choice(available)].append(word)

        return [(c, new_clusters[c], cluster_confidence[c]) for c in new_clusters]

    @cache
    def word_similarity(self, word1: str, word2: str) -> float:
        doc1 = self.nlp(word1)
        doc2 = self.nlp(word2)

        if doc1.vector_norm == 0 or doc2.vector_norm == 0:
            return self.tfidf_similarity(word1, word2)

        return doc1.similarity(doc2)

    @cache
    def tfidf_similarity(self, word1: str, word2: str) -> float:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([word1, word2])
        sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        return sim[0][0]

    def average_similarity(self, word: str, group: list[str]) -> float:
        sims = [self.word_similarity(word, other) for other in group if other != word]
        return sum(sims) / len(sims) if sims else 0.0

    def find_best_replacement(
        self, pool: set[str], used_words: set[str], candidate: list[str]
    ) -> str:
        unused = list(pool - used_words - set(candidate))
        if not unused:
            return random.choice(list(pool - set(candidate)))
        unused.sort(key=lambda w: self.average_similarity(w, candidate), reverse=True)
        return unused[0]

    def modify_candidate(
        self, candidate: list[str], pool: set[str], cluster_id: str
    ) -> list[str]:
        candidate_copy = candidate.copy()

        for word in self.locked_words:
            if word not in candidate_copy:
                if len(candidate_copy) < WORDS:
                    candidate_copy.append(word)
                else:
                    non_locked = [
                        w for w in candidate_copy if w not in self.locked_words
                    ]
                    if non_locked:
                        worst = min(
                            non_locked,
                            key=lambda w: self.average_similarity(w, candidate_copy),
                        )
                        candidate_copy.remove(worst)
                        candidate_copy.append(word)

        non_locked = [w for w in candidate_copy if w not in self.locked_words]
        if non_locked:
            worst = min(
                non_locked, key=lambda w: self.average_similarity(w, candidate_copy)
            )
            candidate_copy.remove(worst)
            if self.reached_close:
                self.locked_words.append(worst)
            elif self.feedback == MISS and len(non_locked) > 1:
                next_worst = min(
                    [w for w in non_locked if w != worst],
                    key=lambda w: self.average_similarity(w, candidate_copy),
                )
                candidate_copy.remove(next_worst)

        if cluster_id not in self.tried_words_per_cluster:
            self.tried_words_per_cluster[cluster_id] = set()

        while len(candidate_copy) < WORDS:
            new_word = self.find_best_replacement(
                pool, self.tried_words_per_cluster[cluster_id], candidate_copy
            )
            candidate_copy.append(new_word)
            self.tried_words_per_cluster[cluster_id].add(new_word)

        return candidate_copy

    def solve_cluster(self, game: ConnectionsGame, cluster_id: str) -> list[str]:
        pool = set(game.all_words)
        tried = set()

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
            logger.info(
                f"Guess: {candidate} | Feedback: {self.feedback} | Locked: {list(self.locked_words)}"
            )

            if self.feedback == CLOSE and not self.reached_close:
                logger.info("Reached CLOSE!")
                self.reached_close = True

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

    def solve_game(self, game: ConnectionsGame) -> list[list[str]]:
        solved_clusters = []
        cluster_ids = list(range(CLUSTERS))
        while game.all_words:
            solved = self.solve_cluster(game, str(cluster_ids[len(solved_clusters)]))
            solved_clusters.append(solved)
            game.all_words -= set(solved)
        return solved_clusters


def simulate(
    filename: str = ANSWERS_FILE, game_id: int = None, visualize: bool = False
) -> None:
    selected_game = select_game(filename, game_id)
    if selected_game is None:
        return

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
    if visualize:
        plot_results(solver.guess_history)


def manual(filename: str = ANSWERS_FILE, game_id: int = None) -> None:
    selected_game = select_game(filename, game_id)
    if selected_game is None:
        return

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
        print(f"Feedback: {FEEDBACK_LABELS[feedback]}")
        if feedback == HIT:
            game.all_words -= set(guess)

    print("\nPuzzle Solved")
    print_final_summary(game_id, solver.guess_history, word_to_level)


def main() -> None:
    simulate()


if __name__ == "__main__":
    main()
