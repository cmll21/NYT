"""
Connections Solver

Uses WordNet and KMeans clustering to solve the Connections game.
"""

import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Set, Tuple

# Use a list of sets rather than a set of sets.
SOLUTION = [
    {"blend", "compound", "cross", "hybrid"},   # composite
    {"lodge", "plant", "stick", "wedge"},         # lodge
    {"deed", "hotel", "house", "token"},          # items in a monopoly box
    {"birth", "cruise", "quality", "remote"}      # __ control
]

# Both the number of words per cluster and the number of clusters are 4.
WORDS = 4
CLUSTERS = 4
MISS = 0
CLOSE = 1
HIT = 2

###############################################################################
# ConnectionsGame Class
###############################################################################

class ConnectionsGame:
    def __init__(self, solution: List[Set[str]]):
        self.solution = solution
        # Collect all words from all clusters into a set.
        self.all_words = {word for cluster in solution for word in cluster}
    
    def valid_guess(self, guess: Set[str]) -> bool:
        return guess.issubset(self.all_words) and len(guess) == WORDS

    def check_guess(self, guess: Set[str]) -> int:
        """
        Check the guess against the solution.
        Returns:
            0 if no words are in the solution,
            1 if one off from the solution,
            2 if all words are in the solution.
        """
        if guess in self.solution:
            return HIT
        if any(len(guess.symmetric_difference(cluster)) == 2 for cluster in self.solution):
            return CLOSE
        return MISS

###############################################################################
# ConnectionsSolver Class
###############################################################################

class ConnectionsSolver:
    def __init__(self):
        # Download necessary NLTK data (if not already downloaded)
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('brown')

    def get_word_profile(self, word: str) -> str:
        synsets = wn.synsets(word)
        if synsets:
            definitions = [s.definition() for s in synsets]
            examples = [ex for s in synsets for ex in s.examples()]
            synonyms = [l.name() for s in synsets for l in s.lemmas()]
            return " ".join(definitions + examples + synonyms)
        return word

    def cluster_words(self, words: List[str], n_clusters: int) -> List[Tuple[int, List[str], float]]:
        """
        Cluster the words using KMeans (with overflow prevention) and sort clusters
        by confidence (average distance from centroid).
        """
        profiles = [self.get_word_profile(word) for word in words]
        
        # Create TF-IDF vectors from the richer profiles.
        vectoriser = TfidfVectorizer(stop_words='english')
        X = vectoriser.fit_transform(profiles)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(X)
        X_dense = X.toarray()
        centroids = kmeans.cluster_centers_

        # Compute the distance of each word vector to its assigned cluster centroid.
        distances = [np.linalg.norm(X_dense[i] - centroids[labels[i]])
                     for i in range(len(words))]
        
        # Organise indices by cluster label.
        clusters_indices: Dict[int, List[int]] = {label: [] for label in range(n_clusters)}
        for i, label in enumerate(labels):
            clusters_indices[label].append(i)
        
        # Sort each cluster by distance (closest words first).
        for label in clusters_indices:
            clusters_indices[label].sort(key=lambda i: distances[i])
        
        # Limit each cluster to at most WORDS elements.
        new_clusters = {label: clusters_indices[label][:WORDS] for label in clusters_indices}
        
        # Identify overflow indices (words beyond the allowed WORDS per cluster).
        overflow = []
        for label in clusters_indices:
            if len(clusters_indices[label]) > WORDS:
                overflow.extend(clusters_indices[label][WORDS:])
        
        # Redistribute each overflow word to the nearest cluster that is not yet full.
        for i in overflow:
            word_vec = X_dense[i]
            dists_to_centroids = [np.linalg.norm(word_vec - centroid) for centroid in centroids]
            sorted_centroids = np.argsort(dists_to_centroids)
            for c in sorted_centroids:
                if len(new_clusters[c]) < WORDS:
                    new_clusters[c].append(i)
                    break
        
        # Build final clusters by converting indices to actual words.
        final_clusters: Dict[int, List[str]] = {c: [words[i] for i in indices] for c, indices in new_clusters.items()}
        
        # Recompute cluster centroids based on final indices and compute average distance (confidence).
        final_centroids = {}
        for c, indices in new_clusters.items():
            if indices:
                final_centroids[c] = np.mean([X_dense[i] for i in indices], axis=0)
            else:
                final_centroids[c] = None
        
        cluster_confidence = {}
        for c, indices in new_clusters.items():
            if indices and final_centroids[c] is not None:
                distances_cluster = [np.linalg.norm(X_dense[i] - final_centroids[c]) for i in indices]
                cluster_confidence[c] = np.mean(distances_cluster)
            else:
                cluster_confidence[c] = float('inf')
        
        # Sort clusters by confidence (lower average distance indicates higher confidence).
        sorted_clusters = sorted(final_clusters.items(), key=lambda item: cluster_confidence[item[0]])
        # Include confidence in the output.
        sorted_clusters = [(label, words_list, cluster_confidence[label]) for label, words_list in sorted_clusters]
        return sorted_clusters


if __name__ == "__main__":
    # Build a sorted list of all unique words from the solution clusters.
    all_words = sorted({word for cluster in SOLUTION for word in cluster})
    
    solver = ConnectionsSolver()
    clusters = solver.cluster_words(all_words, CLUSTERS)
    
    for label, cluster_words, confidence in clusters:
        print(f"Cluster {label} (confidence: {confidence:.4f}): {cluster_words}")
