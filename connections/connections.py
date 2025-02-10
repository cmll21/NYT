"""
Connections Solver

Uses WordNet and KMeans clustering to solve the Connections game.
"""

import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Tuple, Dict, Generator, Set

SOLUTION = {{"blend", "compound", "cross", "hybrid"},   # composite
            {"lodge", "plant", "stick", "wedge"},       # lodge
            {"deed", "hotel", "house", "token"},        # items in a monopoly box
            {"birth", "cruise", "quality", "remote"}    # __ control

}

###############################################################################
# ConnectionsGame Class
###############################################################################

class ConnectionsGame:
    def __init__(self, solution: Set[Set[str]]):
        self.all_words = {word for cluster in solution for word in cluster}

###############################################################################
# ConnectionsSolver Class
###############################################################################

class ConnectionsSolver:
    def __init__(self):
        # Download necessary NLTK data (if not already downloaded)
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('brown')