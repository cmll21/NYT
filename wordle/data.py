"""
Wordle data loading helpers.
"""

WLEN: int = 5


def load_words(filename: str) -> set[str]:
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f if len(line.strip()) == WLEN}
