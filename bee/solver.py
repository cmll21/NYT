"""
Bee core logic.
"""

from nltk.corpus import words


def valid_word(word: str, center_letter: str, letters: str) -> bool:
    return (
        len(word) >= 4
        and center_letter in word
        and all(c in letters for c in word if c != center_letter)
    )


def generate_words(center_letter: str, letters: str) -> list[str]:
    word_list = set(words.words())
    return [word for word in word_list if valid_word(word, center_letter, letters)]
