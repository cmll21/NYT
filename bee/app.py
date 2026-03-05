"""
Bee app orchestration.
"""

from .solver import generate_words


def manual() -> None:
    center_letter = ""
    letters = ""

    while len(center_letter) != 1 or not center_letter.isalpha():
        center_letter = input("Enter the center letter: ").strip().lower()
    while (
        len(letters) != 6
        or len(set(letters)) != 6
        or not all(c.isalpha() for c in letters)
        or center_letter in letters
    ):
        letters = input("Enter the 6 surrounding letters: ").strip().lower()

    for word in generate_words(center_letter, letters):
        print(word)


def simulate(center_letter: str, letters: str) -> list[str]:
    return generate_words(center_letter, letters)


def main() -> None:
    manual()


if __name__ == "__main__":
    main()
