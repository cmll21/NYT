from nltk.corpus import words


def main() -> None:
    word_list = set(words.words())

    center_letter = ""
    letters = ""

    # Handle input
    while len(center_letter) != 1 or not center_letter.isalpha():
        center_letter = input("Enter the center letter: ").strip().lower()
    while (
        len(set(letters)) != 6
        or not all(c.isalpha() for c in letters)
        or center_letter in letters
    ):
        letters = input("Enter the 6 surrounding letters: ").strip().lower()

    for word in word_list:
        if (
            len(word) >= 4
            and center_letter in word
            and all(c in letters for c in word if c != center_letter)
        ):
            print(word)


if __name__ == "__main__":
    main()
