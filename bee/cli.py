import argparse

from .app import manual, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description="Bee Solver CLI")
    parser.add_argument(
        "--mode",
        choices=["simulate", "manual"],
        default="manual",
        help="Choose between 'simulate' and 'manual'.",
    )
    parser.add_argument(
        "--center-letter",
        type=str,
        default="",
        help="Center letter for simulate mode.",
    )
    parser.add_argument(
        "--letters",
        type=str,
        default="",
        help="Six unique surrounding letters for simulate mode.",
    )
    args = parser.parse_args()

    if args.mode == "manual":
        manual()
    else:
        center_letter = args.center_letter.strip().lower()
        letters = args.letters.strip().lower()
        if (
            len(center_letter) != 1
            or len(letters) != 6
            or len(set(letters)) != 6
            or not center_letter.isalpha()
            or not all(c.isalpha() for c in letters)
            or center_letter in letters
        ):
            raise ValueError("Invalid simulate inputs for center/letters.")
        for word in simulate(center_letter, letters):
            print(word)


if __name__ == "__main__":
    main()
