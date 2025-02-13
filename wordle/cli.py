import argparse
from wordle import simulate, manual  # Import your solver functions

def main():
    """Command-line interface for Wordle Solver."""
    parser = argparse.ArgumentParser(description="Wordle Solver CLI")
    
    # Mode Selection
    parser.add_argument(
        "--mode", choices=["simulate", "manual"], default="manual",
        help="Choose between 'simulate' and 'manual'."
    )
    
    # Strategy Selection
    parser.add_argument(
        "--strategy", choices=["frequency", "entropy", "hybrid"], default="hybrid",
        help="Select the solving strategy."
    )
    
    # File Inputs
    parser.add_argument(
        "--words-file", type=str, default="allowed-guesses.txt",
        help="File containing allowed guess words."
    )
    parser.add_argument(
        "--answers-file", type=str, default="answers-alphabetical.txt",
        help="File containing valid Wordle answers."
    )
    
    # Optional Initial Guess
    parser.add_argument(
        "--initial-guess", type=str, default=None,
        help="Specify an initial guess (default: None)."
    )

    args = parser.parse_args()

    # Run the selected mode
    if args.mode == "simulate":
        simulate(all_candidates_file=args.answers_file, all_words_file=args.words_file, 
                 version=args.strategy, initial_guess = args.initial_guess)
    elif args.mode == "manual":
        manual(all_candidates_file=args.answers_file, all_words_file=args.words_file, version=args.strategy)

if __name__ == "__main__":
    main()
