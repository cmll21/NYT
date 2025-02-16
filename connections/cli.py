import argparse
from connections import simulate, manual

def main() -> None:
    parser = argparse.ArgumentParser(description="Connections Game Solver CLI")

    # Mode selection
    parser.add_argument(
        "--mode", choices=["simulate", "manual"], default="simulate",
        help="Choose between 'simulate' and 'manual'."
    )

    # File inputs
    parser.add_argument(
        "--file", type=str, default="connections/answers.json", 
        help="Path to JSON file containing games"
    )
    
    parser.add_argument(
        "--id", type=int, default = None,
        help="Select a specific game ID"
    )

    args = parser.parse_args()
    if args.mode == "simulate":
        simulate(filename = args.file, game_id = args.id)
    elif args.mode == "manual":
        manual(filename = args.file, game_id = args.id)

if __name__ == "__main__":
    main()