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

    # Visualization
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show visualizations after solving"
    )

    args = parser.parse_args()
    if args.mode == "simulate":
        simulate(filename = args.file, game_id = args.id, visualize = args.visualize)
    elif args.mode == "manual":
        manual(filename = args.file, game_id = args.id)

if __name__ == "__main__":
    main()
