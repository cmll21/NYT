import argparse
from .app import simulate, manual


def main() -> None:
    parser = argparse.ArgumentParser(description="Strands Solver CLI")

    parser.add_argument(
        "--mode",
        choices=["simulate", "manual"],
        default="simulate",
        help="Choose between 'simulate' and 'manual'.",
    )
    parser.add_argument(
        "--max-solutions",
        type=int,
        default=20,
        help="Maximum number of solutions to enumerate.",
    )
    parser.add_argument(
        "--no-spangram-required",
        action="store_true",
        help="Allow solutions that do not include a spangram.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show interactive terminal visualization (simulate mode).",
    )
    parser.add_argument(
        "--theme",
        nargs="+",
        default=None,
        help="Optional theme phrase used to rank and guide solution search.",
    )

    args = parser.parse_args()
    require_spangram = not args.no_spangram_required
    theme = " ".join(args.theme).strip() if args.theme else None

    if args.mode == "simulate":
        simulate(
            max_solutions=args.max_solutions,
            require_spangram=require_spangram,
            visualize=args.visualize,
            theme=theme,
        )
    elif args.mode == "manual":
        manual(
            max_solutions=args.max_solutions,
            require_spangram=require_spangram,
            theme=theme,
        )


if __name__ == "__main__":
    main()
