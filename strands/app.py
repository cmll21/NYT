"""
Strands app orchestration.
"""

from .solver import BOARD, StrandsGame, StrandsSolver
from .views import visualize_solutions


def simulate(
    max_solutions: int = 20,
    require_spangram: bool = True,
    visualize: bool = False,
    theme: str | None = None,
) -> None:
    game = StrandsGame(BOARD)
    solver = StrandsSolver(game)
    valid_paths, valid_spangrams = solver.find_words()
    solutions = solver.find_solutions(
        require_spangram=require_spangram,
        max_solutions=max_solutions,
        theme=theme,
    )

    print(f"Found paths: {len(valid_paths)}")
    print(f"Found spangrams: {len(valid_spangrams)}")
    print(f"Solutions: {len(solutions)}")

    if visualize:
        visualize_solutions(game, solutions)


def manual(
    max_solutions: int = 20,
    require_spangram: bool = True,
    theme: str | None = None,
) -> None:
    game = StrandsGame(BOARD)
    solver = StrandsSolver(game)
    solver.find_words()
    solutions = solver.find_solutions(
        require_spangram=require_spangram,
        max_solutions=max_solutions,
        theme=theme,
    )
    visualize_solutions(game, solutions)


def main() -> None:
    manual()


if __name__ == "__main__":
    main()
