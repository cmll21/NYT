"""
Strands terminal UI helpers.
"""

from .solver import Path, StrandsGame

ANSI_RESET = "\033[0m"
ANSI_CLEAR = "\033[H\033[J"
PALETTE = [
    "\033[48;5;31m",
    "\033[48;5;34m",
    "\033[48;5;166m",
    "\033[48;5;99m",
    "\033[48;5;160m",
    "\033[48;5;60m",
    "\033[48;5;22m",
    "\033[48;5;88m",
]
UNUSED_CELL = "\033[48;5;236m"


def _path_word(game: StrandsGame, path: Path) -> str:
    return "".join(game.board[i][j] for i, j in path.cells).lower()


def render_solution(game: StrandsGame, solution: list[Path], index: int, total: int) -> None:
    cell_style: dict[tuple[int, int], str] = {}
    for path_idx, path in enumerate(solution):
        style = PALETTE[path_idx % len(PALETTE)]
        for cell in path.cells:
            cell_style[cell] = style

    print(ANSI_CLEAR, end="")
    print(f"Strands Solutions [{index + 1}/{total}]")
    print()
    for i in range(game.rows):
        row = []
        for j in range(game.cols):
            style = cell_style.get((i, j), UNUSED_CELL)
            row.append(f"{style} {game.board[i][j]} {ANSI_RESET}")
        print("".join(row))
    print()
    for path_idx, path in enumerate(solution):
        style = PALETTE[path_idx % len(PALETTE)]
        word = _path_word(game, path)
        tag = " (spangram)" if path.is_spangram() else ""
        print(f"{style}   {ANSI_RESET} {path_idx + 1}. {word}{tag}")
    print()
    print("Commands: [n]ext, [p]revious, [number], [q]uit")


def visualize_solutions(game: StrandsGame, solutions: list[list[Path]]) -> None:
    if not solutions:
        print("No solutions found.")
        return

    idx = 0
    while True:
        render_solution(game, solutions[idx], idx, len(solutions))
        cmd = input("> ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            break
        if cmd in {"n", "next", ""}:
            idx = (idx + 1) % len(solutions)
            continue
        if cmd in {"p", "prev", "previous"}:
            idx = (idx - 1) % len(solutions)
            continue
        if cmd.isdigit():
            pick = int(cmd)
            if 1 <= pick <= len(solutions):
                idx = pick - 1
