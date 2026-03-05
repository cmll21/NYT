"""
Strands core solver logic.
"""

import logging

import spacy
from nltk.corpus import words

MAX_WLEN = 10
logger = logging.getLogger(__name__)

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BOARD = [
    ["T", "E", "N", "H", "C", "N"],
    ["O", "O", "K", "A", "U", "C"],
    ["R", "A", "H", "O", "L", "R"],
    ["A", "N", "P", "T", "O", "P"],
    ["G", "A", "N", "A", "K", "N"],
    ["C", "E", "D", "U", "I", "O"],
    ["R", "P", "A", "V", "B", "S"],
    ["S", "T", "N", "I", "O", "U"],
]


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True


class Path:
    def __init__(self, cells: list[tuple[int, int]]) -> None:
        self.cells = cells
        self.word = "".join([BOARD[i][j] for i, j in cells])

    def is_spangram(self) -> bool:
        lr = any([c[0] == 0 for c in self.cells]) and any(
            [c[0] == len(BOARD) - 1 for c in self.cells]
        )
        ud = any([c[1] == 0 for c in self.cells]) and any(
            [c[1] == len(BOARD[0]) - 1 for c in self.cells]
        )
        return lr or ud

    def __repr__(self) -> str:
        return f"Path({self.word}, {self.cells})"

    def __str__(self) -> str:
        return self.word

    def __len__(self) -> int:
        return len(self.word)

    def __eq__(self, other) -> bool:
        return not set(self.cells).isdisjoint(other.cells)


class StrandsGame:
    def __init__(self, board: list[list[str]]) -> None:
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])


class StrandsSolver:
    _nlp = None

    def __init__(self, game: StrandsGame) -> None:
        self.game = game

        self.trie = Trie()
        for word in words.words():
            word = word.lower()
            if 3 <= len(word) <= MAX_WLEN:
                self.trie.insert(word)
        self.found_paths = []

    @classmethod
    def _get_nlp(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load("en_core_web_lg")
        return cls._nlp

    def _theme_similarity_scores(
        self, theme_text: str, candidate_words: list[str]
    ) -> list[float]:
        nlp = self._get_nlp()
        theme_doc = nlp(theme_text)
        if theme_doc.vector_norm == 0:
            logger.warning("Theme vector is empty: %r", theme_text)
            return [0.0] * len(candidate_words)

        scores = []
        for word in candidate_words:
            doc = nlp(word)
            if doc.vector_norm == 0:
                scores.append(0.0)
            else:
                scores.append(float(theme_doc.similarity(doc)))
        return scores

    def dfs(
        self,
        i: int,
        j: int,
        current_cells: list[tuple[int, int]],
        visited,
        prefix: str,
        node: TrieNode,
    ) -> None:
        letter = self.game.board[i][j].lower()
        if letter not in node.children:
            return
        prefix += letter
        node = node.children[letter]

        if node.is_word and len(prefix) >= 3:
            path_obj = Path(current_cells)
            self.found_paths.append(path_obj)

        if len(prefix) >= MAX_WLEN:
            return

        for di, dj in DIRECTIONS:
            ni, nj = i + di, j + dj
            if (
                0 <= ni < self.game.rows
                and 0 <= nj < self.game.cols
                and (ni, nj) not in visited
            ):
                new_cells = current_cells + [(ni, nj)]
                self.dfs(ni, nj, new_cells, visited | {(ni, nj)}, prefix, node)

    def find_words(self) -> tuple[list[Path], list[Path]]:
        self.found_paths = []
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                self.dfs(i, j, [(i, j)], {(i, j)}, "", self.trie.root)

        spangrams = [path for path in self.found_paths if path.is_spangram()]
        print(f"Found {len(self.found_paths)} words, {len(spangrams)} are spangrams")
        return self.found_paths, spangrams

    def find_solutions(
        self,
        require_spangram: bool = True,
        max_solutions: int | None = None,
        theme: str | None = None,
    ) -> list[list[Path]]:
        if not self.found_paths:
            self.find_words()

        total_cells = self.game.rows * self.game.cols
        full_mask = (1 << total_cells) - 1

        def cell_to_bit(i: int, j: int) -> int:
            return 1 << (i * self.game.cols + j)

        def path_to_mask(path: Path) -> int:
            mask = 0
            for i, j in path.cells:
                mask |= cell_to_bit(i, j)
            return mask

        unique: dict[tuple[tuple[int, int], ...], Path] = {}
        for path in self.found_paths:
            key = tuple(path.cells)
            if key not in unique:
                unique[key] = path
        candidates = list(unique.values())
        if not candidates:
            return []

        theme_text = theme.strip().lower() if theme is not None else ""
        row_theme_scores = [0.0] * len(candidates)
        if theme_text:
            row_theme_scores = self._theme_similarity_scores(
                theme_text, [path.word.lower() for path in candidates]
            )

        row_cell_masks = [path_to_mask(path) for path in candidates]
        spangram_rows_mask = 0
        for i, path in enumerate(candidates):
            if path.is_spangram():
                spangram_rows_mask |= 1 << i
        if require_spangram and spangram_rows_mask == 0:
            return []

        col_rows = [0] * total_cells
        for row_idx, cell_mask in enumerate(row_cell_masks):
            bit = 1 << row_idx
            m = cell_mask
            while m:
                lsb = m & -m
                cell_idx = lsb.bit_length() - 1
                col_rows[cell_idx] |= bit
                m ^= lsb

        conflict_rows = [0] * len(candidates)
        for row_idx, cell_mask in enumerate(row_cell_masks):
            conf = 0
            m = cell_mask
            while m:
                lsb = m & -m
                cell_idx = lsb.bit_length() - 1
                conf |= col_rows[cell_idx]
                m ^= lsb
            conflict_rows[row_idx] = conf

        def choose_most_constrained_col(
            uncovered_cols: int, active_rows_mask: int
        ) -> int:
            best_col = -1
            best_count = None
            cols = uncovered_cols
            while cols:
                lsb = cols & -cols
                col = lsb.bit_length() - 1
                options = col_rows[col] & active_rows_mask
                count = options.bit_count()
                if count == 0:
                    return -2
                if best_count is None or count < best_count:
                    best_count = count
                    best_col = col
                    if count == 1:
                        break
                cols ^= lsb
            return best_col

        solutions_idx: list[list[int]] = []
        seen_signatures: set[tuple[int, ...]] = set()

        def backtrack(
            uncovered_cols: int,
            active_rows_mask: int,
            chosen: list[int],
            has_spangram: bool,
        ) -> None:
            if max_solutions is not None and len(solutions_idx) >= max_solutions:
                return

            if uncovered_cols == 0:
                if require_spangram and not has_spangram:
                    return
                signature = tuple(sorted(chosen))
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    solutions_idx.append(chosen.copy())
                return

            if require_spangram and not has_spangram:
                if (active_rows_mask & spangram_rows_mask) == 0:
                    return

            col = choose_most_constrained_col(uncovered_cols, active_rows_mask)
            if col < 0:
                return

            row_options_mask = col_rows[col] & active_rows_mask
            row_options: list[int] = []
            m = row_options_mask
            while m:
                lsb = m & -m
                row_idx = lsb.bit_length() - 1
                row_options.append(row_idx)
                m ^= lsb
            if theme_text:
                row_options.sort(
                    key=lambda idx: (row_theme_scores[idx], len(candidates[idx])),
                    reverse=True,
                )
            else:
                row_options.sort(key=lambda idx: len(candidates[idx]), reverse=True)

            for row_idx in row_options:
                next_uncovered = uncovered_cols & ~row_cell_masks[row_idx]
                next_active = active_rows_mask & ~conflict_rows[row_idx]
                chosen.append(row_idx)
                backtrack(
                    next_uncovered,
                    next_active,
                    chosen,
                    has_spangram or ((spangram_rows_mask >> row_idx) & 1) == 1,
                )
                chosen.pop()
                if max_solutions is not None and len(solutions_idx) >= max_solutions:
                    return

        all_rows_mask = (1 << len(candidates)) - 1
        backtrack(full_mask, all_rows_mask, [], False)

        if theme_text:

            def score_solution(solution: list[int]) -> tuple[float, float]:
                total = sum(row_theme_scores[idx] for idx in solution)
                avg = total / len(solution) if solution else 0.0
                return avg, total

            solutions_idx.sort(key=score_solution, reverse=True)

        return [[candidates[idx] for idx in solution] for solution in solutions_idx]


def simulate(
    max_solutions: int = 20,
    require_spangram: bool = True,
    visualize: bool = False,
    theme: str | None = None,
) -> None:
    from .views import visualize_solutions

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
    from .views import visualize_solutions

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
