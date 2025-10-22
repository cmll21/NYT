"""
Strands Solver

"""

from nltk.corpus import words
from typing import List, Tuple

MAX_WLEN = 10

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)]

# BOARD = [
#     ['T', 'E', 'N', 'H', 'C', 'N'],
#     ['O', 'O', 'K', 'A', 'U', 'C'],
#     ['R', 'A', 'H', 'O', 'L', 'R'],
#     ['A', 'N', 'P', 'T', 'O', 'P'],
#     ['G', 'A', 'N', 'A', 'K', 'N'],
#     ['C', 'E', 'D', 'U', 'I', 'O'],
#     ['R', 'P', 'A', 'V', 'B', 'S'],
#     ['S', 'T', 'N', 'I', 'O', 'U']
# ]

BOARD = [
    ['B', 'A', 'D'],
    ['C', 'Y', 'O'],
    ['A', 'T', 'G'],
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
    def __init__(self, cells: List[Tuple[int, int]]) -> None:
        self.cells = cells
        self.word = "".join([BOARD[i][j] for i, j in cells])

    def is_spangram(self) -> bool:
        lr = any([c[0] == 0 for c in self.cells]) and any([c[0] == len(BOARD) - 1 for c in self.cells])
        ud = any([c[1] == 0 for c in self.cells]) and any([c[1] == len(BOARD[0]) - 1 for c in self.cells])
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
    def __init__(self, board: List[List[str]]) -> None:
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])


class StrandsSolver:
    def __init__(self, game: StrandsGame) -> None:
        self.game = game

        self.trie = Trie()
        for word in words.words():
            word = word.lower()
            if 3 <= len(word) <= MAX_WLEN:
                self.trie.insert(word)
        self.found_paths = []

    def dfs(self, i: int, j: int, current_cells: List[Tuple[int, int]], visited, prefix: str, node: TrieNode) -> None:
        letter = self.game.board[i][j].lower()
        if letter not in node.children:
            return  # No word in the dictionary starts with this prefix.
        # Update the prefix and trie node.
        prefix += letter
        node = node.children[letter]

        # If it's a valid word and at least 3 letters, record the path.
        if node.is_word and len(prefix) >= 3:
            path_obj = Path(current_cells)
            print(f"Found word: {path_obj.word.lower()} at path: {path_obj.cells}")
            self.found_paths.append(path_obj)

        # Prune if we reached maximum allowed word length.
        if len(prefix) >= MAX_WLEN:
            return

        # Explore all 8 adjacent directions.
        for di, dj in DIRECTIONS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.game.rows and 0 <= nj < self.game.cols and (ni, nj) not in visited:
                new_cells = current_cells + [(ni, nj)]
                self.dfs(ni, nj, new_cells, visited | {(ni, nj)}, prefix, node)

    def find_words(self) -> Tuple[List[Path], List[Path]]:
        self.found_paths = []
        # Start DFS from each cell using the trie's root.
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                # Start with an empty prefix and the trie root.
                self.dfs(i, j, [(i, j)], {(i, j)}, "", self.trie.root)

        # Filter for spangrams if needed
        spangrams = [path for path in self.found_paths if path.is_spangram()]
        print(f"Found {len(self.found_paths)} words, {len(spangrams)} are spangrams")
        return self.found_paths, spangrams

    def find_solution(self) -> List[Path]:
        # Implement using bitmasking and backtracking to find solution
        ...




def main():
    game = StrandsGame(BOARD)
    solver = StrandsSolver(game)
    valid_paths, valid_spangrams = solver.find_words()

    print(valid_paths)
    print(valid_spangrams)


if __name__ == "__main__":
    main()
