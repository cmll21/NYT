"""
Connections data I/O helpers.
"""

import json
import logging
import random

ANSWERS_FILE = "connections/answers.json"

logger = logging.getLogger(__name__)


def load_games(filename: str) -> list[dict[str, object]]:
    try:
        with open(filename, "r") as f:
            games = json.load(f)
        return games
    except Exception as e:
        logger.error(f"Error loading file {filename}: {e}")
        raise


def pick_random_game(games: list[dict[str, object]]) -> dict[str, object]:
    return random.choice(games)


def extract_solution(game: dict[str, object]) -> list[set[str]]:
    return [set(answer["members"]) for answer in game["answers"]]


def build_word_level_mapping(game: dict[str, object]) -> dict[str, int]:
    mapping = {}
    for answer in game["answers"]:
        level = answer["level"]
        for word in answer["members"]:
            mapping[word] = level
    return mapping


def select_game(filename: str = ANSWERS_FILE, game_id: int = None):
    games = load_games(filename)
    if game_id is not None:
        selected_game = next((game for game in games if game["id"] == game_id), None)
    else:
        selected_game = pick_random_game(games)

    if not selected_game:
        logger.error("No game found with the provided ID.")
        return

    return selected_game
