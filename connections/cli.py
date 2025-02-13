import argparse
import json
import random
import nltk
from typing import List, Dict, Any
from connections import (
    load_games, pick_random_game, extract_solution, build_word_level_mapping,
    ConnectionsGame, ConnectionsSolver, print_final_summary, COLOURS
)

# ------------------------------
# CLI Interface
# ------------------------------
def play_connections(filename: str, game_id: int = None, auto_solve: bool = True):
    """CLI interface for the Connections game solver."""
    
    # Load games from file
    games = load_games(filename)
    if not games:
        print("Error: No games found in the provided JSON file.")
        return

    # Select game by ID or randomly
    if game_id is not None:
        selected_game = next((game for game in games if game["id"] == game_id), None)
        if not selected_game:
            print(f"Error: No game found with ID {game_id}.")
            return
    else:
        selected_game = pick_random_game(games)

    game_id = selected_game["id"]

    # Extract solution and words
    solution = extract_solution(selected_game)
    all_words = {word for cluster in solution for word in cluster}
    word_to_level = build_word_level_mapping(selected_game)

    # Initialize game & solver
    game = ConnectionsGame(solution, all_words)
    solver = ConnectionsSolver()

    # Interactive play mode
    while game.all_words:
        # Show remaining words
        print("\nRemaining words:", ", ".join(game.all_words))
        
        # AI Solver move
        if auto_solve:
            guess = solver.solve_cluster(game)
        else:
            guess = input("\nEnter guess: ").upper().strip().split(",")
            guess = [word.strip() for word in guess]
        
        # Validate guess
        if not game.valid_guess(guess):
            print("Invalid guess.")
            continue

        # Check guess
        feedback = game.check_guess(guess)
        solver.guess_history.append(guess)
        
        # Display feedback
        if not auto_solve:
            print("Feedback:", "".join([COLOURS[word_to_level[word]] for word in guess]))

        # If it's a HIT, remove the words from the pool
        if feedback == 2:
            game.all_words -= set(guess)

    # Show final results
    print("\nPuzzle Solved\n")
    print_final_summary(game_id, solver.guess_history, word_to_level)

# ------------------------------
# CLI Argument Parser
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connections Game Solver CLI")
    parser.add_argument("--file", type=str, default="answers.json", help="Path to JSON file containing games")
    parser.add_argument("--id", type=int, help="Select a specific game ID")
    parser.add_argument("--manual", action="store_true", help="Play manually instead of AI solving")

    args = parser.parse_args()
    play_connections(filename=args.file, game_id=args.id, auto_solve=not args.manual)
