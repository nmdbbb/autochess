import chess
import chess.pgn
import requests
import json
import os
from typing import Optional
import argparse

class ChessCLI:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.game_id: Optional[str] = None
        self.board = chess.Board()
        self.game_state = None

    def start_new_game(self):
        """Start a new game and get the game ID"""
        response = requests.post(f"{self.api_url}/game/new")
        if response.status_code == 200:
            self.game_state = response.json()
            self.game_id = self.game_state['board'].split(' ')[-1]
            self.board = chess.Board(self.game_state['board'])
            print("\nNew game started!")
            self.print_board()
        else:
            print("Failed to start new game")

    def make_move(self, move_uci: str):
        """Make a move in the current game"""
        if not self.game_id:
            print("No active game. Start a new game first.")
            return

        try:
            response = requests.post(
                f"{self.api_url}/game/{self.game_id}/move",
                json={"move_uci": move_uci}
            )
            if response.status_code == 200:
                self.game_state = response.json()
                self.board = chess.Board(self.game_state['board'])
                self.print_board()
            else:
                print("Invalid move")
        except Exception as e:
            print(f"Error making move: {e}")

    def get_ai_move(self):
        """Get AI's move in the current game"""
        if not self.game_id:
            print("No active game. Start a new game first.")
            return

        try:
            response = requests.post(f"{self.api_url}/game/{self.game_id}/ai-move")
            if response.status_code == 200:
                self.game_state = response.json()
                self.board = chess.Board(self.game_state['board'])
                print("\nAI's move:")
                self.print_board()
            else:
                print("Failed to get AI move")
        except Exception as e:
            print(f"Error getting AI move: {e}")

    def print_board(self):
        """Print the current board state"""
        print("\n" + str(self.board))
        print(f"\nFEN: {self.board.fen()}")
        print(f"Status: {self.game_state['status']}")
        print(f"Moves: {' '.join(self.game_state['moves'])}")

    def play(self):
        """Main game loop"""
        print("Welcome to AlphaZero Chess CLI!")
        print("Commands:")
        print("  new - Start a new game")
        print("  move <uci> - Make a move (e.g., 'move e2e4')")
        print("  ai - Get AI's move")
        print("  quit - Exit the game")
        print("  help - Show this help message")

        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("\nCommands:")
                print("  new - Start a new game")
                print("  move <uci> - Make a move (e.g., 'move e2e4')")
                print("  ai - Get AI's move")
                print("  quit - Exit the game")
                print("  help - Show this help message")
            elif command == "new":
                self.start_new_game()
            elif command.startswith("move "):
                move_uci = command.split(" ")[1]
                self.make_move(move_uci)
            elif command == "ai":
                self.get_ai_move()
            else:
                print("Invalid command. Type 'help' for available commands.")

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Chess CLI')
    parser.add_argument('--api-url', default='http://localhost:8000',
                      help='API URL (default: http://localhost:8000)')
    args = parser.parse_args()

    cli = ChessCLI(api_url=args.api_url)
    cli.play()

if __name__ == "__main__":
    main() 