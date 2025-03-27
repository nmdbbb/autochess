import chess
import chess.engine
from chess_env.board import ChessBoard
from models.utils import load_model
from mcts.mcts import MCTS
from utils.config import load_config
import time

def evaluate_vs_stockfish(n_games=10, stockfish_path="stockfish"):
    config = load_config()
    model = load_model("data/models/model_best.pth")
    mcts = MCTS(model, config['mcts'])

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    wins, losses, draws = 0, 0, 0

    for i in range(n_games):
        board = ChessBoard()
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        print(f"\nGame {i+1}: {'AI plays White' if color == chess.WHITE else 'AI plays Black'}")

        while not board.is_game_over():
            if board.board.turn == color:
                move = mcts.select_move(board)
            else:
                result = engine.play(board.board, chess.engine.Limit(time=0.1))
                move = result.move
            board.push(move)

        print("Result:", board.result())
        r = board.result()
        if r == "1-0":
            if color == chess.WHITE:
                wins += 1
            else:
                losses += 1
        elif r == "0-1":
            if color == chess.BLACK:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    engine.quit()

    print("\n=== Evaluation vs Stockfish ===")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    total = wins + losses + draws
    print(f"Win Rate: {wins/total:.2%}, Draw Rate: {draws/total:.2%}")

if __name__ == "__main__":
    evaluate_vs_stockfish(n_games=10, stockfish_path="stockfish")
