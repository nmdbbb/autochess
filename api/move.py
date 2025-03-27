from flask import Flask, request, jsonify
import chess
from chess_env.board import ChessBoard
from models.utils import load_model
from mcts.mcts import MCTS
from utils.config import load_config

app = Flask(__name__)
model = load_model("data/models/model_best.pth")
config = load_config()
mcts = MCTS(model, config['mcts'])

@app.route("/api/move", methods=["POST"])
def get_move():
    data = request.get_json()
    fen = data.get("fen")
    if not fen:
        return jsonify({"error": "Missing FEN"}), 400

    board = ChessBoard()
    board.board = chess.Board(fen)
    move = mcts.select_move(board)
    return jsonify({"move": move.uci()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
