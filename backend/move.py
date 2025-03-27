
from flask import Flask, request, jsonify
from flask_cors import CORS
import chess

app = Flask(__name__)
CORS(app)

@app.route("/api/move", methods=["POST"])
def get_ai_move():
    data = request.get_json()
    fen = data.get("fen", "")
    board = chess.Board(fen)

    # Placeholder: chọn nước đi ngẫu nhiên
    move = str(next(iter(board.legal_moves)))

    return jsonify({"move": move})

if __name__ == "__main__":
    app.run(debug=True)
