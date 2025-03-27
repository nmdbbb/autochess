import chess
import numpy as np
from collections import deque

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()
        self.history = deque(maxlen=8)
        self._update_history()

    def push(self, move):
        self.board.push(move)
        self._update_history()

    def push_uci(self, move):
        self.board.push_uci(move)
        self._update_history()

    def legal_moves(self):
        return list(self.board.legal_moves)

    def is_game_over(self):
        return self.board.is_game_over()

    def result(self):
        return self.board.result()

    def result_value(self):
        result = self.board.result()
        return {'1-0': 1, '0-1': -1, '1/2-1/2': 0}.get(result, 0)

    def encode(self):
        planes = []
        for past_board in self.history:
            for piece_type in range(1, 7):
                for color in [chess.WHITE, chess.BLACK]:
                    plane = np.zeros((8, 8), dtype=np.float32)
                    for square in past_board.pieces(piece_type, color):
                        row, col = divmod(square, 8)
                        plane[row][col] = 1
                    planes.append(plane)

        planes.append(np.full((8, 8), self.board.turn, dtype=np.float32))

        planes.append(np.full((8, 8), self.board.has_kingside_castling_rights(chess.WHITE), dtype=np.float32))
        planes.append(np.full((8, 8), self.board.has_queenside_castling_rights(chess.WHITE), dtype=np.float32))
        planes.append(np.full((8, 8), self.board.has_kingside_castling_rights(chess.BLACK), dtype=np.float32))
        planes.append(np.full((8, 8), self.board.has_queenside_castling_rights(chess.BLACK), dtype=np.float32))

        ep_plane = np.zeros((8, 8), dtype=np.float32)
        if self.board.ep_square is not None:
            row, col = divmod(self.board.ep_square, 8)
            ep_plane[row][col] = 1
        planes.append(ep_plane)

        while len(planes) < 119:
            planes.append(np.zeros((8, 8), dtype=np.float32))

        return np.stack(planes)

    def _update_history(self):
        self.history.append(self.board.copy(stack=False))

    def __str__(self):
        return str(self.board)

    @property
    def turn(self):
        return int(self.board.turn)
