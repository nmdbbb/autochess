import chess
import torch
from utils.board_encoding import encode_board

class GameState:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.position_history = []
        self.repetitions = {}
        self._update_repetitions()
    
    def make_move(self, move: chess.Move) -> None:
        """Make a move and update game state"""
        # Make the move
        self.board.push(move)
        
        # Update histories
        self.move_history.append(move)
        self.position_history.append(self.board.copy())
        
        # Update repetition counter
        self._update_repetitions()
    
    def get_encoded_state(self) -> torch.Tensor:
        """Get encoded state for neural network input"""
        encoded = encode_board(
            board=self.board,
            history=self.position_history[-2:-1] if len(self.position_history) > 1 else None,
            move_history=self.move_history,
            repetitions=self.repetitions
        )
        return encoded.unsqueeze(0)  # Add batch dimension
    
    def _update_repetitions(self) -> None:
        """Update position repetition counter"""
        # Get FEN string without move counters
        fen = self.board.fen().split(' ')[0]
        
        # Update counter
        self.repetitions[fen] = self.repetitions.get(fen, 0) + 1
    
    def is_game_over(self) -> bool:
        """Check if game is over"""
        return self.board.is_game_over()
    
    def get_result(self) -> float:
        """Get game result from white's perspective"""
        if not self.board.is_game_over():
            return 0
        
        result = self.board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0 