import unittest
import torch
import sys
from pathlib import Path
import chess
import yaml
import os

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.transformer import ChessTransformer
from mcts.mcts import MCTS
from utils.board_encoding import encode_board
from selfplay.game_state import GameState

class TestSelfPlayLoop(unittest.TestCase):
    def setUp(self):
        # Load configuration
        config_path = os.path.join(project_root, 'config/training_config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = ChessTransformer(
            d_model=self.config['model']['d_model'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['nhead'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout']
        )
        
        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            num_simulations=self.config['mcts']['simulations'],
            c_puct=self.config['mcts']['cpuct']
        )
    
    def test_game_state_initialization(self):
        """Test that GameState initializes correctly"""
        game_state = GameState()
        
        # Check initial board state
        self.assertEqual(game_state.board.fen(), chess.STARTING_FEN)
        self.assertEqual(len(game_state.move_history), 0)
        self.assertEqual(len(game_state.position_history), 1)
        self.assertEqual(len(game_state.repetitions), 1)
    
    def test_move_making(self):
        """Test that moves are properly recorded in GameState"""
        game_state = GameState()
        
        # Make a move
        move = chess.Move.from_uci("e2e4")
        game_state.make_move(move)
        
        # Check state updates
        self.assertEqual(len(game_state.move_history), 1)
        self.assertEqual(len(game_state.position_history), 2)
        self.assertEqual(game_state.move_history[0], move)
        
        # Compare only the piece placement part of FEN
        expected_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        actual_fen = game_state.board.fen().split()[0]
        self.assertEqual(actual_fen, expected_fen)
    
    def test_encoding_consistency(self):
        """Test that board encoding is consistent with moves"""
        game_state = GameState()
        
        # Make a move
        move = chess.Move.from_uci("e2e4")
        game_state.make_move(move)
        encoded = game_state.get_encoded_state()
        
        # Check encoding shape
        self.assertEqual(encoded.shape, (1, 119, 8, 8))
        
        # Check piece positions in encoding
        # White pawn should be at e4 (square 28)
        piece_planes = encoded[0, :48]  # First 48 planes are piece positions
        
        # White pawn plane index: current position (0) * 24 + color(0) * 12 + piece_type(0) * 2 = 0
        # e4 is (4, 4) in chess coordinates, but we need to flip the row
        self.assertEqual(piece_planes[0, 3, 4], 1.0)  # White pawn at e4 (row 3 = rank 5)
        
        # Original square should be empty
        self.assertEqual(piece_planes[0, 1, 4], 0.0)  # No white pawn at e2 (row 1 = rank 2)
    
    def test_game_completion(self):
        """Test that games complete properly"""
        game_state = GameState()
        
        # Play a short game
        moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            game_state.make_move(move)
        
        # Check game result
        self.assertTrue(game_state.board.is_game_over())
        self.assertTrue(game_state.board.is_checkmate())
        self.assertEqual(game_state.board.result(), "1-0")
    
    def test_repetition_tracking(self):
        """Test that position repetitions are properly tracked"""
        game_state = GameState()
        
        # Make moves that repeat the position
        moves = ["g1f3", "b8c6", "f3g1", "c6b8"]  # Return to starting position
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            game_state.make_move(move)
        
        # Check repetition count
        board_fen = game_state.board.fen().split()[0]
        self.assertEqual(game_state.repetitions[board_fen], 2)

if __name__ == '__main__':
    unittest.main() 