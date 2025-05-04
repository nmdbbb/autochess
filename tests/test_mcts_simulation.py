import unittest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
import chess

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.transformer import ChessTransformer
from mcts.mcts import MCTS
from utils.board_encoding import encode_board

class TestMCTSSimulation(unittest.TestCase):
    def setUp(self):
        self.model = ChessTransformer(
            d_model=256,
            num_layers=6,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.mcts = MCTS(
            model=self.model,
            num_simulations=100,  # Reduced for testing
            c_puct=1.5
        )
        self.board = chess.Board()
    
    def test_legal_move_selection(self):
        """Test that MCTS selects legal moves"""
        encoded = encode_board(self.board).unsqueeze(0)
        move = self.mcts.search(self.board, encoded)
        
        # Check move is legal
        self.assertIn(move, self.board.legal_moves)
    
    def test_game_termination(self):
        """Test that MCTS handles game termination correctly"""
        # Create a checkmate position (Fool's mate)
        board = chess.Board()
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]  # Fool's mate sequence
        for move in moves:
            board.push_san(move)
        
        encoded = encode_board(board).unsqueeze(0)
        
        # MCTS should recognize the game is over
        move = self.mcts.search(board, encoded)
        self.assertTrue(board.is_game_over())
        self.assertTrue(board.is_checkmate())
    
    def test_value_propagation(self):
        """Test that MCTS properly propagates values"""
        # Create a checkmate position (Fool's mate)
        board = chess.Board()
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]  # Fool's mate sequence
        for move in moves:
            board.push_san(move)
        
        encoded = encode_board(board).unsqueeze(0)
        
        # Run MCTS
        move = self.mcts.search(board, encoded)
        
        # Check that the value is correct for checkmate
        self.assertTrue(board.is_game_over())
        self.assertTrue(board.is_checkmate())
        self.assertEqual(self.mcts.root.value(), -1.0)  # Black wins
    
    def test_policy_usage(self):
        """Test that MCTS uses the model's policy predictions"""
        encoded = encode_board(self.board).unsqueeze(0)
        
        # Get model's policy predictions
        with torch.no_grad():
            policy_logits, _ = self.model(encoded)
            policy = F.softmax(policy_logits, dim=-1)
        
        # Run MCTS
        move = self.mcts.search(self.board, encoded)
        
        # Check that the selected move has non-zero policy probability
        move_idx = self.mcts._encode_move(move)
        self.assertGreater(policy[0, move_idx], 0)
    
    def test_simulation_count(self):
        """Test that MCTS performs the correct number of simulations"""
        encoded = encode_board(self.board).unsqueeze(0)
        
        # Run MCTS
        move = self.mcts.search(self.board, encoded)
        
        # Check total visit count matches simulation count
        root = self.mcts.root
        self.assertEqual(root.visit_count, self.mcts.num_simulations)

if __name__ == '__main__':
    unittest.main() 