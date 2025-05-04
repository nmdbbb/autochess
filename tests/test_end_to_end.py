import unittest
import chess
import torch
import numpy as np
from models.transformer import ChessTransformer
from mcts.mcts import MCTS
from selfplay.game_state import GameState
from utils.board_encoding import encode_board

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        # Initialize model
        self.model = ChessTransformer(
            d_model=128,
            num_layers=3,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1
        )
        self.model.eval()
        
        # Initialize MCTS
        self.mcts = MCTS(
            model=self.model,
            num_simulations=10,  # Reduced for testing
            c_puct=1.0
        )
        
        # Initialize game state
        self.game_state = GameState()

    def test_model_inference(self):
        """Test that model can process a board position and return valid policy and value"""
        # Get encoded state
        encoded_state = self.game_state.get_encoded_state()
        
        # Ensure correct shape
        self.assertEqual(encoded_state.shape, (1, 119, 8, 8))
        
        # Get model predictions
        with torch.no_grad():
            policy, value = self.model(encoded_state)
        
        # Check policy shape (1, 4672) - includes all possible moves including promotions
        self.assertEqual(policy.shape, (1, 4672))
        
        # Check value is scalar between -1 and 1
        self.assertTrue(-1 <= value.item() <= 1)
        
        # Normalize policy and check it sums to approximately 1
        policy_probs = torch.softmax(policy, dim=-1)
        self.assertAlmostEqual(torch.sum(policy_probs).item(), 1.0, places=5)

    def test_mcts_search(self):
        """Test that MCTS can perform a search and return a valid move"""
        # Get initial state
        board = self.game_state.board
        encoded_state = self.game_state.get_encoded_state()
        
        # Perform MCTS search
        move = self.mcts.search(board, encoded_state)
        
        # Check that returned move is legal
        self.assertIn(move, board.legal_moves)
        
        # Make the move
        self.game_state.make_move(move)
        
        # Check that game state is updated correctly
        self.assertEqual(len(self.game_state.move_history), 1)
        self.assertEqual(self.game_state.board.move_stack[-1], move)

    def test_game_completion(self):
        """Test playing a complete game"""
        max_moves = 10  # Limit moves for testing
        moves_made = 0
        
        while not self.game_state.board.is_game_over() and moves_made < max_moves:
            # Get current state
            board = self.game_state.board
            encoded_state = self.game_state.get_encoded_state()
            
            # Get move from MCTS
            move = self.mcts.search(board, encoded_state)
            
            # Verify move is legal
            self.assertIn(move, board.legal_moves)
            
            # Make move
            self.game_state.make_move(move)
            moves_made += 1
        
        # Verify game state
        self.assertGreater(len(self.game_state.move_history), 0)
        self.assertEqual(len(self.game_state.board.move_stack), moves_made)

    def test_board_encoding_consistency(self):
        """Test that board encoding remains consistent through a sequence of moves"""
        # Make a specific move
        move = chess.Move.from_uci("e2e4")
        self.game_state.make_move(move)
        encoded = self.game_state.get_encoded_state()
        
        # Check piece planes
        piece_planes = encoded[0, :48]  # First 48 planes are piece positions
        
        # White pawn should be at e4 (row 3, col 4 in the encoding)
        self.assertEqual(piece_planes[0, 3, 4].item(), 1.0)
        
        # Original square should be empty
        self.assertEqual(piece_planes[0, 6, 4].item(), 0.0)

    def test_value_prediction_accuracy(self):
        """Test that value predictions are reasonable in clear positions"""
        # Create a position where white is clearly winning (up a queen)
        self.game_state.board.set_fen("rnb1kbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
        encoded_state = self.game_state.get_encoded_state()
        
        # Get value prediction
        with torch.no_grad():
            _, value = self.model(encoded_state)
        
        # Value should favor white (positive)
        self.assertGreater(value.item(), 0)

if __name__ == '__main__':
    unittest.main() 