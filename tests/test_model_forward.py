import unittest
import torch
import sys
from pathlib import Path
import torch.nn.functional as F

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.transformer import ChessTransformer
from utils.board_encoding import encode_board
import chess

class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.model = ChessTransformer(
            d_model=256,
            num_layers=6,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.board = chess.Board()
    
    def test_forward_pass(self):
        """Test that the model can process a board state"""
        # Encode board state
        encoded = encode_board(self.board)
        encoded = encoded.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        policy_logits, value = self.model(encoded)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (1, 4672))  # Policy head
        self.assertEqual(value.shape, (1,))        # Value head
        
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        # Check policy is valid probability distribution
        self.assertTrue(torch.all(policy >= 0))
        self.assertTrue(torch.allclose(policy.sum(dim=1), torch.ones(1)))
        
        # Check value is in [-1, 1]
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))
    
    def test_batch_processing(self):
        """Test that the model can process multiple board states"""
        # Create batch of 4 board states
        batch_size = 4
        encoded = torch.stack([
            encode_board(chess.Board()).unsqueeze(0)
            for _ in range(batch_size)
        ]).squeeze(1)
        
        # Forward pass
        policy_logits, value = self.model(encoded)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (batch_size, 4672))
        self.assertEqual(value.shape, (batch_size,))
        
        # Convert logits to probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        # Check policy is valid probability distribution
        self.assertTrue(torch.all(policy >= 0))
        self.assertTrue(torch.allclose(policy.sum(dim=1), torch.ones(batch_size)))
        
        # Check value is in [-1, 1]
        self.assertTrue(torch.all(value >= -1))
        self.assertTrue(torch.all(value <= 1))

if __name__ == '__main__':
    unittest.main() 