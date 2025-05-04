import unittest
import torch
import sys
from pathlib import Path
import chess
import numpy as np
import torch.nn.functional as F

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models.transformer import ChessTransformer, loss_function
from utils.board_encoding import encode_board

class TestTrainStep(unittest.TestCase):
    def setUp(self):
        self.model = ChessTransformer(
            d_model=256,
            num_layers=6,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0005,
            weight_decay=1e-4
        )
        
        # Create a batch of board states
        self.batch_size = 4
        self.boards = [chess.Board() for _ in range(self.batch_size)]
        self.encoded_states = torch.stack([
            encode_board(board).unsqueeze(0)
            for board in self.boards
        ]).squeeze(1)
    
    def test_forward_backward(self):
        """Test that forward and backward passes work correctly"""
        # Forward pass
        policy_logits, value_pred = self.model(self.encoded_states)
        
        # Create dummy targets
        policy_target = torch.softmax(torch.randn(self.batch_size, 4672), dim=1)
        value_target = torch.randn(self.batch_size)
        
        # Compute loss
        loss, loss_dict = loss_function(
            policy_logits, policy_target,
            value_pred, value_target,
            self.model
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check that loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check that all components of loss are present
        self.assertIn('value_loss', loss_dict)
        self.assertIn('policy_loss', loss_dict)
        self.assertIn('l2_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network"""
        # Forward pass
        policy_logits, value_pred = self.model(self.encoded_states)
        
        # Create dummy targets
        policy_target = torch.softmax(torch.randn(self.batch_size, 4672), dim=1)
        value_target = torch.randn(self.batch_size)
        
        # Compute loss
        loss, _ = loss_function(
            policy_logits, policy_target,
            value_pred, value_target,
            self.model
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients exist and are not zero
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)))
    
    def test_optimizer_step(self):
        """Test that optimizer step updates parameters"""
        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        # Forward pass
        policy_logits, value_pred = self.model(self.encoded_states)
        
        # Create dummy targets
        policy_target = torch.softmax(torch.randn(self.batch_size, 4672), dim=1)
        value_target = torch.randn(self.batch_size)
        
        # Compute loss and update
        loss, _ = loss_function(
            policy_logits, policy_target,
            value_pred, value_target,
            self.model
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Check that parameters have changed
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertFalse(torch.allclose(param, initial_params[name]))
    
    def test_loss_components(self):
        """Test that loss components are computed correctly"""
        # Forward pass
        policy_logits, value_pred = self.model(self.encoded_states)
        
        # Create dummy targets
        policy_target = torch.softmax(torch.randn(self.batch_size, 4672), dim=1)
        value_target = torch.randn(self.batch_size)
        
        # Compute loss
        loss, loss_dict = loss_function(
            policy_logits, policy_target,
            value_pred, value_target,
            self.model
        )
        
        # Check value loss (MSE)
        value_loss = torch.mean((value_pred - value_target) ** 2)
        self.assertAlmostEqual(loss_dict['value_loss'], value_loss.item())
        
        # Check policy loss (KL divergence)
        policy = F.log_softmax(policy_logits, dim=-1)
        policy_loss = F.kl_div(policy, policy_target, reduction='batchmean')
        self.assertAlmostEqual(loss_dict['policy_loss'], policy_loss.item())
        
        # Check L2 loss
        l2_loss = 0
        for param in self.model.parameters():
            l2_loss += torch.norm(param) ** 2
        l2_loss *= 1e-4
        self.assertAlmostEqual(loss_dict['l2_loss'], l2_loss.item())

if __name__ == '__main__':
    unittest.main() 