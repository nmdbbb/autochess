import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from models.transformer import ChessTransformer
from mcts.mcts import MCTS
from selfplay.self_play import SelfPlay

class AlphaZeroTrainer:
    def __init__(self, 
                 num_iterations: int = 1000,
                 num_games_per_iteration: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 num_simulations: int = 800):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessTransformer().to(self.device)
        self.mcts = MCTS(self.model, num_simulations=num_simulations)
        self.self_play = SelfPlay(self.model, self.mcts, num_games=num_games_per_iteration)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()
        
        self.writer = SummaryWriter('runs/alphazero')
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        
        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('runs', exist_ok=True)

    def train(self):
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            # Generate self-play data
            print("Generating self-play data...")
            self.self_play.generate_data()
            
            # Train on batches
            print("Training on generated data...")
            self._train_on_batches()
            
            # Save checkpoint
            self._save_checkpoint(iteration)
            
            # Log metrics
            self._log_metrics(iteration)

    def _train_on_batches(self):
        self.model.train()
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        while len(self.self_play.replay_buffer) >= self.batch_size:
            # Get batch
            states, policies, values = self.self_play.get_batch(self.batch_size)
            states = states.to(self.device)
            policies = policies.to(self.device)
            values = values.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_loss(policy_pred, policies)
            value_loss = self.value_loss(value_pred, values)
            total_loss = policy_loss + value_loss
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        # Calculate average losses
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        return avg_policy_loss, avg_value_loss

    def _save_checkpoint(self, iteration: int):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer': list(self.self_play.replay_buffer)
        }
        torch.save(checkpoint, f'checkpoints/iteration_{iteration}.pt')

    def _log_metrics(self, iteration: int):
        policy_loss, value_loss = self._train_on_batches()
        
        self.writer.add_scalar('Loss/Policy', policy_loss, iteration)
        self.writer.add_scalar('Loss/Value', value_loss, iteration)
        self.writer.add_scalar('Replay Buffer Size', len(self.self_play.replay_buffer), iteration)

if __name__ == "__main__":
    trainer = AlphaZeroTrainer()
    trainer.train() 