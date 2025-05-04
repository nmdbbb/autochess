import ray
import yaml
import torch
import numpy as np
from typing import Dict, List, Tuple
from models.transformer import ChessTransformer
from mcts.mcts import MCTS
import chess
import os

@ray.remote(num_gpus=1)
class SelfPlayWorker:
    def __init__(self, config: Dict):
        self.config = config
        self.model = ChessTransformer(
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout']
        )
        self.mcts = MCTS(
            self.model,
            num_simulations=config['mcts']['num_simulations'],
            c_puct=config['mcts']['c_puct']
        )
        self.temperature = config['self_play']['temperature']

    def play_game(self) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        board = chess.Board()
        game_data = []
        
        for _ in range(self.config['self_play']['max_moves']):
            if board.is_game_over():
                break
            
            # Get move from MCTS with temperature
            move = self.mcts.search(board)
            
            # Get training data
            state = self._board_to_tensor(board)
            policy = self._get_policy(board)
            value = self._get_value(state)
            
            game_data.append((state, policy, value))
            board.push(move)
        
        # Update values based on game outcome
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
        
        return [(s, p, final_value * (-1) ** (i % 2)) 
                for i, (s, p, _) in enumerate(game_data)]

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(8, 8, 13)
        for i in range(8):
            for j in range(8):
                square = chess.square(i, j)
                piece = board.piece_at(square)
                if piece:
                    piece_type = piece.piece_type - 1
                    color = 1 if piece.color else 0
                    tensor[i, j, piece_type + (color * 6)] = 1
        tensor[:, :, 12] = 1 if board.turn else 0
        return tensor

    def _get_policy(self, board: chess.Board) -> np.ndarray:
        root = self.mcts._select(self.mcts.Node(board))
        self.mcts._expand(root)
        
        policy = np.zeros(4672)
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for child_move, child in root.children.items():
            move_idx = child_move.from_square * 64 + child_move.to_square
            policy[move_idx] = child.visit_count / total_visits
        
        return policy

    def _get_value(self, state: torch.Tensor) -> float:
        with torch.no_grad():
            _, value = self.model(state.unsqueeze(0))
            return value.item()

@ray.remote(num_gpus=1)
class TrainingWorker:
    def __init__(self, config: Dict):
        self.config = config
        self.model = ChessTransformer(
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout']
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

    def train_batch(self, batch: List[Tuple[torch.Tensor, np.ndarray, float]]) -> Dict[str, float]:
        states = torch.stack([x[0] for x in batch])
        policies = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        values = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1)
        
        self.optimizer.zero_grad()
        policy_pred, value_pred = self.model(states)
        
        policy_loss = self.policy_loss(policy_pred, policies)
        value_loss = self.value_loss(value_pred, values)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['gradient_clip']
        )
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

def main():
    # Load configuration
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Ray
    ray.init(
        address=config['distributed']['ray']['redis_address'],
        _redis_password=config['distributed']['ray']['redis_password']
    )
    
    # Create workers
    num_workers = config['distributed']['num_workers']
    self_play_workers = [SelfPlayWorker.remote(config) for _ in range(num_workers)]
    training_worker = TrainingWorker.remote(config)
    
    # Training loop
    for iteration in range(config['training']['num_iterations']):
        print(f"\nIteration {iteration + 1}/{config['training']['num_iterations']}")
        
        # Generate self-play data
        print("Generating self-play data...")
        game_data = ray.get([
            worker.play_game.remote()
            for worker in self_play_workers
        ])
        
        # Flatten game data
        all_data = [item for game in game_data for item in game]
        
        # Train on batches
        print("Training on generated data...")
        batch_size = config['training']['batch_size']
        num_batches = len(all_data) // batch_size
        
        for batch_idx in range(num_batches):
            batch = all_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            metrics = ray.get(training_worker.train_batch.remote(batch))
            
            if batch_idx % config['logging']['log_interval'] == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}")
                print(f"Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"Value Loss: {metrics['value_loss']:.4f}")
                print(f"Total Loss: {metrics['total_loss']:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % config['checkpoint']['save_interval'] == 0:
            model_state = ray.get(training_worker.model.state_dict.remote())
            torch.save({
                'iteration': iteration,
                'model_state_dict': model_state,
                'config': config
            }, f"{config['checkpoint']['checkpoint_dir']}/iteration_{iteration}.pt")

if __name__ == "__main__":
    main() 