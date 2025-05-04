import chess
import torch
import numpy as np
from typing import List, Tuple
from collections import deque
import random

class SelfPlay:
    def __init__(self, model, mcts, num_games: int = 100, max_moves: int = 200):
        self.model = model
        self.mcts = mcts
        self.num_games = num_games
        self.max_moves = max_moves
        self.replay_buffer = deque(maxlen=1000000)  # Store up to 1M positions

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        # Convert chess board to tensor representation
        # Output shape: [119, 8, 8]
        device = next(self.model.parameters()).device
        tensor = torch.zeros(119, 8, 8, device=device)
        
        # Piece planes (12 pieces * 8 squares = 96 planes)
        piece_idx = 0
        for piece_type in range(1, 7):  # 1=pawn, 2=knight, ..., 6=king
            for color in [True, False]:  # True=white, False=black
                for rank in range(8):
                    for file in range(8):
                        square = chess.square(file, rank)
                        piece = board.piece_at(square)
                        if piece and piece.piece_type == piece_type and piece.color == color:
                            tensor[piece_idx, rank, file] = 1
                piece_idx += 1
        
        # Additional features (23 planes)
        # Repetition counters (2 planes)
        tensor[96:98] = 0  # We don't track repetitions yet
        
        # Color (1 plane)
        tensor[98].fill_(1 if board.turn else 0)
        
        # Total move count (1 plane)
        tensor[99].fill_(board.fullmove_number)
        
        # Castling rights (4 planes)
        tensor[100].fill_(board.has_kingside_castling_rights(chess.WHITE))
        tensor[101].fill_(board.has_queenside_castling_rights(chess.WHITE))
        tensor[102].fill_(board.has_kingside_castling_rights(chess.BLACK))
        tensor[103].fill_(board.has_queenside_castling_rights(chess.BLACK))
        
        # No-progress count (1 plane)
        tensor[104].fill_(board.halfmove_clock)
        
        # Remaining planes (14) set to 0 for now
        # These could be used for additional features in the future
        
        return tensor

    def _get_training_data(self, board: chess.Board, move: chess.Move) -> Tuple[torch.Tensor, np.ndarray, float]:
        # Get state representation
        state = self._board_to_tensor(board)
        
        # Get policy and value from MCTS
        encoded_state = state.unsqueeze(0)  # Add batch dimension
        move = self.mcts.search(board, encoded_state)
        
        # Create policy vector
        policy = np.zeros(4672)  # All possible chess moves
        total_visits = sum(child.visit_count for child in self.mcts.root.children.values())
        
        for child_move, child in self.mcts.root.children.items():
            move_idx = self.mcts._encode_move(child_move)
            policy[move_idx] = child.visit_count / total_visits
        
        # Get value from root node
        value = self.mcts.root.value()
        
        return state, policy, value

    def play_game(self) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        board = chess.Board()
        game_data = []
        
        for _ in range(self.max_moves):
            if board.is_game_over():
                break
            
            # Get state representation and move from MCTS
            state = self._board_to_tensor(board)
            encoded_state = state.unsqueeze(0)  # Add batch dimension
            move = self.mcts.search(board, encoded_state)
            
            if move is None:  # Game is over
                break
                
            # Get training data before making the move
            state, policy, value = self._get_training_data(board, move)
            game_data.append((state, policy, value))
            
            # Make the move
            board.push(move)
        
        # Determine game outcome
        result = board.result()
        if result == "1-0":
            final_value = 1.0
        elif result == "0-1":
            final_value = -1.0
        else:
            final_value = 0.0
        
        # Update values based on game outcome
        updated_data = []
        for i, (state, policy, _) in enumerate(game_data):
            # Alternate perspective
            value = final_value * (-1) ** (i % 2)
            updated_data.append((state, policy, value))
        
        return updated_data

    def generate_data(self) -> None:
        for game_idx in range(self.num_games):
            print(f"Playing game {game_idx + 1}/{self.num_games}")
            game_data = self.play_game()
            self.replay_buffer.extend(game_data)
            
            # Save checkpoint every 10 games
            if (game_idx + 1) % 10 == 0:
                self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        # Save model checkpoint
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'replay_buffer': list(self.replay_buffer)
        }, f'checkpoints/checkpoint_{len(self.replay_buffer)}.pt')

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample random batch from replay buffer
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        
        states = torch.stack([x[0] for x in batch])
        policies = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        values = torch.tensor([x[2] for x in batch], dtype=torch.float32).unsqueeze(1)
        
        return states, policies, values 