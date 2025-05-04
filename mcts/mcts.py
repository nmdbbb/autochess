import math
import chess
import torch
import numpy as np
from typing import Dict, Tuple

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[chess.Move, 'Node'] = {}
        self.outcome = None
    
    def value(self) -> float:
        """Get mean value of node"""
        if self.outcome is not None:
            return self.outcome
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, policy: Dict[chess.Move, float], outcome: float = None):
        """Expand node with policy predictions"""
        self.outcome = outcome
        for move, prob in policy.items():
            self.children[move] = Node(prior=prob)

class MCTS:
    def __init__(self, model, num_simulations: int, c_puct: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None  # Initialize root node
    
    def search(self, board: chess.Board, encoded_state: torch.Tensor) -> chess.Move:
        """
        Run MCTS search and return best move
        
        Args:
            board: Current chess position
            encoded_state: Encoded board state tensor of shape (1, 119, 8, 8)
        """
        # If game is over, return None and set root value
        if board.is_game_over():
            self.root = Node(prior=0)
            self.root.outcome = self._game_result_to_value(board)
            return None
        
        # Create new root node
        self.root = Node(prior=0)
        
        # Get initial policy and value
        policy, value = self._predict(encoded_state)
        policy_dict = {
            move: policy[0][self._encode_move(move)] 
            for move in board.legal_moves
        }
        
        # Normalize policy for legal moves
        policy_sum = sum(policy_dict.values())
        if policy_sum > 0:
            policy_dict = {
                move: prob / policy_sum 
                for move, prob in policy_dict.items()
            }
        else:
            # If all legal moves have zero probability, use uniform distribution
            policy_dict = {
                move: 1.0 / len(policy_dict) 
                for move in policy_dict.keys()
            }
        
        # Expand root node
        outcome = None if not board.is_game_over() else self._game_result_to_value(board)
        self.root.expand(policy_dict, outcome)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = self.root
            sim_board = board.copy()
            search_path = [node]
            
            # Selection
            while node.children and not sim_board.is_game_over():
                action, node = self._select_child(node, sim_board)
                sim_board.push(action)
                search_path.append(node)
            
            # Expansion and evaluation
            value = self._game_result_to_value(sim_board) if sim_board.is_game_over() else None
            
            if value is None:
                # Get policy and value prediction
                encoded = encoded_state  # TODO: Update encoded state for new position
                policy, value = self._predict(encoded)
                
                # Create policy dictionary for legal moves
                policy_dict = {
                    move: policy[0][self._encode_move(move)] 
                    for move in sim_board.legal_moves
                }
                
                # Normalize policy
                policy_sum = sum(policy_dict.values())
                if policy_sum > 0:
                    policy_dict = {
                        move: prob / policy_sum 
                        for move, prob in policy_dict.items()
                    }
                else:
                    policy_dict = {
                        move: 1.0 / len(policy_dict) 
                        for move in policy_dict.keys()
                    }
                
                node.expand(policy_dict)
            
            # Backpropagate
            self._backpropagate(search_path, value, sim_board.turn)
        
        # If no children (game over), return None
        if not self.root.children:
            return None
        
        # Select move with highest visit count
        return max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]
    
    def _predict(self, encoded_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value predictions from model"""
        with torch.no_grad():
            policy, value = self.model(encoded_state)
        return policy, value
    
    def _select_child(self, node: Node, board: chess.Board) -> Tuple[chess.Move, Node]:
        """Select child node using PUCT algorithm"""
        if node.outcome is not None:
            # Game is over, return any child (shouldn't matter)
            return next(iter(node.children.items()))
        
        # Calculate UCB scores
        total_visits = sum(child.visit_count for child in node.children.values())
        
        def ucb_score(move: chess.Move, child: Node) -> float:
            if child.visit_count == 0:
                q_value = 0
            else:
                q_value = -child.value()  # Negative because value is from other player's perspective
            
            return q_value + self.c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visit_count)
        
        return max(node.children.items(), key=lambda x: ucb_score(x[0], x[1]))
    
    def _backpropagate(self, search_path: list, value: float, root_turn: bool):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            if value is not None:
                if root_turn == (len(search_path) % 2 == 1):
                    node.value_sum += value
                else:
                    node.value_sum -= value
    
    def _game_result_to_value(self, board: chess.Board) -> float:
        """Convert game result to value"""
        if board.is_checkmate():
            return -1 if board.turn else 1
        return 0  # Draw
    
    def _encode_move(self, move: chess.Move) -> int:
        """Encode chess move to index in policy vector"""
        # This is a simplified encoding - you might want to use a more sophisticated one
        from_square = move.from_square
        to_square = move.to_square
        
        # Calculate base index for move
        move_idx = from_square * 64 + to_square
        
        # Add offset for promotions
        if move.promotion:
            promotion_piece = move.promotion
            if promotion_piece == chess.QUEEN:
                move_idx += 64 * 64
            elif promotion_piece == chess.ROOK:
                move_idx += 64 * 64 + 1
            elif promotion_piece == chess.BISHOP:
                move_idx += 64 * 64 + 2
            elif promotion_piece == chess.KNIGHT:
                move_idx += 64 * 64 + 3
        
        return move_idx 