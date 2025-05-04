import math
import numpy as np
import chess
from typing import Dict, List, Optional
import torch

class Node:
    def __init__(self, state: chess.Board, parent: Optional['Node'] = None, action: Optional[chess.Move] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[chess.Move, 'Node'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

class MCTS:
    def __init__(self, model, num_simulations: int = 800, c_puct: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def _board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        # Convert chess board to tensor representation
        # [8, 8, 13] tensor: 12 pieces + 1 color
        tensor = torch.zeros(8, 8, 13)
        
        for i in range(8):
            for j in range(8):
                square = chess.square(i, j)
                piece = board.piece_at(square)
                if piece:
                    # Piece type (0-5) and color (0-1)
                    piece_type = piece.piece_type - 1
                    color = 1 if piece.color else 0
                    tensor[i, j, piece_type + (color * 6)] = 1
        
        # Add color to move
        tensor[:, :, 12] = 1 if board.turn else 0
        return tensor

    def _select(self, node: Node) -> Node:
        while node.children:
            # Select child with highest UCB score
            best_score = float('-inf')
            best_move = None
            
            for move, child in node.children.items():
                # UCB formula
                score = child.value_sum / (child.visit_count + 1e-8) + \
                       self.c_puct * child.prior * math.sqrt(node.visit_count) / (child.visit_count + 1)
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            node = node.children[best_move]
        return node

    def _expand(self, node: Node) -> None:
        # Get policy and value predictions from model
        board_tensor = self._board_to_tensor(node.state).unsqueeze(0)
        with torch.no_grad():
            policy, value = self.model(board_tensor)
        
        # Store value prediction
        node.value_sum = value.item()
        
        # Expand all legal moves
        for move in node.state.legal_moves:
            child_state = node.state.copy()
            child_state.push(move)
            child = Node(child_state, parent=node, action=move)
            child.prior = policy[0, move.from_square * 64 + move.to_square].item()
            node.children[move] = child

    def _backup(self, node: Node, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Alternate perspective
            node = node.parent

    def search(self, board: chess.Board) -> chess.Move:
        root = Node(board)
        
        for _ in range(self.num_simulations):
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.state.is_game_over():
                self._expand(node)
            
            # Simulation
            value = 0.0
            if node.state.is_game_over():
                result = node.state.result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
            
            # Backpropagation
            self._backup(node, value)
        
        # Select best move based on visit counts
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_move 