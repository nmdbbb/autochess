import numpy as np
import math
import copy
import chess

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, config):
        self.model = model.eval()
        self.simulations = config['simulations']
        self.cpuct = config['cpuct']
        self.temperature = config['temperature']
        self.dirichlet_alpha = config['dirichlet_alpha']

    def self_play(self, board, max_moves):
        data = []
        for _ in range(max_moves):
            if board.is_game_over():
                break
            root = self.run_mcts(board)
            pi = self._get_policy(root)
            move_idx = self._sample_move(pi)
            move = self._index_to_move(move_idx, board)
            data.append((board.encode(), pi, None))
            board.push(move)

        result = board.result_value()
        for i in range(len(data)):
            state, pi, _ = data[i]
            data[i] = (state, pi, result)
            result = -result

        return data

    def select_move(self, board):
        root = self.run_mcts(board)
        pi = self._get_policy(root)
        move_idx = self._sample_move(pi, temperature=0)
        return self._index_to_move(move_idx, board)

    def run_mcts(self, board):
        root = MCTSNode(copy.deepcopy(board))
        self._expand_node(root)
        self._add_dirichlet_noise(root)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            while node.expanded():
                move, node = self._select_child(node)
                board = copy.deepcopy(node.board)
                search_path.append(node)

            value = self._expand_node(node)
            self._backpropagate(search_path, value)

        return root

    def _expand_node(self, node):
        board_tensor = node.board.encode()
        board_tensor = np.expand_dims(board_tensor, axis=0)
        import torch
        with torch.no_grad():
            policy_logits, value = self.model(torch.tensor(board_tensor))
        policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()

        legal_moves = node.board.legal_moves()
        for move in legal_moves:
            move_idx = self._move_to_index(move)
            node.children[move] = MCTSNode(copy.deepcopy(node.board), parent=node, prior=policy[move_idx])

        return value.item()

    def _select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1e-8)

        best_score = -float('inf')
        best_move, best_child = None, None

        for move, child in node.children.items():
            ucb = child.value() + self.cpuct * child.prior * sqrt_total / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_move, best_child = move, child

        best_child.board.push(best_move)
        return best_move, best_child

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _add_dirichlet_noise(self, node):
        legal_moves = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            node.children[move].prior = 0.75 * node.children[move].prior + 0.25 * noise[i]

    def _get_policy(self, root):
        policy = np.zeros(4672)
        for move, child in root.children.items():
            move_idx = self._move_to_index(move)
            policy[move_idx] = child.visit_count
        policy = policy ** (1 / self.temperature)
        policy /= np.sum(policy) + 1e-8
        return policy

    def _sample_move(self, policy, temperature=1):
        if temperature == 0:
            return np.argmax(policy)
        return np.random.choice(len(policy), p=policy)

    def _move_to_index(self, move):
        uci = move.uci()
        from_sq = chess.SQUARE_NAMES.index(uci[:2])
        to_sq = chess.SQUARE_NAMES.index(uci[2:4])
        promotion_offset = {None: 0, 'n': 0, 'b': 1, 'r': 2, 'q': 3}
        prom = promotion_offset.get(uci[4:] if len(uci) > 4 else None, 0)
        return 64 * from_sq + to_sq + prom * 4096

    def _index_to_move(self, index, board):
        for move in board.legal_moves():
            if self._move_to_index(move) == index:
                return move
        return list(board.legal_moves())[0]  # fallback
