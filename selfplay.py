from chess_env.board import ChessBoard
from mcts.mcts import MCTS
from models.utils import load_model
from utils.config import load_config
import pickle, os

def generate_selfplay_data():
    config = load_config()
    model = load_model()
    mcts = MCTS(model, config['mcts'])

    buffer = []
    for _ in range(config['selfplay']['games_per_iteration']):
        board = ChessBoard()
        game_data = mcts.self_play(board, config['selfplay']['max_moves'])
        buffer.extend(game_data)

    os.makedirs('data', exist_ok=True)
    with open('data/replay_buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)
