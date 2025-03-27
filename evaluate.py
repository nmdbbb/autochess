from models.utils import load_model, save_model
from utils.logger import log_evaluation
from chess_env.board import ChessBoard
from mcts.mcts import MCTS
from utils.config import load_config

def evaluate_model():
    config = load_config()
    new_model = load_model('data/models/model_latest.pth')
    old_model = load_model('data/models/model_best.pth')

    new_mcts = MCTS(new_model, config['mcts'])
    old_mcts = MCTS(old_model, config['mcts'])

    new_wins, old_wins, draws = 0, 0, 0
    for i in range(20):
        board = ChessBoard()
        result = play_match(new_mcts, old_mcts, board)
        if result == 1:
            new_wins += 1
        elif result == -1:
            old_wins += 1
        else:
            draws += 1

    log_evaluation(new_wins, old_wins, draws)
    if new_wins > old_wins:
        save_model(new_model, 'data/models/model_best.pth')


def play_match(mcts1, mcts2, board):
    players = [mcts1, mcts2]
    current = 0
    while not board.is_game_over():
        move = players[current].select_move(board)
        board.push(move)
        current = 1 - current
    return board.result_value()  # 1, -1, 0