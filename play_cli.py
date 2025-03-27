from models.utils import load_model
from chess_env.board import ChessBoard
from mcts.mcts import MCTS
from utils.config import load_config

def play_cli():
    config = load_config()
    model = load_model('data/models/model_best.pth')
    mcts = MCTS(model, config['mcts'])

    board = ChessBoard()
    while not board.is_game_over():
        print(board)
        if board.turn == 0:
            move = input("Your move (in UCI, e.g. e2e4): ")
            board.push_uci(move)
        else:
            move = mcts.select_move(board)
            board.push(move)

    print("Game over. Result:", board.result())

if __name__ == '__main__':
    play_cli()
