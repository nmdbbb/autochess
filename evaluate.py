import chess
import chess.engine
import torch
import yaml
import os
from typing import List, Tuple, Dict
from models.transformer import ChessTransformer
from mcts.mcts import MCTS
import time
import statistics
from tqdm import tqdm

def load_model(checkpoint_path: str) -> ChessTransformer:
    """Load the model from a checkpoint"""
    model = ChessTransformer()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def play_game(model: ChessTransformer, mcts: MCTS, engine: chess.engine.SimpleEngine, 
             model_plays_white: bool) -> Tuple[str, int, List[float]]:
    """Play a game between the model and Stockfish"""
    board = chess.Board()
    move_times = []
    
    while not board.is_game_over():
        start_time = time.time()
        
        if (board.turn == chess.WHITE and model_plays_white) or \
           (board.turn == chess.BLACK and not model_plays_white):
            # Model's turn
            move = mcts.search(board)
        else:
            # Stockfish's turn
            result = engine.play(board, chess.engine.Limit(time=1.0))
            move = result.move
        
        move_time = time.time() - start_time
        move_times.append(move_time)
        
        board.push(move)
    
    result = board.result()
    return result, len(board.move_stack), move_times

def evaluate_model(config_path: str, num_games: int = 100, 
                  stockfish_path: str = "stockfish") -> Dict:
    """Evaluate the model against Stockfish"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the latest checkpoint
    try:
        checkpoint_files = [f for f in os.listdir('checkpoints') if f.startswith('iteration_')]
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        model = load_model(f'checkpoints/{latest_checkpoint}')
        print(f"Loaded checkpoint: {latest_checkpoint}")
    except:
        print("No checkpoints found. Starting with untrained model.")
        model = ChessTransformer()
    
    # Initialize MCTS
    mcts = MCTS(
        model,
        num_simulations=config['mcts']['num_simulations'],
        c_puct=config['mcts']['c_puct']
    )
    
    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    # Play games
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'move_counts': [],
        'move_times': []
    }
    
    for i in tqdm(range(num_games), desc="Playing games"):
        # Alternate colors
        model_plays_white = i % 2 == 0
        
        # Play game
        result, move_count, move_times = play_game(model, mcts, engine, model_plays_white)
        
        # Update results
        if result == "1-0":
            if model_plays_white:
                results['wins'] += 1
            else:
                results['losses'] += 1
        elif result == "0-1":
            if model_plays_white:
                results['losses'] += 1
            else:
                results['wins'] += 1
        else:
            results['draws'] += 1
        
        results['move_counts'].append(move_count)
        results['move_times'].extend(move_times)
    
    # Calculate statistics
    stats = {
        'win_rate': results['wins'] / num_games,
        'loss_rate': results['losses'] / num_games,
        'draw_rate': results['draws'] / num_games,
        'avg_moves_per_game': statistics.mean(results['move_counts']),
        'avg_move_time': statistics.mean(results['move_times']),
        'std_move_time': statistics.stdev(results['move_times']) if len(results['move_times']) > 1 else 0
    }
    
    # Clean up
    engine.quit()
    
    return stats

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero Chess model against Stockfish')
    parser.add_argument('--config', default='config/training_config.yaml',
                      help='Path to training config file')
    parser.add_argument('--num-games', type=int, default=100,
                      help='Number of games to play')
    parser.add_argument('--stockfish', default='stockfish',
                      help='Path to Stockfish executable')
    args = parser.parse_args()
    
    stats = evaluate_model(args.config, args.num_games, args.stockfish)
    
    print("\nEvaluation Results:")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Loss Rate: {stats['loss_rate']:.2%}")
    print(f"Draw Rate: {stats['draw_rate']:.2%}")
    print(f"Average Moves per Game: {stats['avg_moves_per_game']:.1f}")
    print(f"Average Move Time: {stats['avg_move_time']:.3f}s")
    print(f"Move Time Standard Deviation: {stats['std_move_time']:.3f}s")

if __name__ == "__main__":
    main() 