
## AlphaZero Chess Implementation

A PyTorch implementation of the AlphaZero algorithm for chess, featuring a transformer-based architecture, MCTS search, and distributed training capabilities.

## Project Overview

This project implements the AlphaZero algorithm for chess, using modern deep learning techniques and a transformer-based neural network. The implementation includes:

- Transformer-based policy and value network
- Monte Carlo Tree Search (MCTS) for move selection
- Self-play training pipeline
- Distributed training support
- Web-based UI for playing against the trained model

## Project Structure

```
alphazero-chess/
├── config/               # Configuration files
│   └── training_config.yaml
├── models/              # Neural network architecture
│   └── transformer.py
├── mcts/               # Monte Carlo Tree Search implementation
│   └── mcts.py
├── selfplay/           # Self-play and game state management
│   └── game_state.py
├── utils/              # Utility functions
│   └── board_encoding.py
├── train/              # Training pipeline
├── distributed/        # Distributed training components
├── web/               # Web interface
├── backend/           # API backend
├── tests/             # Unit and integration tests
├── checkpoints/       # Model checkpoints
└── runs/              # Training logs and metrics
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- python-chess
- PyYAML
- Additional dependencies in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nmdbbb/autochess.git
cd autochess
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Configure training parameters in `config/training_config.yaml`
2. Start training:
```bash
python -m train.train
```

For distributed training:
```bash
docker-compose up
```

### Playing Against the Model

1. Start the web interface:
```bash
python -m web.app
```

2. Open a browser and navigate to `http://localhost:8000`

### Evaluating Model Strength

Run evaluation against other chess engines:
```bash
python evaluate.py --model-path checkpoints/best_model.pt
```

### Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test files:
```bash
python -m pytest tests/test_selfplay_loop.py -v
```

## Model Architecture

The implementation uses a transformer-based architecture:

- Input: 119-channel 8x8 board representation
  - Current position (48 planes)
  - Previous position (48 planes)
  - Metadata planes (repetition count, move count, etc.)
- Transformer encoder with self-attention
- Dual head output:
  - Policy head: 4672 move probabilities
  - Value head: Game outcome prediction [-1, 1]

## Training Pipeline

The training process follows the AlphaZero methodology:

1. Self-play data generation
   - MCTS with 800 simulations per move
   - Temperature-based exploration
   - Position and game outcome recording

2. Neural network training
   - Policy loss (cross-entropy)
   - Value loss (MSE)
   - L2 regularization

3. Model evaluation
   - Comparison with previous versions
   - ELO rating tracking

## Configuration

Key parameters in `config/training_config.yaml`:

```yaml
model:
  d_model: 256
  num_layers: 6
  nhead: 8
  dim_feedforward: 1024
  dropout: 0.1

mcts:
  simulations: 800
  cpuct: 1.0
  
training:
  batch_size: 2048
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-4
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepMind's AlphaZero papers
- The python-chess library
- The PyTorch team and community

## References

1. Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
2. Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play