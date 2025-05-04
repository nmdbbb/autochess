import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

class ChessTransformer(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12, num_heads=12, dropout=0.1):
        super().__init__()
        
        # Chess board representation: 8x8x12 (pieces) + 1 (color) = 769
        self.input_embedding = nn.Linear(769, hidden_size)
        
        # Transformer configuration
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=64,  # 8x8 board
        )
        
        self.transformer = BertModel(config)
        
        # Policy head: predicts move probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4672)  # All possible chess moves
        )
        
        # Value head: predicts game outcome
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, board_state):
        # board_state: [batch_size, 8, 8, 13]
        batch_size = board_state.size(0)
        
        # Flatten the board state
        x = board_state.view(batch_size, -1)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add position embeddings
        position_ids = torch.arange(64, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x.view(batch_size, 64, -1)  # Reshape for transformer
        
        # Transformer
        transformer_output = self.transformer(
            inputs_embeds=x,
            position_ids=position_ids,
            return_dict=True
        )
        
        # Get the [CLS] token output for value prediction
        cls_output = transformer_output.last_hidden_state[:, 0]
        
        # Policy and value predictions
        policy = self.policy_head(transformer_output.last_hidden_state.view(batch_size, -1))
        value = self.value_head(cls_output)
        
        return policy, value 