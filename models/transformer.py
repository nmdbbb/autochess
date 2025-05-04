import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        return self.transformer(x, mask)

class ChessTransformer(nn.Module):
    def __init__(self, d_model=256, num_layers=6, num_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        # Input dimensions
        self.input_channels = 119  # AlphaZero-style input planes
        self.board_size = 8
        self.d_model = d_model
        
        # Patch embedding (each square is a patch)
        self.patch_embedding = nn.Linear(self.input_channels, d_model)
        
        # Learnable position embeddings (8x8 board = 64 positions)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model))
        
        # Layer normalization before transformer
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4672)  # Full chess move space
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, 119, 8, 8]
        batch_size = x.size(0)
        
        # Reshape input to [batch_size, 64, 119]
        # Each square becomes a token with 119 channels
        x = x.permute(0, 2, 3, 1)  # [batch_size, 8, 8, 119]
        x = x.reshape(batch_size, 64, self.input_channels)
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch_size, 64, d_model]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Layer normalization and dropout
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Transformer encoder
        x = self.transformer(x)  # [batch_size, 64, d_model]
        
        # Global average pooling over squares for value head
        x_pooled = x.mean(dim=1)  # [batch_size, d_model]
        
        # Value and policy predictions
        value = self.value_head(x_pooled)  # [batch_size, 1]
        value = value.squeeze(-1)  # [batch_size]
        policy_logits = self.policy_head(x_pooled)  # [batch_size, 4672]
        
        return policy_logits, value

def loss_function(policy_pred, policy_target, value_pred, value_target, model, weight_decay=1e-4):
    """
    Compute the combined loss for policy and value predictions
    
    Args:
        policy_pred: predicted move logits [batch_size, 4672]
        policy_target: target move probabilities [batch_size, 4672]
        value_pred: predicted game outcome [batch_size]
        value_target: target game outcome [batch_size]
        model: the transformer model (for L2 regularization)
        weight_decay: L2 regularization coefficient
    """
    # MSE loss for value head
    value_loss = F.mse_loss(value_pred, value_target)
    
    # Cross entropy loss for policy head
    # Use KL divergence since we have probability distributions
    policy_loss = F.kl_div(
        F.log_softmax(policy_pred, dim=-1),
        policy_target,
        reduction='batchmean'
    )
    
    # L2 regularization
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param) ** 2
    l2_loss *= weight_decay
    
    # Total loss
    total_loss = value_loss + policy_loss + l2_loss
    
    return total_loss, {
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item(),
        'l2_loss': l2_loss.item(),
        'total_loss': total_loss.item()
    } 