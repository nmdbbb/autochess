import torch
from models.resnet import AlphaZeroResNet
from utils.config import load_config

def load_model(path=None):
    cfg = load_config()['model']
    model = AlphaZeroResNet(cfg['input_planes'], cfg['channels'], cfg['residual_blocks'])
    if path:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)

def loss_fn(policy_pred, value_pred, policy_target, value_target):
    value_loss = F.mse_loss(value_pred.squeeze(), value_target)
    policy_loss = -torch.mean(torch.sum(policy_target * torch.log_softmax(policy_pred, dim=1), dim=1))
    return value_loss + policy_loss
