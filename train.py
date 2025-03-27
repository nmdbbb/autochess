from models.utils import load_model, save_model
from utils.config import load_config
from utils.logger import log_training
import torch, pickle

def train_model():
    config = load_config()
    model = load_model()

    with open('data/replay_buffer.pkl', 'rb') as f:
        data = pickle.load(f)

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    model.train()

    for epoch in range(1):  # chỉ 1 epoch mỗi iteration
        for batch in create_batches(data, config['training']['batch_size']):
            loss = model.train_step(batch, optimizer)
            log_training(loss)

    save_model(model, 'data/models/model_latest.pth')


def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]