import os
from selfplay import generate_selfplay_data
from train import train_model
from evaluate import evaluate_model
from utils.plot import plot_loss, plot_eval
from utils.config import load_config
import shutil

def train_loop(total_iterations=100):
    config = load_config()
    os.makedirs("data/models", exist_ok=True)

    for i in range(total_iterations):
        print(f"\n==== Iteration {i+1}/{total_iterations} ====")

        print("[1/4] Self-play")
        generate_selfplay_data()

        print("[2/4] Training")
        train_model()

        print("[3/4] Evaluate vs Previous")
        evaluate_model()

        print("[4/4] Save checkpoint")
        latest_model = "data/models/model_latest.pth"
        iter_model = f"data/models/model_{i+1:03d}.pth"
        shutil.copyfile(latest_model, iter_model)

    print("\n==== Training Finished ====")
    plot_loss()
    plot_eval()

if __name__ == '__main__':
    train_loop(total_iterations=50)