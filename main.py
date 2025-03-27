from selfplay import generate_selfplay_data
from train import train_model
from evaluate import evaluate_model

def main():
    print("[1/3] Running self-play...")
    generate_selfplay_data()

    print("[2/3] Training model...")
    train_model()

    print("[3/3] Evaluating model...")
    evaluate_model()

if __name__ == '__main__':
    main()