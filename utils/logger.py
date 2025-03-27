import time

def log_training(loss):
    print(f"[Train] Loss: {loss:.4f} @ {time.strftime('%H:%M:%S')}")

def log_evaluation(wins_new, wins_old, draws):
    total = wins_new + wins_old + draws
    print(f"[Eval] New: {wins_new}, Old: {wins_old}, Draws: {draws}, Win rate: {wins_new / total:.2%}")