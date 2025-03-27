import matplotlib.pyplot as plt
import json, os

loss_log_file = "data/loss_log.json"
eval_log_file = "data/eval_log.json"

def save_loss(epoch, loss):
    os.makedirs("data", exist_ok=True)
    data = []
    if os.path.exists(loss_log_file):
        with open(loss_log_file, 'r') as f:
            data = json.load(f)
    data.append({"epoch": epoch, "loss": loss})
    with open(loss_log_file, 'w') as f:
        json.dump(data, f)

def save_eval(iteration, new, old, draw):
    os.makedirs("data", exist_ok=True)
    data = []
    if os.path.exists(eval_log_file):
        with open(eval_log_file, 'r') as f:
            data = json.load(f)
    data.append({"iter": iteration, "new": new, "old": old, "draw": draw})
    with open(eval_log_file, 'w') as f:
        json.dump(data, f)

def plot_loss():
    if not os.path.exists(loss_log_file): return
    with open(loss_log_file, 'r') as f:
        data = json.load(f)
    xs = [d['epoch'] for d in data]
    ys = [d['loss'] for d in data]
    plt.plot(xs, ys, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_eval():
    if not os.path.exists(eval_log_file): return
    with open(eval_log_file, 'r') as f:
        data = json.load(f)
    xs = [d['iter'] for d in data]
    new = [d['new'] for d in data]
    old = [d['old'] for d in data]
    draw = [d['draw'] for d in data]
    plt.plot(xs, new, label="New Wins")
    plt.plot(xs, old, label="Old Wins")
    plt.plot(xs, draw, label="Draws")
    plt.xlabel("Iteration")
    plt.ylabel("Games")
    plt.title("Evaluation vs Old Model")
    plt.legend()
    plt.grid()
    plt.show()
