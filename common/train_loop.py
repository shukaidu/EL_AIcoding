"""Generic training loop and training history plot."""
import sys
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_training_history(train_loss_history, test_loss_history, save_path):
    """Save training_history.png (log scale)."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss_history, label="Train loss", linewidth=0.8)
    plt.plot(epochs, test_loss_history, label="Test loss", linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and testing error history")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
