"""Shared utilities: device selection and training curve plot."""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_training_history(hist_tr, hist_te, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = len(hist_tr)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, n + 1), hist_tr, label="Train", lw=0.8)
    plt.plot(range(1, n + 1), hist_te, label="Test", lw=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
