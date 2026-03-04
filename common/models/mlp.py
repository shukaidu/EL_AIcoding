"""MLP for 1D Burgers (ReLU) and 2D linear wave (Identity)."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Configurable MLP: activation 'relu' (1D) or 'identity' (2D linear)."""

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=256,
        num_layers=5,
        activation="relu",
    ):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.Identity()
        layers = []
        in_dim = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act)
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
