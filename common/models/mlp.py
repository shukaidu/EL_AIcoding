"""MLP: activation 'relu' (Burgers) or 'identity' (2D linear)."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=5, activation="relu"):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.Identity()
        layers = []
        d = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), act]
            d = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
