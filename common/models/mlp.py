"""MLP with ReLU activations."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=5):
        super().__init__()
        layers = []
        d = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), nn.ReLU()]
            d = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
