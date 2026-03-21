"""MLP with configurable activation."""
import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "linear": nn.Identity,
    "identity": nn.Identity,
    "none": nn.Identity,
}


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, activation):
        super().__init__()
        act = _ACTIVATIONS[activation]
        layers = []
        d = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), act()]
            d = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
