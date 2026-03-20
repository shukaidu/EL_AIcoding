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


def _make_activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported MLP activation: {name}")
    return _ACTIVATIONS[key]()


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=5, activation="relu"):
        super().__init__()
        layers = []
        d = input_size
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_size), _make_activation(activation)]
            d = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
