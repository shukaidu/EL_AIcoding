"""ShrinkCNN: patch in -> patch out, each block shrinks H,W by 4. L = (Nx - nx)//4."""
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(ch, ch, 5, 1, 0, bias=True)

    def forward(self, x):
        y = self.conv(self.relu(x))
        return y + x[:, :, 2:-2, 2:-2]


class ShrinkCNN(nn.Module):
    def __init__(self, Cin=3, Cout=3, base=32, Nx=64, nx=32):
        super().__init__()
        L = (Nx - nx) // 4
        assert 4 * L == Nx - nx and L >= 1
        layers = [nn.Conv2d(Cin, base, 5, 1, 0, bias=True)]
        for _ in range(L - 1):
            layers.append(ResBlock(base))
        layers.append(nn.Conv2d(base, Cout, 1, 1, 0, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
