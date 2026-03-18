"""UNet model: same interface as CNN(Cin, Cout, base, Nx, nx)."""
import torch
import torch.nn as nn


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


def _make_pool(pooling, ch):
    if pooling == "max":
        return nn.MaxPool2d(2)
    elif pooling == "avg":
        return nn.AvgPool2d(2)
    elif pooling == "stride":
        return nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    else:
        raise ValueError(f"Unknown pooling: {pooling!r}")


class UNet(nn.Module):
    def __init__(self, Cin, Cout, base, Nx, nx, pooling="max"):
        super().__init__()
        self._crop_offset = (Nx - nx) // 2
        self._nx = nx

        self.enc0 = _double_conv(Cin, base)
        self.pool0 = _make_pool(pooling, base)
        self.enc1 = _double_conv(base, base * 2)
        self.pool1 = _make_pool(pooling, base * 2)

        self.bottleneck = _double_conv(base * 2, base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec1 = _double_conv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec2 = _double_conv(base * 2, base)

        self.out_conv = nn.Conv2d(base, Cout, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        b = self.bottleneck(self.pool1(e1))

        d1 = self.dec1(torch.cat([self.up1(b), e1], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e0], dim=1))

        o = self._crop_offset
        n = self._nx
        d2 = d2[:, :, o:o + n, o:o + n]
        return self.out_conv(d2)
