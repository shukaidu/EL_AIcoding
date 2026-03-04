import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch, ch, 5, 1, 0, bias=True)

    def forward(self, x):
        # pre-activation version: activate before each conv
        y = self.relu(x)
        y = self.conv1(y)

        s = x[:, :, 2:-2, 2:-2]
        return y + s
        #return y

class ShrinkCNN(nn.Module):
    """
    - First block is a plain shrinking layer (no residual, no projection).
    - Then L-1 residual blocks with fixed channels.
    - Each block shrinks H,W by 4. L = (Nx - nx)//4.
    """
    def __init__(self, Cin=3, Cout=3, base=32, Nx=64, nx=32):
        super().__init__()
        L = (Nx - nx) // 4
        print("number layers: "+str(L))
        assert 4 * L == (Nx - nx) and L >= 1, "Require (Nx - nx) divisible by 4 and ≥ 4."

        layers = [nn.Conv2d(Cin,  base, 5, 1, 0, bias=True)]
        for _ in range(L - 1):                     # all residual, fixed channels
            layers.append(ResBlock(base))
        layers.append(nn.Conv2d(base, Cout, 1, 1, 0, bias=True))  # head
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def tv_isotropic_per_channel(u, eps=1e-6):
    """
    u: (N, C, H, W). TV per channel.
    Uses forward diffs (no padding); ignores last row/col.
    Returns: tensor of shape (C,)
    """
    dx = u[:, :, 1:, :] - u[:, :, :-1, :]   # (N, C, nx-1, ny)
    dy = u[:, :, :, 1:] - u[:, :, :, :-1]   # (N, C, nx, ny-1)

    #tvdx = (dx**2 + eps).sqrt()            # (N, C, nx-1,   ny)
    #tvdy = (dy**2 + eps).sqrt()            # (N, C, nx, ny-1)
    tvdx = (dx**2 + eps)                    # (N, C, nx-1,   ny)
    tvdy = (dy**2 + eps)                    # (N, C, nx, ny-1)

    # mean over N and spatial dims → per-channel TV
    tv_c = tvdx.mean(dim=(0, 2, 3)) + tvdy.mean(dim=(0, 2, 3))  # (C,)
    return tv_c
