"""Load .mat and build train/test DataLoaders."""
import torch
import scipy.io
from sklearn.model_selection import train_test_split


def _loaders(X, Y, device, b_size, test_split=0.2):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=test_split, random_state=42)
    tr_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32).to(device),
        torch.tensor(Ytr, dtype=torch.float32).to(device),
    )
    te_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xte, dtype=torch.float32).to(device),
        torch.tensor(Yte, dtype=torch.float32).to(device),
    )
    return (
        torch.utils.data.DataLoader(tr_ds, batch_size=b_size, shuffle=True),
        torch.utils.data.DataLoader(te_ds, batch_size=b_size, shuffle=False),
    )


def load_mat(path, device, b_size=100, test_split=0.2):
    """Load flat (features x samples) .mat — used by burgers_1d and wave_2d_linear."""
    d = scipy.io.loadmat(path)
    xi = torch.tensor(d["input_tensor"], dtype=torch.float32).t()
    xo = torch.tensor(d["output_tensor"], dtype=torch.float32).t()
    tl, vl = _loaders(xi.numpy(), xo.numpy(), device, b_size, test_split)
    return tl, vl, xi.size(1), xo.size(1), xi.size(0)


def load_wave_2d_nonlinear(path, device, b_size=100, test_split=0.2, residual=False):
    import numpy as np
    d = scipy.io.loadmat(path)
    xi, xo = d["input_tensor"], d["output_tensor"]

    # Per-channel stats from training split only (avoid leakage)
    idx_tr, _ = train_test_split(np.arange(len(xi)), test_size=test_split, random_state=42)
    ch_mean = xi[idx_tr].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)  # (1,C,1,1)
    ch_std = xi[idx_tr].std(axis=(0, 2, 3), keepdims=True).clip(1e-8).astype(np.float32)

    xi_norm = (xi - ch_mean) / ch_std
    xo_norm = (xo - ch_mean) / ch_std
    if residual:
        # crop xi center to match xo spatial size, then compute delta
        _, _, Nx, _ = xi.shape
        _, _, nx, _ = xo.shape
        o = (Nx - nx) // 2
        xi_center_norm = xi_norm[:, :, o:o + nx, o:o + nx]
        target_norm = xo_norm - xi_center_norm
    else:
        target_norm = xo_norm

    tl, vl = _loaders(xi_norm, target_norm, device, b_size, test_split)
    _, C_in, Nx, Ny = xi.shape
    _, C_out, nx, ny = xo.shape
    stats = {"ch_mean": ch_mean.squeeze(), "ch_std": ch_std.squeeze()}
    return tl, vl, xi.shape[0], C_in, C_out, Nx, Ny, nx, ny, stats
