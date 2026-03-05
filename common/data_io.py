"""Load .mat data and build DataLoaders for each problem type."""
import torch
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split


def _make_loaders(X, Y, device, b_size, test_split=0.2):
    """Split X,Y with train_test_split, build TensorDataset + DataLoader. X,Y: numpy (N, ...)."""
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=test_split, random_state=42
    )
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32).to(device),
        torch.tensor(Y_tr, dtype=torch.float32).to(device),
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_te, dtype=torch.float32).to(device),
        torch.tensor(Y_te, dtype=torch.float32).to(device),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=b_size, shuffle=False)
    return train_loader, test_loader


def _load_mat_flat(filename, device, b_size=100, test_split=0.2):
    """Load .mat with input_tensor (d_in, N), output_tensor (d_out, N); return loaders, N_i, N_o, N_samp."""
    data = scipy.io.loadmat(filename)
    data_in = torch.tensor(data["input_tensor"], dtype=torch.float32).t()
    data_out = torch.tensor(data["output_tensor"], dtype=torch.float32).t()
    train_loader, test_loader = _make_loaders(
        data_in.numpy(), data_out.numpy(), device, b_size, test_split
    )
    N_i, N_o = data_in.size(1), data_out.size(1)
    return train_loader, test_loader, N_i, N_o, data_in.size(0)


def load_burgers_1d(filename, device, b_size=100, test_split=0.2):
    """input_tensor (N_in, N), output_tensor (N_out, N). Returns loaders, N_i, N_o, N_samp."""
    return _load_mat_flat(filename, device, b_size, test_split)


def load_wave_2d_linear(filename, device, b_size=100, test_split=0.2):
    """input_tensor (2*patch^2, N), output_tensor (2*nwd^2, N). Returns loaders, N_i, N_o, N_samp."""
    return _load_mat_flat(filename, device, b_size, test_split)


def load_wave_2d_nonlinear(filename, device, b_size=100, test_split=0.2):
    """input_tensor (N, C, H, W), output_tensor (N, C, h, w). Returns loaders and dims."""
    data = scipy.io.loadmat(filename)
    data_in = data["input_tensor"]
    data_out = data["output_tensor"]
    N_samp = data_in.shape[0]
    C_in, Nx, Ny = data_in.shape[1], data_in.shape[2], data_in.shape[3]
    C_out, nx, ny = data_out.shape[1], data_out.shape[2], data_out.shape[3]
    train_loader, test_loader = _make_loaders(
        data_in, data_out, device, b_size, test_split
    )
    return train_loader, test_loader, N_samp, C_in, C_out, Nx, Ny, nx, ny
