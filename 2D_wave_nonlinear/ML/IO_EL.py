import torch
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split


def mat2pyt(filename, device, b_size=100, test_split=0.2):
    """Load .mat (input_tensor, output_tensor), split train/test, return loaders and dimensions."""
    data = scipy.io.loadmat(filename)
    data_in = data["input_tensor"]
    data_out = data["output_tensor"]
    N_samp = data_in.shape[0]
    C_in = data_in.shape[1]
    Nx = data_in.shape[2]
    Ny = data_in.shape[3]
    C_out = data_out.shape[1]
    nx = data_out.shape[2]
    ny = data_out.shape[3]

    X_train, X_test, Y_train, Y_test = train_test_split(
        data_in, data_out, test_size=test_split, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)

    train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
    test_ds = torch.utils.data.TensorDataset(X_test, Y_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=b_size, shuffle=False)

    return train_loader, test_loader, N_samp, C_in, C_out, Nx, Ny, nx, ny
