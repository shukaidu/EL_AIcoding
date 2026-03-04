import torch
import torch.nn as nn
import os
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np


class myNN(nn.Module):
    """Linear NN (Identity activation). Configurable width/depth for better accuracy."""
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=5):
        super(myNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Identity())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Identity())
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def mat2pyt(filename, device, b_size=100, test_split=0.2):
    data = scipy.io.loadmat(filename)
    data_in = data["input_tensor"]
    data_out = data["output_tensor"]
    data_in = torch.tensor(data_in, dtype=torch.float32).t()
    data_out = torch.tensor(data_out, dtype=torch.float32).t()
    N_i = data_in.size(dim=1)
    N_o = data_out.size(dim=1)
    N_samp = data_in.size(dim=0)
    data_in_train, data_in_test, data_out_train, data_out_test = train_test_split(
        data_in.cpu().numpy(), data_out.cpu().numpy(),
        test_size=test_split, random_state=42,
    )
    data_in_train = torch.tensor(data_in_train, dtype=torch.float32).to(device)
    data_out_train = torch.tensor(data_out_train, dtype=torch.float32).to(device)
    data_in_test = torch.tensor(data_in_test, dtype=torch.float32).to(device)
    data_out_test = torch.tensor(data_out_test, dtype=torch.float32).to(device)
    train_dataset = torch.utils.data.TensorDataset(data_in_train, data_out_train)
    test_dataset = torch.utils.data.TensorDataset(data_in_test, data_out_test)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size, shuffle=False)
    return train_dataloader, test_dataloader, N_i, N_o, N_samp
