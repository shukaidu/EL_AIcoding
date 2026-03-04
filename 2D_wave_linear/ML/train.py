"""
Train 2D wave NN. Run from 2D_wave_linear folder after gen_data.py:
  python gen_data.py
  python ML/train.py
"""
import os
import torch
from scipy.io import savemat

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_data_path = os.path.join(_project_root, "ML", "data_wave.mat")
_model_path = os.path.join(_script_dir, "data_wave_model.pth")
_error_path = os.path.join(_script_dir, "data_wave_error.mat")

from nn_model import mat2pyt, myNN
from snapshot import save_checkpoint

b_size = 128
num_epochs = 2500
hidden_size = 256
num_layers = 5
lr_schedule = [(1000, 1e-3), (2000, 1e-4), (2500, 1e-5)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
print(f"Data: {_data_path}")
print(f"NN: hidden_size={hidden_size}, num_layers={num_layers}")

dataloader_train, dataloader_test, N_i, N_o, N_samp = mat2pyt(
    _data_path, device, b_size, test_split=0.2
)

# Per-component loss weight: balance so u and v gradients are comparable (v was underfit when using raw L1)
n_out = N_o // 2  # first n_out = u, second n_out = v
train_targets = torch.cat([batch[1] for batch in dataloader_train], dim=0)
std_u = train_targets[:, :n_out].std().item() + 1e-8
std_v = train_targets[:, n_out:].std().item() + 1e-8
# Normalize so both terms contribute equally: loss = loss_u/std_u + loss_v/std_v
w_u = 1.0 / std_u
w_v = 1.0 / std_v
print(f"Output scale: std_u={std_u:.6f}, std_v={std_v:.6f} -> loss = loss_u/{std_u:.4f} + loss_v/{std_v:.4f}")

model = myNN(N_i, N_o, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = torch.nn.L1Loss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss_history = []
test_loss_history = []

def get_lr(epoch):
    for step_epoch, lr in lr_schedule:
        if epoch <= step_epoch:
            return lr
    return lr_schedule[-1][1]

for epoch in range(1, num_epochs + 1):
    lr = get_lr(epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    model.train()
    epoch_train_loss = 0.0
    for inputs, targets in dataloader_train:
        outputs = model(inputs)
        loss_u = criterion(outputs[:, :n_out], targets[:, :n_out])
        loss_v = criterion(outputs[:, n_out:], targets[:, n_out:])
        loss = w_u * loss_u + w_v * loss_v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    avg_train = epoch_train_loss / len(dataloader_train)
    train_loss_history.append(avg_train)

    model.eval()
    epoch_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            outputs = model(inputs)
            loss_u = criterion(outputs[:, :n_out], targets[:, :n_out])
            loss_v = criterion(outputs[:, n_out:], targets[:, n_out:])
            epoch_test_loss += (w_u * loss_u + w_v * loss_v).item()
    avg_test = epoch_test_loss / len(dataloader_test)
    test_loss_history.append(avg_test)

    print(f"Epoch [{epoch}/{num_epochs}], lr={lr:.0e}, Train: {avg_train:.10f}, Test: {avg_test:.10f}")

save_checkpoint(
    model, optimizer, epoch, train_loss_history, test_loss_history, _model_path,
    hidden_size=hidden_size, num_layers=num_layers,
)
savemat(_error_path, {"train_err": train_loss_history, "test_err": test_loss_history})
print(f"Saved {_model_path}")

# Training history plot (same style as 1D_burgers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
_history_plot_path = os.path.join(_script_dir, "training_history.png")
epochs = range(1, len(train_loss_history) + 1)
plt.figure(figsize=(7, 4))
plt.plot(epochs, train_loss_history, label="Train loss", linewidth=0.8)
plt.plot(epochs, test_loss_history, label="Test loss", linewidth=0.8)
plt.xlabel("Epoch")
plt.ylabel("L1 loss")
plt.title("Training and testing error history")
plt.legend()
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(_history_plot_path, dpi=150)
plt.close()
print(f"Saved training history plot to {_history_plot_path}")
