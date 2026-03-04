"""
Train the Burgers NN on data_res.mat. Run from project root:
  python gen_data.py
  python ml/train.py
"""
import os
import torch

# Resolve paths: data and model in ml/
_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(_script_dir, "data_res.mat")
_model_path = os.path.join(_script_dir, "data_res_model.pth")
_error_path = os.path.join(_script_dir, "data_res_error.mat")
_history_plot_path = os.path.join(_script_dir, "training_history.png")

from nn_model import mat2pyt, myNN
from snapshot import save_checkpoint

dataname = _data_path
modelname = _model_path
training_error = _error_path

# 超参：更大 batch、更多 epoch、可配置网络
b_size = 128
num_epochs = 2500
hidden_size = 256
num_hidden_layers = 6
lr_schedule = [
    (1000, 1e-3),
    (2000, 1e-4),
    (2500, 1e-5),
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
print(f"Data: {dataname}")
print(f"NN: hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}")

dataloader_train, dataloader_test, N_i, N_o, N_samp = mat2pyt(
    dataname, device, b_size, test_split=0.2
)

model = myNN(
    N_i, N_o,
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
).to(device)
criterion = torch.nn.L1Loss()
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
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    epoch_train_loss = 0.0
    epoch_test_loss = 0.0

    model.train()
    for inputs, targets in dataloader_train:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(dataloader_train)
    train_loss_history.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader_test:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_test_loss += loss.item()

    avg_test_loss = epoch_test_loss / len(dataloader_test)
    test_loss_history.append(avg_test_loss)

    print(
        f"Epoch [{epoch}/{num_epochs}], lr={lr:.0e}, Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}"
    )

print(f"\nFinal: Train Loss = {train_loss_history[-1]:.10f}, Test Loss = {test_loss_history[-1]:.10f}")

save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss_history,
    test_loss_history,
    modelname,
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
)
from scipy.io import savemat

savemat(training_error, {"train_err": train_loss_history, "test_err": test_loss_history})
print(f"Saved model to {modelname}")

# Plot and save training / testing error history
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
