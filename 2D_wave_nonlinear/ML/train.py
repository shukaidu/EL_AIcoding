"""
Train 2D nonlinear shallow-water ShrinkCNN. Run from repo root:
  python gen_data.py
  python ML/train.py
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_common_EL import get_device, setup_problem, train_and_eval
from SLmodel_EL import save_checkpoint as save_ckpt

_ml_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_ml_dir)
dataname = os.path.join(_parent_dir, "ML", "data_wave.mat")
modelname = os.path.join(_ml_dir, "data_wave_model.pth")
training_error = os.path.join(_ml_dir, "data_wave_error.mat")
_history_plot_path = os.path.join(_ml_dir, "training_history.png")

b_size = 128
num_epochs = 300
base = 64
# LR 调度（300 epoch，约 10–15 分钟）
lr_schedule = [(180, 1e-3), (260, 1e-4), (300, 1e-5)]

device = get_device()
dataloader_train, dataloader_test, model, lam_tv, param_ratio, criterion = setup_problem(
    dataname, b_size, device, base=base
)

# 按分量归一化损失：用训练集各通道 std 的倒数作为权重，平衡 h / qx / qy
train_targets = torch.cat([batch[1] for batch in dataloader_train], dim=0)
std_per_ch = train_targets.std(dim=(0, 2, 3)).to(device) + 1e-8
component_weights = (1.0 / std_per_ch).to(device)
print(f"Output scale (std per ch): {std_per_ch.cpu().numpy()} -> loss weights = 1/std")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def get_lr(epoch):
    for step_epoch, lr in lr_schedule:
        if epoch <= step_epoch:
            return lr
    return lr_schedule[-1][1]


train_loss_history = []
test_loss_history = []
for epoch in range(1, num_epochs + 1):
    lr = get_lr(epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    model, optimizer, _, train_loss_history, test_loss_history = train_and_eval(
        model=model,
        optimizer=optimizer,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        lam_tv=lam_tv,
        param_ratio=param_ratio,
        criterion=criterion,
        num_epochs=1,
        start_epoch=epoch - 1,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        modelname=None,
        training_error=None,
        component_weights=component_weights,
    )

save_ckpt(model, optimizer, num_epochs, train_loss_history, test_loss_history, modelname, base=base)
if training_error:
    from scipy.io import savemat
    savemat(training_error, {"train_err": train_loss_history, "test_err": test_loss_history})
print(f"Saved {modelname}")

# Training and testing error history plot (same style as 1D_burgers / 2D_wave_linear)
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
