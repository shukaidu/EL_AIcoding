"""
Single training entry. Run from repo root:
  python -m ml.train --problem burgers_1d
  python -m ml.train --problem wave_2d_linear
  python -m ml.train --problem wave_2d_nonlinear
"""
import os
import sys
import argparse
import torch
from scipy.io import savemat

# Repo root = parent of ml/
_ml_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_ml_dir)
sys.path.insert(0, _repo_root)

from common.data_io import load_burgers_1d, load_wave_2d_linear, load_wave_2d_nonlinear
from common.models import MLP, ShrinkCNN, tv_isotropic_per_channel
from common.train_loop import get_device, plot_training_history
from ml.snapshot import save_checkpoint


def get_lr(epoch, lr_schedule):
    for step_epoch, lr in lr_schedule:
        if epoch <= step_epoch:
            return lr
    return lr_schedule[-1][1]


def run_epochs(
    model,
    train_loader,
    test_loader,
    optimizer,
    num_epochs,
    lr_schedule,
    device,
    train_step_fn,
    test_step_fn,
    flush_print=False,
):
    """
    train_step_fn(model, inputs, targets) -> loss tensor.
    test_step_fn(model, inputs, targets) -> scalar for logging.
    Returns (train_loss_history, test_loss_history).
    """
    train_loss_history = []
    test_loss_history = []
    for epoch in range(1, num_epochs + 1):
        lr = get_lr(epoch, lr_schedule)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        model.train()
        epoch_train = 0.0
        for inputs, targets in train_loader:
            loss = train_step_fn(model, inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train += loss.item()
        train_loss_history.append(epoch_train / len(train_loader))
        model.eval()
        epoch_test = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                epoch_test += test_step_fn(model, inputs, targets)
        test_loss_history.append(epoch_test / len(test_loader))
        kw = {"flush": True} if flush_print else {}
        print(
            f"Epoch [{epoch}/{num_epochs}], lr={lr:.0e}, Train: {train_loss_history[-1]:.10f}, Test: {test_loss_history[-1]:.10f}",
            **kw,
        )
    return train_loss_history, test_loss_history


def save_artifacts(model, optimizer, num_epochs, train_loss_history, test_loss_history, data_dir, cfg, **ckpt_extra):
    """Save checkpoint, error mat, and training plot."""
    save_checkpoint(
        model, optimizer, num_epochs, train_loss_history, test_loss_history,
        os.path.join(data_dir, cfg.model_pth),
        **ckpt_extra,
    )
    savemat(
        os.path.join(data_dir, cfg.error_mat),
        {"train_err": train_loss_history, "test_err": test_loss_history},
    )
    plot_training_history(train_loss_history, test_loss_history, os.path.join(data_dir, "training_history.png"))
    print(f"Saved {os.path.join(data_dir, cfg.model_pth)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    args = parser.parse_args()
    problem = args.problem

    data_dir = os.path.join(_repo_root, "data", problem)
    os.makedirs(data_dir, exist_ok=True)
    device = get_device()
    print(f"Training on: {device}")

    if problem == "burgers_1d":
        import config.burgers_1d_config as cfg
        dataname = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(dataname):
            print(f"Data not found: {dataname}. Run: python gen_data.py --problem burgers_1d")
            return
        train_loader, test_loader, N_i, N_o, _ = load_burgers_1d(dataname, device, b_size=cfg.b_size, test_split=0.2)
        model = MLP(N_i, N_o, hidden_size=cfg.hidden_size, num_layers=cfg.num_hidden_layers, activation="relu").to(device)
        criterion = torch.nn.L1Loss()
        def train_step(m, i, t):
            return criterion(m(i), t)
        def test_step(m, i, t):
            return criterion(m(i), t).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loss_history, test_loss_history = run_epochs(
            model, train_loader, test_loader, optimizer, cfg.num_epochs, cfg.lr_schedule, device, train_step, test_step,
        )
        save_artifacts(
            model, optimizer, cfg.num_epochs, train_loss_history, test_loss_history, data_dir, cfg,
            hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
        )
        return

    if problem == "wave_2d_linear":
        import config.wave_2d_linear_config as cfg
        dataname = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(dataname):
            print(f"Data not found: {dataname}. Run: python gen_data.py --problem wave_2d_linear")
            return
        train_loader, test_loader, N_i, N_o, _ = load_wave_2d_linear(dataname, device, b_size=cfg.b_size, test_split=0.2)
        n_out = N_o // 2
        train_targets = torch.cat([b[1] for b in train_loader], dim=0)
        std_u = train_targets[:, :n_out].std().item() + 1e-8
        std_v = train_targets[:, n_out:].std().item() + 1e-8
        w_u, w_v = 1.0 / std_u, 1.0 / std_v
        print(f"Output scale: std_u={std_u:.6f}, std_v={std_v:.6f}")
        model = MLP(N_i, N_o, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, activation="identity").to(device)
        criterion = torch.nn.L1Loss(reduction="mean")
        def train_step(m, i, t):
            return w_u * criterion(m(i)[:, :n_out], t[:, :n_out]) + w_v * criterion(m(i)[:, n_out:], t[:, n_out:])
        def test_step(m, i, t):
            return (w_u * criterion(m(i)[:, :n_out], t[:, :n_out]) + w_v * criterion(m(i)[:, n_out:], t[:, n_out:])).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loss_history, test_loss_history = run_epochs(
            model, train_loader, test_loader, optimizer, cfg.num_epochs, cfg.lr_schedule, device, train_step, test_step,
        )
        save_artifacts(
            model, optimizer, cfg.num_epochs, train_loss_history, test_loss_history, data_dir, cfg,
            hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        )
        return

    if problem == "wave_2d_nonlinear":
        import config.wave_2d_nonlinear_config as cfg
        dataname = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(dataname):
            print(f"Data not found: {dataname}. Run: python gen_data.py --problem wave_2d_nonlinear")
            return
        train_loader, test_loader, _, C_in, C_out, Nx, Ny, nx, ny = load_wave_2d_nonlinear(
            dataname, device, b_size=cfg.b_size, test_split=0.2
        )
        print("number layers:", (Nx - nx) // 4, flush=True)
        print("Computing output scale from train set...", flush=True)
        n_ch = 3
        sum_ = torch.zeros(n_ch, device=device)
        sum_sq = torch.zeros(n_ch, device=device)
        count = 0
        for _, targets in train_loader:
            v = targets.mean(dim=(0, 2, 3))
            s2 = (targets ** 2).mean(dim=(0, 2, 3))
            n = targets.size(0) * targets.size(2) * targets.size(3)
            sum_ += v * n
            sum_sq += s2 * n
            count += n
        mean_per_ch = sum_ / count
        std_per_ch = (sum_sq / count - mean_per_ch ** 2).clamp(min=0).sqrt() + 1e-8
        component_weights = (1.0 / std_per_ch).to(device)
        print(f"Output scale (std per ch): {std_per_ch.cpu().numpy()} -> loss weights = 1/std", flush=True)
        model = ShrinkCNN(Cin=C_in, Cout=C_out, base=cfg.base, Nx=Nx, nx=nx).to(device)
        lam_tv = torch.tensor([1e-2, 0, 0], device=device)
        criterion = torch.nn.L1Loss(reduction="none")
        def train_step(m, i, t):
            out = m(i)
            diff = criterion(out, t)
            data_loss_vec = diff.mean(dim=(0, 2, 3))
            tv_vec = tv_isotropic_per_channel(out)
            return (data_loss_vec * component_weights + lam_tv * tv_vec).sum()
        def test_step(m, i, t):
            out = m(i)
            diff = criterion(out, t)
            data_loss_vec = diff.mean(dim=(0, 2, 3))
            tv_vec = tv_isotropic_per_channel(out)
            return (data_loss_vec * component_weights + lam_tv * tv_vec).sum().item()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loss_history, test_loss_history = run_epochs(
            model, train_loader, test_loader, optimizer, cfg.num_epochs, cfg.lr_schedule, device, train_step, test_step,
            flush_print=True,
        )
        save_artifacts(
            model, optimizer, cfg.num_epochs, train_loss_history, test_loss_history, data_dir, cfg,
            base=cfg.base,
        )
        return


if __name__ == "__main__":
    main()
