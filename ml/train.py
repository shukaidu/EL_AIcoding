"""Train: python -m ml.train --problem burgers_1d|wave_2d_linear|wave_2d_nonlinear"""
import os
import sys
import argparse
import torch
from scipy.io import savemat

_ml_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_ml_dir)
sys.path.insert(0, _repo_root)

from common.data_io import load_burgers_1d, load_wave_2d_linear, load_wave_2d_nonlinear
from common.models import MLP, ShrinkCNN
from common.train_loop import get_device, plot_training_history
from ml.snapshot import save_checkpoint


def get_lr(epoch, lr_schedule):
    for e, lr in lr_schedule:
        if epoch <= e:
            return lr
    return lr_schedule[-1][1]


def run_epochs(model, train_loader, test_loader, optimizer, num_epochs, lr_schedule, device, train_step_fn, test_step_fn):
    hist_tr, hist_te = [], []
    for epoch in range(1, num_epochs + 1):
        lr = get_lr(epoch, lr_schedule)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        model.train()
        tr = 0.0
        for i, t in train_loader:
            loss = train_step_fn(model, i, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr += loss.item()
        tr /= len(train_loader)
        hist_tr.append(tr)
        model.eval()
        with torch.no_grad():
            te = sum(test_step_fn(model, i, t) for i, t in test_loader) / len(test_loader)
        hist_te.append(te)
        print(f"Epoch [{epoch}/{num_epochs}], lr={lr:.0e}, Train: {tr:.6f}, Test: {te:.6f}")
    return hist_tr, hist_te


def save_artifacts(model, optimizer, num_epochs, hist_tr, hist_te, data_dir, cfg, **extra):
    save_checkpoint(model, optimizer, num_epochs, hist_tr, hist_te, os.path.join(data_dir, cfg.model_pth), **extra)
    savemat(os.path.join(data_dir, cfg.error_mat), {"train_err": hist_tr, "test_err": hist_te})
    plot_training_history(hist_tr, hist_te, os.path.join(data_dir, "training_history.png"))
    print(f"Saved {os.path.join(data_dir, cfg.model_pth)}")


def _run(model, train_loader, test_loader, cfg, data_dir, device, **ckpt_extra):
    crit = torch.nn.L1Loss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_schedule[0][1])
    train_step = lambda m, i, t: crit(m(i), t)
    test_step = lambda m, i, t: crit(m(i), t).item()
    hist_tr, hist_te = run_epochs(model, train_loader, test_loader, opt, cfg.num_epochs, cfg.lr_schedule, device, train_step, test_step)
    save_artifacts(model, opt, cfg.num_epochs, hist_tr, hist_te, data_dir, cfg, **ckpt_extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    problem = p.parse_args().problem
    data_dir = os.path.join(_repo_root, "data", problem)
    os.makedirs(data_dir, exist_ok=True)
    device = get_device()
    print(f"Training on: {device}")

    if problem == "burgers_1d":
        import config.burgers_1d_config as cfg
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem burgers_1d")
            return
        tl, vl, N_i, N_o, _ = load_burgers_1d(path, device, b_size=cfg.b_size, test_split=0.2)
        model = MLP(N_i, N_o, hidden_size=cfg.hidden_size, num_layers=cfg.num_hidden_layers, activation="relu").to(device)
        _run(model, tl, vl, cfg, data_dir, device, hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers)
        return

    if problem == "wave_2d_linear":
        import config.wave_2d_linear_config as cfg
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem wave_2d_linear")
            return
        tl, vl, N_i, N_o, _ = load_wave_2d_linear(path, device, b_size=cfg.b_size, test_split=0.2)
        model = MLP(N_i, N_o, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, activation="identity").to(device)
        _run(model, tl, vl, cfg, data_dir, device, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers)
        return

    if problem == "wave_2d_nonlinear":
        import config.wave_2d_nonlinear_config as cfg
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem wave_2d_nonlinear")
            return
        tl, vl, _, C_in, C_out, Nx, Ny, nx, ny = load_wave_2d_nonlinear(path, device, b_size=cfg.b_size, test_split=0.2)
        model = ShrinkCNN(Cin=C_in, Cout=C_out, base=cfg.base, Nx=Nx, nx=nx).to(device)
        _run(model, tl, vl, cfg, data_dir, device, base=cfg.base)
        return


if __name__ == "__main__":
    main()
