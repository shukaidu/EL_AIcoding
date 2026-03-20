"""Train: python -m ml.train --problem burgers_1d|wave_2d_linear|wave_2d_nonlinear"""
import importlib
import os
import sys
import argparse
import torch
from scipy.io import savemat

_ml_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_ml_dir)
sys.path.insert(0, _repo_root)

from ml.data_io import load_mat, load_wave_2d_nonlinear
from ml.models import MLP, CNN, UNet
from ml.train_loop import get_device, plot_training_history
from ml.snapshot import save_checkpoint


def _tv_loss(pred, smooth_weight, smooth_mode):
    """Total-variation regularisation loss over spatial dimensions."""
    C = pred.shape[1]
    weights = [smooth_weight] * C if isinstance(smooth_weight, (int, float)) else list(smooth_weight)
    tv = 0.0
    for c, w in enumerate(weights):
        if w == 0.0:
            continue
        pc = pred[:, c]
        gx = pc[:, :, 1:] - pc[:, :, :-1]
        gy = pc[:, 1:, :] - pc[:, :-1, :]
        tv_c = gx.abs().mean() + gy.abs().mean()
        if smooth_mode == "relative":
            tv_c = tv_c / pc.abs().mean().clamp(min=1e-6)
        tv = tv + w * tv_c
    return tv


def _eval_loss(model, loader, crit, crit_none, param_ratio, device):
    """Mean test loss over loader."""
    model.eval()
    with torch.no_grad():
        if param_ratio is not None:
            w = torch.tensor(param_ratio, dtype=torch.float32, device=device)
            total = sum((crit_none(model(i), t).mean(dim=(0, 2, 3)) * w).sum().item()
                        for i, t in loader)
        else:
            total = sum(crit(model(i), t).item() for i, t in loader)
    return total / len(loader)


def _run_epochs(model, train_loader, test_loader, optimizer, num_epochs, lr_schedule,
                smooth_weight=0.0, smooth_mode="absolute", param_ratio=None):
    crit = torch.nn.L1Loss(reduction="mean")
    crit_none = torch.nn.L1Loss(reduction="none")
    hist_tr, hist_te = [], []
    device = next(model.parameters()).device

    for epoch in range(1, num_epochs + 1):
        lr = next(lr for e, lr in lr_schedule if epoch <= e)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        tr = 0.0
        for inp, tgt in train_loader:
            pred = model(inp)
            if param_ratio is not None:
                w = torch.tensor(param_ratio, dtype=torch.float32, device=pred.device)
                loss = (crit_none(pred, tgt).mean(dim=(0, 2, 3)) * w).sum()
            else:
                loss = crit(pred, tgt)
            tv = _tv_loss(pred, smooth_weight, smooth_mode)
            if tv != 0.0:
                loss = loss + tv
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr += loss.item()
        hist_tr.append(tr / len(train_loader))

        te = _eval_loss(model, test_loader, crit, crit_none, param_ratio, device)
        hist_te.append(te)
        print(f"Epoch [{epoch}/{num_epochs}], lr={lr:.0e}, Train: {hist_tr[-1]:.6f}, Test: {te:.6f}")

    return hist_tr, hist_te


def _run(model, train_loader, test_loader, cfg, data_dir, **ckpt_extra):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_schedule[0][1])
    hist_tr, hist_te = _run_epochs(
        model, train_loader, test_loader, opt, cfg.num_epochs, cfg.lr_schedule,
        smooth_weight=getattr(cfg, "smooth_weight", 0.0),
        smooth_mode=getattr(cfg, "smooth_mode", "absolute"),
        param_ratio=getattr(cfg, "param_ratio", None),
    )
    save_checkpoint(model, opt, cfg.num_epochs, hist_tr, hist_te,
                    os.path.join(data_dir, cfg.model_pth), **ckpt_extra)
    savemat(os.path.join(data_dir, cfg.error_mat), {"train_err": hist_tr, "test_err": hist_te})
    plot_training_history(hist_tr, hist_te, os.path.join(data_dir, "training_history.png"))
    print(f"Saved {os.path.join(data_dir, cfg.model_pth)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    problem = p.parse_args().problem
    data_dir = os.path.join(_repo_root, "data", problem)
    os.makedirs(data_dir, exist_ok=True)
    device = get_device()
    print(f"Training on: {device}")

    _MLP_PROBLEMS = {
        "burgers_1d": "config.burgers_1d_config",
        "wave_2d_linear": "config.wave_2d_linear_config",
    }
    if problem in _MLP_PROBLEMS:
        cfg = importlib.import_module(_MLP_PROBLEMS[problem])
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem {problem}")
            return
        tl, vl, N_i, N_o, _ = load_mat(path, device, b_size=cfg.b_size)
        activation = getattr(cfg, "activation", "relu")
        model = MLP(N_i, N_o, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, activation=activation).to(device)
        _run(model, tl, vl, cfg, data_dir, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, activation=activation)
        return

    if problem == "wave_2d_nonlinear":
        import config.wave_2d_nonlinear_config as cfg
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem wave_2d_nonlinear")
            return
        residual = getattr(cfg, "residual", False)
        tl, vl, _, C_in, C_out, Nx, Ny, nx, ny, stats = load_wave_2d_nonlinear(
            path, device, b_size=cfg.b_size, residual=residual)
        model_type = getattr(cfg, "model_type", "cnn").lower()
        if model_type == "unet":
            pooling = getattr(cfg, "pooling", "max")
            model = UNet(Cin=C_in, Cout=C_out, base=cfg.base, Nx=Nx, nx=nx, pooling=pooling).to(device)
        else:
            pooling = "max"
            model = CNN(Cin=C_in, Cout=C_out, base=cfg.base, Nx=Nx, nx=nx).to(device)
        _run(model, tl, vl, cfg, data_dir, base=cfg.base, model_type=model_type, pooling=pooling,
             ch_mean=stats["ch_mean"], ch_std=stats["ch_std"], residual=residual)


if __name__ == "__main__":
    main()
