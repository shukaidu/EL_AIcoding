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


def _run_epochs(model, train_loader, test_loader, optimizer, cfg):
    crit_none = torch.nn.L1Loss(reduction="none")
    hist_tr, hist_te = [], []
    device = next(model.parameters()).device
    nonlinear = hasattr(cfg, "smooth_weight")
    if nonlinear:
        w = torch.tensor(cfg.param_ratio, dtype=torch.float32, device=device)
        smooth_weight = cfg.smooth_weight
        smooth_mode = cfg.smooth_mode

    def _loss(pred, tgt):
        raw = crit_none(pred, tgt)
        if not nonlinear:
            return raw.mean()
        loss = (raw.mean(dim=(0, 2, 3)) * w).sum()
        weights = [smooth_weight] * pred.shape[1] if isinstance(smooth_weight, (int, float)) else list(smooth_weight)
        tv = 0.0
        for c, sw in enumerate(weights):
            if sw == 0.0:
                continue
            pc = pred[:, c]
            gx = pc[:, :, 1:] - pc[:, :, :-1]
            gy = pc[:, 1:, :] - pc[:, :-1, :]
            tv_c = gx.abs().mean() + gy.abs().mean()
            if smooth_mode == "relative":
                tv_c = tv_c / pc.abs().mean().clamp(min=1e-6)
            tv = tv + sw * tv_c
        return loss + tv

    for epoch in range(1, cfg.num_epochs + 1):
        lr = next(lr for e, lr in cfg.lr_schedule if epoch <= e)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        tr = 0.0
        for inp, tgt in train_loader:
            loss = _loss(model(inp), tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr += loss.item()
        hist_tr.append(tr / len(train_loader))

        model.eval()
        with torch.no_grad():
            te = sum(_loss(model(inp), tgt).item() for inp, tgt in test_loader) / len(test_loader)
        hist_te.append(te)
        print(f"Epoch [{epoch}/{cfg.num_epochs}], lr={lr:.0e}, Train: {hist_tr[-1]:.6f}, Test: {te:.6f}")

    return hist_tr, hist_te


def _run(model, train_loader, test_loader, cfg, data_dir):
    """训练并保存曲线，返回 (opt, hist_tr, hist_te)。checkpoint 由调用方负责保存。"""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_schedule[0][1])
    hist_tr, hist_te = _run_epochs(model, train_loader, test_loader, opt, cfg)
    savemat(os.path.join(data_dir, cfg.error_mat), {"train_err": hist_tr, "test_err": hist_te})
    plot_training_history(hist_tr, hist_te, os.path.join(data_dir, "training_history.png"))
    return opt, hist_tr, hist_te


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
        tl, vl, N_i, N_o, _ = load_mat(path, device, cfg.b_size, cfg.test_split)
        model = MLP(N_i, N_o, cfg.hidden_size, cfg.num_layers, cfg.activation).to(device)
        opt, hist_tr, hist_te = _run(model, tl, vl, cfg, data_dir)
        save_checkpoint(model, opt, cfg.num_epochs, hist_tr, hist_te,
                        os.path.join(data_dir, cfg.model_pth),
                        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, activation=cfg.activation)
        print(f"Saved {os.path.join(data_dir, cfg.model_pth)}")
        return

    if problem == "wave_2d_nonlinear":
        import config.wave_2d_nonlinear_config as cfg
        path = os.path.join(data_dir, cfg.data_mat)
        if not os.path.isfile(path):
            print(f"Data not found: {path}. Run: python gen_data.py --problem wave_2d_nonlinear")
            return
        tl, vl, _, C_in, C_out, Nx, Ny, nx, ny, stats = load_wave_2d_nonlinear(path, device, cfg.b_size, cfg.test_split, cfg.residual)
        if cfg.model_type.lower() == "unet":
            model = UNet(C_in, C_out, cfg.base, Nx, nx, cfg.pooling).to(device)
        else:
            model = CNN(C_in, C_out, cfg.base, Nx, nx).to(device)
        opt, hist_tr, hist_te = _run(model, tl, vl, cfg, data_dir)
        save_checkpoint(model, opt, cfg.num_epochs, hist_tr, hist_te,
                        os.path.join(data_dir, cfg.model_pth),
                        base=cfg.base, model_type=cfg.model_type.lower(),
                        pooling=cfg.pooling, residual=cfg.residual,
                        ch_mean=stats["ch_mean"], ch_std=stats["ch_std"])
        print(f"Saved {os.path.join(data_dir, cfg.model_pth)}")


if __name__ == "__main__":
    main()
