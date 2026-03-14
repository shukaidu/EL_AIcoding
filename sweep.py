"""Hyperparameter sweep for wave_2d_nonlinear.
Usage: python sweep.py
Each config: trains model → saves to standard path → runs full compare (with figures).
Output images: data/wave_2d_nonlinear/sweep/<label>/
Summary table printed at end.
"""
import os, sys, time
import numpy as np
import torch

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN
from ml.train_loop import get_device
from ml.snapshot import save_checkpoint
from ml.train import _run_epochs
from compare import _compare_wave_2d_nonlinear

import config.wave_2d_nonlinear_config as cfg

# ── Sweep grid ────────────────────────────────────────────────────────────────
LR_150 = [(60, 3e-4), (110, 1e-4), (140, 3e-5), (150, 1e-5)]
LR_200 = [(80, 3e-4), (150, 1e-4), (180, 3e-5), (200, 1e-5)]

CONFIGS = [
    {"label": "base64_ep150", "base": 64, "num_epochs": 150,
     "lr_schedule": LR_150, "smooth_weight": [0.0, 0.02, 0.02], "smooth_mode": "relative"},
    {"label": "base32_ep150", "base": 32, "num_epochs": 150,
     "lr_schedule": LR_150, "smooth_weight": [0.0, 0.02, 0.02], "smooth_mode": "relative"},
    {"label": "base48_ep200", "base": 48, "num_epochs": 200,
     "lr_schedule": LR_200, "smooth_weight": [0.0, 0.02, 0.02], "smooth_mode": "relative"},
    {"label": "base64_ep200", "base": 64, "num_epochs": 200,
     "lr_schedule": LR_200, "smooth_weight": [0.0, 0.02, 0.02], "smooth_mode": "relative"},
    {"label": "base48_tv01",  "base": 48, "num_epochs": 150,
     "lr_schedule": LR_150, "smooth_weight": [0.0, 0.01, 0.01], "smooth_mode": "relative"},
]
# ─────────────────────────────────────────────────────────────────────────────

# ── Main sweep ────────────────────────────────────────────────────────────────
device = get_device()
print(f"Device: {device}\n")

data_dir  = os.path.join(_repo_root, "data", "wave_2d_nonlinear")
sweep_dir = os.path.join(data_dir, "sweep")
model_path = os.path.join(data_dir, cfg.model_pth)

tl, vl, _, C_in, C_out, Nx, Ny, nx, ny = load_wave_2d_nonlinear(
    os.path.join(data_dir, cfg.data_mat), device, b_size=cfg.b_size
)

results = []

for entry in CONFIGS:
    label = entry["label"]
    sw    = entry["smooth_weight"]
    sm    = entry["smooth_mode"]
    _base = entry.get("base", cfg.base)
    _ep   = entry.get("num_epochs", cfg.num_epochs)
    _lr   = entry.get("lr_schedule", cfg.lr_schedule)
    out_dir = os.path.join(sweep_dir, label)
    print(f"\n{'='*60}")
    print(f"Config: {label}  base={_base}  ep={_ep}  w={sw}  mode={sm}")
    print('='*60)

    # train
    model = CNN(Cin=C_in, Cout=C_out, base=_base, Nx=Nx, nx=nx).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=_lr[0][1])
    t0 = time.time()
    hist_tr, hist_te = _run_epochs(
        model, tl, vl, opt,
        _ep, _lr,
        smooth_weight=sw, smooth_mode=sm,
    )
    train_time = time.time() - t0
    print(f"  Train done in {train_time:.0f}s  final test={hist_te[-1]:.6f}")

    # save to standard path (so compare can find it)
    save_checkpoint(model, opt, _ep, hist_tr, hist_te, model_path, base=_base)

    # compare with figures
    model.eval()
    l1_mean, l1_max, sp_t, nn_t = _compare_wave_2d_nonlinear(data_dir, out_dir, model=model)
    speedup = sp_t / nn_t if nn_t > 0 else float("nan")
    print(f"  L1 mean={l1_mean:.6f}  max={l1_max:.6f}  speedup={speedup:.1f}x")
    results.append(dict(label=label, base=_base, ep=_ep, sw=sw,
                        l1_mean=l1_mean, l1_max=l1_max,
                        speedup=speedup, train_time=train_time))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"{'Label':<22} {'base':>5} {'ep':>5} {'smooth_weight':<20} {'L1 mean':>10} {'L1 max':>10} {'Speedup':>8}")
print("-"*80)
for r in results:
    print(f"{r['label']:<22} {r['base']:>5} {r['ep']:>5} {str(r['sw']):<20} {r['l1_mean']:>10.6f} {r['l1_max']:>10.6f} {r['speedup']:>7.1f}x")
print("="*80)

best = min(results, key=lambda x: x["l1_mean"])
print(f"\nBest: {best['label']}  base={best['base']}  ep={best['ep']}  L1 mean={best['l1_mean']:.6f}")
