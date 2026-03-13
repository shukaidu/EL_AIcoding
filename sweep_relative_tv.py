"""Sweep: per-channel relative TV regularization for wave_2d_nonlinear.

Usage:
    python sweep_relative_tv.py
"""
import sys, io, os, re, shutil
import torch
import config.wave_2d_nonlinear_config as cfg
from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN
from ml.train import _run
from compare import _compare_wave_2d_nonlinear
from ml.train_loop import get_device

PARAM_GRID = [
    {"smooth_weight": 0.01,  "smooth_mode": "relative"},
    {"smooth_weight": 0.05,  "smooth_mode": "relative"},
    {"smooth_weight": 0.1,   "smooth_mode": "relative"},
    {"smooth_weight": 0.2,   "smooth_mode": "relative"},
    {"smooth_weight": 0.5,   "smooth_mode": "relative"},
]

BASELINE = {"smooth_weight": 0.05, "smooth_mode": "absolute", "base": 48, "l1_mean": 0.014217}

data_dir = "data/wave_2d_nonlinear"
sweep_dir = os.path.join(data_dir, "sweep_rel")
os.makedirs(sweep_dir, exist_ok=True)

device = get_device()
print(f"Device: {device}")

# Load data once
data_path = os.path.join(data_dir, cfg.data_mat)
if not os.path.isfile(data_path):
    print(f"Data not found: {data_path}. Run: python gen_data.py --problem wave_2d_nonlinear")
    sys.exit(1)

tl, vl, _, C_in, C_out, Nx, Ny, nx, ny = load_wave_2d_nonlinear(
    data_path, device, b_size=cfg.b_size
)

BASE = 48  # fixed

results = []
for params in PARAM_GRID:
    sw = params["smooth_weight"]
    mode = params["smooth_mode"]
    tag = f"sw{sw}"
    model_rel = f"sweep_rel/model_{tag}.pth"
    cmp_dir = os.path.join(sweep_dir, f"compare_{tag}")
    os.makedirs(cmp_dir, exist_ok=True)

    cfg.smooth_weight = sw
    cfg.smooth_mode = mode
    cfg.base = BASE
    cfg.model_pth = model_rel

    print(f"\n=== smooth_weight={sw}, smooth_mode={mode}, base={BASE} ===")
    model = CNN(Cin=C_in, Cout=C_out, base=BASE, Nx=Nx, nx=nx).to(device)
    _run(model, tl, vl, cfg, data_dir, base=BASE)

    buf = io.StringIO()
    sys.stdout, old = buf, sys.stdout
    _compare_wave_2d_nonlinear(data_dir, cmp_dir)
    sys.stdout = old
    output = buf.getvalue()
    print(output)

    m = re.search(r"L1 error mean: ([0-9.]+)", output)
    l1 = float(m.group(1)) if m else float("inf")
    results.append({"smooth_weight": sw, "smooth_mode": mode, "l1_mean": l1, "tag": tag})
    print(f"  -> L1 mean = {l1:.6f}")

# Summary table (ASCII only to avoid cp1252 encoding errors)
print("\n=== Relative TV Sweep Results (sorted by L1) ===")
print(f"  {'mode':<10} {'sw':>6} {'base':>5}   L1")
print(f"  {'-'*10} {'-'*6} {'-'*5}   {'-'*10}")
for r in sorted(results, key=lambda x: x["l1_mean"]):
    print(f"  {r['smooth_mode']:<10} {r['smooth_weight']:>6.3f} {BASE:>5}   {r['l1_mean']:.6f}")

# Baseline comparison row
print(f"\n  Baseline (absolute, sw={BASELINE['smooth_weight']}, base={BASELINE['base']}): L1={BASELINE['l1_mean']:.6f}")

best = min(results, key=lambda x: x["l1_mean"])
print(f"\nBest relative TV: smooth_weight={best['smooth_weight']} -> L1={best['l1_mean']:.6f}")
if best["l1_mean"] < BASELINE["l1_mean"]:
    print(f"  -> IMPROVEMENT over baseline ({best['l1_mean']:.6f} < {BASELINE['l1_mean']:.6f})")
    print(f"  -> Update config: smooth_weight={best['smooth_weight']}, smooth_mode='relative', base={BASE}")
else:
    print(f"  -> No improvement over baseline (best={best['l1_mean']:.6f} >= baseline={BASELINE['l1_mean']:.6f})")
    print("  -> Keep config: smooth_mode='absolute', smooth_weight=0.05, base=48")
