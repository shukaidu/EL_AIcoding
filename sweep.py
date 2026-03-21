"""Hyperparameter sweep for wave_2d_nonlinear — sweeps TV smooth_weight.
Usage: python sweep.py
Each config: trains model -> runs full compare (with figures).
Output images: data/wave_2d_nonlinear/sweep/<label>/
Summary table printed at end.
"""
import os
import sys
import time
import types
import torch

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN, UNet
from ml.train_loop import get_device
from ml.snapshot import save_checkpoint
from ml.train import _run_epochs
from compare import _compare_wave_2d_nonlinear

import config.wave_2d_nonlinear_config as cfg

# ---------------------------------------------------------------------------
# Sweep grid: vary TV smooth_weight only; all other params from config
# ---------------------------------------------------------------------------
SMOOTH_WEIGHTS = [
    [0.0, 0.1,  0.1 ],
    [0.0, 1.0,  1.0 ],
    [0.0, 10.0, 10.0],
    [0.1, 1.0,  1.0 ],
    [1, 1.0,  1.0 ],
]


def main():
    device = get_device()
    print(f"Device: {device}\n")

    data_dir = os.path.join(_repo_root, "data", "wave_2d_nonlinear")
    sweep_dir = os.path.join(data_dir, "sweep")
    model_path = os.path.join(data_dir, cfg.model_pth)

    tl, vl, _, C_in, C_out, Nx, Ny, nx, ny, stats = load_wave_2d_nonlinear(
        os.path.join(data_dir, cfg.data_mat), device, cfg.b_size, cfg.test_split, cfg.residual
    )
    ch_mean, ch_std = stats["ch_mean"], stats["ch_std"]

    results = []
    for sw in SMOOTH_WEIGHTS:
        label = f"tv_{sw[0]}_{sw[1]}_{sw[2]}"
        out_dir = os.path.join(sweep_dir, label)

        sweep_cfg = types.SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")})
        sweep_cfg.smooth_weight = sw

        print(f"\n{'='*60}")
        print(f"smooth_weight={sw}")
        print("=" * 60)

        if cfg.model_type.lower() == "unet":
            model = UNet(C_in, C_out, sweep_cfg.base, Nx, nx, sweep_cfg.pooling).to(device)
        else:
            model = CNN(C_in, C_out, sweep_cfg.base, Nx, nx).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=sweep_cfg.lr_schedule[0][1])

        t0 = time.time()
        hist_tr, hist_te = _run_epochs(model, tl, vl, opt, sweep_cfg)
        train_time = time.time() - t0
        print(f"  Train done in {train_time:.0f}s  final test={hist_te[-1]:.6f}")

        save_checkpoint(model, opt, sweep_cfg.num_epochs, hist_tr, hist_te, model_path,
                        base=sweep_cfg.base, model_type=sweep_cfg.model_type,
                        pooling=sweep_cfg.pooling, residual=sweep_cfg.residual,
                        ch_mean=ch_mean, ch_std=ch_std)

        model.eval()
        l1_mean, l1_max, sp_t, nn_t = _compare_wave_2d_nonlinear(
            data_dir, out_dir, model=model, residual=sweep_cfg.residual,
            ch_mean=ch_mean, ch_std=ch_std)
        speedup = sp_t / nn_t if nn_t > 0 else float("nan")
        print(f"  L1 mean={l1_mean:.6f}  max={l1_max:.6f}  speedup={speedup:.1f}x")
        results.append(dict(label=label, sw=sw, l1_mean=l1_mean, l1_max=l1_max,
                            speedup=speedup, train_time=train_time))

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'smooth_weight':<25} {'L1 mean':>10} {'L1 max':>10} {'Speedup':>8}")
    print("-" * 70)
    for r in results:
        print(f"{str(r['sw']):<25} {r['l1_mean']:>10.6f} {r['l1_max']:>10.6f} {r['speedup']:>7.1f}x")
    print("=" * 70)

    best = min(results, key=lambda x: x["l1_mean"])
    print(f"\nBest: smooth_weight={best['sw']}  L1 mean={best['l1_mean']:.6f}")


if __name__ == "__main__":
    main()
