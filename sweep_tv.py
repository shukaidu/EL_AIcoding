"""TV parameter sweep for wave_2d_nonlinear.
Usage: python sweep_tv.py
Each config: trains model → saves to standard path → runs full compare (with figures).
Output images: data/wave_2d_nonlinear/sweep/<label>/
Summary table printed at end.
"""
import os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

from ml.data_io import load_wave_2d_nonlinear
from ml.models import CNN
from ml.train_loop import get_device
from ml.snapshot import save_checkpoint, load_checkpoint
from ml.train import _run_epochs

import config.wave_2d_nonlinear_config as cfg
from pde.wave_2d_nonlinear import setup_wave2d_nonlinear, advance_tscreen

# ── Sweep grid ────────────────────────────────────────────────────────────────
CONFIGS = [
    {"smooth_weight": [0.0,   0.02, 0.02], "smooth_mode": "relative", "label": "h0.000"},
    {"smooth_weight": [0.001, 0.02, 0.02], "smooth_mode": "relative", "label": "h0.001"},
    {"smooth_weight": [0.002, 0.02, 0.02], "smooth_mode": "relative", "label": "h0.002"},
]
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_ext(u, nst):
    u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
    return np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])


def _speedup_str(t_ref, t_nn):
    return f"{t_ref/t_nn:.1f}x" if t_nn > 0 else "N/A"


def _run_compare(model, device, out_dir):
    """Full compare with figures. Returns (l1_mean, l1_max, sp_elapsed, nn_elapsed)."""
    h, qx, qy, rhs, dt, xx, yy = setup_wave2d_nonlinear(
        cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
        g=cfg.g, h0=cfg.h0, f_coriolis=cfg.f_coriolis,
        nu_h=cfg.nu_h, nu_q=cfg.nu_q,
        initial_condition=cfg.compare_ic, rng_seed=cfg.compare_seed,
    )
    steps_per_nn = cfg.TSCREEN * cfg.njp
    n_nn_steps = int(round(cfg.compare_TF / (steps_per_nn * dt)))
    nwd, nst, ps = cfg.nwd, cfg.nst, cfg.patch_side

    # warmup
    warmup_steps = int(round(cfg.warmup_T / (cfg.TSCREEN * dt)))
    for _ in range(warmup_steps):
        h, qx, qy = advance_tscreen(h, qx, qy, rhs, dt, cfg.TSCREEN)

    initial = np.stack([h - cfg.h0, qx, qy], axis=-1)
    U_nn = initial.copy()
    spec_frames = [initial.copy()]
    nn_frames = [U_nn.copy()]
    spec_time_list = [0.0]
    nn_time_list = [0.0]
    sp_elapsed = nn_elapsed = 0.0

    for _ in range(n_nn_steps):
        t0 = time.perf_counter()
        h, qx, qy = advance_tscreen(h, qx, qy, rhs, dt, steps_per_nn)
        sp_elapsed += time.perf_counter() - t0
        spec_frames.append(np.stack([h - cfg.h0, qx, qy], axis=-1))
        spec_time_list.append(sp_elapsed)

        U_ext = [_boundary_ext(U_nn[:, :, c], nst) for c in range(3)]
        U_out = np.zeros_like(U_nn)
        t0 = time.perf_counter()
        for ii in range(cfg.nx // nwd):
            for jj in range(cfg.ny // nwd):
                s1 = slice(ii * nwd, ii * nwd + ps)
                s2 = slice(jj * nwd, jj * nwd + ps)
                inp = np.stack([U_ext[c][s1, s2] for c in range(3)], axis=0).astype(np.float32)
                inp_t = torch.tensor(inp).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp_t).cpu().numpy().squeeze(0)
                U_out[ii*nwd:(ii+1)*nwd, jj*nwd:(jj+1)*nwd, :] = np.transpose(out, (1, 2, 0))
        nn_elapsed += time.perf_counter() - t0
        U_nn = U_out
        nn_frames.append(U_nn.copy())
        nn_time_list.append(nn_elapsed)

    errs = [np.abs(spec_frames[i] - nn_frames[i]).mean() for i in range(len(nn_frames))]
    l1_mean = float(np.mean(errs))
    l1_max  = float(np.max(errs))

    # shared color limits per channel
    comp_names = ["h - h0", "qx", "qy"]
    clims = []
    for c in range(3):
        lo = min(np.array([f[:,:,c] for f in spec_frames]).min(),
                 np.array([f[:,:,c] for f in nn_frames]).min())
        hi = max(np.array([f[:,:,c] for f in spec_frames]).max(),
                 np.array([f[:,:,c] for f in nn_frames]).max())
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    os.makedirs(out_dir, exist_ok=True)
    dt_nn = steps_per_nn * dt
    plot_indices = np.linspace(0, len(nn_frames) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        t_val = i * dt_nn
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for c in range(3):
            pc0 = axes[0, c].pcolormesh(xx, yy, spec_frames[i][:,:,c].T, shading="auto")
            axes[0, c].set_title(f"Spectral {comp_names[c]}")
            pc1 = axes[1, c].pcolormesh(xx, yy, nn_frames[i][:,:,c].T, shading="auto")
            axes[1, c].set_title(f"NN {comp_names[c]}")
            pc0.set_clim(clims[c]); pc1.set_clim(clims[c])
            fig.colorbar(pc0, ax=[axes[0,c], axes[1,c]], label=comp_names[c])
        fig.text(0.02, 0.98,
                 f"Spectral: {spec_time_list[i]:.3f} s\nNN: {nn_time_list[i]:.4f} s\n"
                 f"Speedup: {_speedup_str(spec_time_list[i], nn_time_list[i])}\n"
                 f"L1 mean: {l1_mean:.6f}",
                 fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.suptitle(f"t = {t_val:.3f}")
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=120)
        plt.close()

    print(f"  Saved {len(plot_indices)} figures to {out_dir}/")
    return l1_mean, l1_max, sp_elapsed, nn_elapsed


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
    out_dir = os.path.join(sweep_dir, label)
    print(f"\n{'='*60}")
    print(f"Config: {label}  w={sw}  mode={sm}")
    print('='*60)

    # train
    model = CNN(Cin=C_in, Cout=C_out, base=cfg.base, Nx=Nx, nx=nx).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr_schedule[0][1])
    t0 = time.time()
    hist_tr, hist_te = _run_epochs(
        model, tl, vl, opt,
        cfg.num_epochs, cfg.lr_schedule,
        smooth_weight=sw, smooth_mode=sm,
    )
    train_time = time.time() - t0
    print(f"  Train done in {train_time:.0f}s  final test={hist_te[-1]:.6f}")

    # save to standard path (so compare can find it)
    save_checkpoint(model, opt, cfg.num_epochs, hist_tr, hist_te, model_path, base=cfg.base)

    # compare with figures
    model.eval()
    l1_mean, l1_max, sp_t, nn_t = _run_compare(model, device, out_dir)
    speedup = sp_t / nn_t if nn_t > 0 else float("nan")
    print(f"  L1 mean={l1_mean:.6f}  max={l1_max:.6f}  speedup={speedup:.1f}x")
    results.append(dict(label=label, sw=sw, l1_mean=l1_mean, l1_max=l1_max,
                        speedup=speedup, train_time=train_time))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Label':<22} {'smooth_weight':<26} {'L1 mean':>10} {'L1 max':>10} {'Speedup':>8}")
print("-"*75)
for r in results:
    print(f"{r['label']:<22} {str(r['sw']):<26} {r['l1_mean']:>10.6f} {r['l1_max']:>10.6f} {r['speedup']:>7.1f}x")
print("="*75)

best = min(results, key=lambda x: x["l1_mean"])
print(f"\nBest: {best['label']}  L1 mean={best['l1_mean']:.6f}")
