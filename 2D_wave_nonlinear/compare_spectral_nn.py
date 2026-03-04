"""
Compare spectral (traditional) and NN solutions at each timestep.
State: U = (h-h0, qx, qy). Plots all 3 components spectral vs NN per frame.
Runtime comparison (spectral vs NN) is shown on each figure.
Model structure (Nx, nwd) is inferred from the checkpoint so it matches the trained model.
"""
import os
import sys
import re
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pde import wave2d_spectral
from params_wave_ml import nwd, njp, TSCREEN, nx, ny

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "ML", "data_wave_model.pth")


def boundary_ext(u, nst):
    """Periodic extension by nst in each direction. u (nx, ny)."""
    u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
    u_ext = np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])
    return u_ext


def integrate_nn_uv_cnn(model, U0, nwd, nst, patch_side, device):
    """One NN step: U0 (nx, ny, 3) -> U_nn (nx, ny, 3). Periodic + patch CNN."""
    nx, ny, _ = U0.shape
    U_nn = np.zeros_like(U0)
    u1_ext = boundary_ext(U0[:, :, 0], nst)
    u2_ext = boundary_ext(U0[:, :, 1], nst)
    u3_ext = boundary_ext(U0[:, :, 2], nst)
    for ii in range(nx // nwd):
        for jj in range(ny // nwd):
            ind1 = slice(ii * nwd, ii * nwd + patch_side)
            ind2 = slice(jj * nwd, jj * nwd + patch_side)
            tmp_in = np.stack([
                u1_ext[ind1, ind2],
                u2_ext[ind1, ind2],
                u3_ext[ind1, ind2],
            ], axis=0)
            # (3, H, W) -> (1, 3, H, W)
            inp = torch.tensor(tmp_in, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
            out = out.cpu().numpy().squeeze(0)  # (3, nwd, nwd)
            U_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd, :] = np.transpose(out, (1, 2, 0))
    return U_nn


def _infer_nx_from_checkpoint(path):
    """ShrinkCNN: net.0..net.L with L = (Nx - nx)//4. Return Nx (patch_side)."""
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    max_idx = -1
    for key in sd:
        m = re.match(r"net\.(\d+)\.", key)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx < 0:
        raise RuntimeError("Could not infer model architecture from checkpoint")
    # L = max_idx, Nx = nwd + 4*L
    return nwd + 4 * max_idx


def main():
    if not os.path.isfile(model_path):
        print("Train the model first: python gen_data.py && python ML/train.py")
        return
    out_dir = os.path.join(_script_dir, "compare_spectral_nn")
    os.makedirs(out_dir, exist_ok=True)

    # Infer patch_side from checkpoint so we build the same architecture as trained
    patch_side = _infer_nx_from_checkpoint(model_path)
    nst = (patch_side - nwd) // 2

    TF_compare = 1.0
    initial_condition = "random"
    rng_seed = 42

    # Spectral trajectory (same grid as training data), with timing
    t0_spectral = time.perf_counter()
    t_hist, U_hist, xx, yy, _, _, _, _ = wave2d_spectral(
        initial_condition=initial_condition,
        TF=TF_compare,
        TSCREEN=TSCREEN,
        nx=nx,
        ny=ny,
        rng_seed=rng_seed,
    )
    spectral_total_time = time.perf_counter() - t0_spectral
    n_frames = U_hist.shape[3]
    # Approximate cumulative time to reach frame k (linear in frame index)
    spectral_time_per_frame = [spectral_total_time * k / max(1, n_frames) for k in range(n_frames)]

    # Load CNN（base 从 checkpoint 读取以兼容不同训练配置）
    sys.path.insert(0, os.path.join(_script_dir, "ML"))
    from NN_EL import ShrinkCNN
    from SLmodel_EL import load_checkpoint

    ckpt = torch.load(model_path, map_location="cpu")
    base = ckpt.get("base", 32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ShrinkCNN(Cin=3, Cout=3, base=base, Nx=patch_side, nx=nwd).to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    U_nn = U_hist[:, :, :, 0].copy()
    nn_frames = [U_nn.copy()]
    nn_time_list = [0.0]  # cumulative time to reach each frame
    t0_nn = time.perf_counter()
    for _ in range(n_frames - 1):
        U_nn = integrate_nn_uv_cnn(model, U_nn, nwd, nst, patch_side, device)
        nn_frames.append(U_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)

    # 自检：每帧 L1 误差（Spectral vs NN，全网格平均）
    err_per_frame = [np.abs(U_hist[:, :, :, k] - nn_frames[k]).mean() for k in range(n_frames)]
    err_mean = np.mean(err_per_frame)
    err_max = np.max(err_per_frame)
    print(f"  L1 error (mean over frames): {err_mean:.6f}, max frame: {err_max:.6f}")

    comp_names = ["h - h0", "qx", "qy"]
    clims = []
    for c in range(3):
        spec_min = U_hist[:, :, c, :].min()
        spec_max = U_hist[:, :, c, :].max()
        nn_min = np.array([f[:, :, c].min() for f in nn_frames]).min()
        nn_max = np.array([f[:, :, c].max() for f in nn_frames]).max()
        lo = min(spec_min, nn_min)
        hi = max(spec_max, nn_max)
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    for k in range(n_frames):
        t_val = t_hist[k]
        time_spec = spectral_time_per_frame[k]
        time_nn = nn_time_list[k]
        speedup = time_spec / time_nn if time_nn > 0 else float("inf")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"IC: {initial_condition}  t = {t_val:.3f}", fontsize=12)
        for c, name in enumerate(comp_names):
            ax = axes[0, c]
            im = ax.pcolormesh(xx, yy, U_hist[:, :, c, k], shading="auto")
            ax.set_title(f"Spectral {name}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            im.set_clim(clims[c])
            plt.colorbar(im, ax=ax)
            ax = axes[1, c]
            im = ax.pcolormesh(xx, yy, nn_frames[k][:, :, c], shading="auto")
            ax.set_title(f"NN {name}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            im.set_clim(clims[c])
            plt.colorbar(im, ax=ax)
        # Runtime comparison (same style as 1D_burgers)
        time_text = (
            f"Spectral runtime: {time_spec:.3f} s\n"
            f"NN runtime: {time_nn:.4f} s\n"
            f"NN speedup: {speedup:.1f}×"
        )
        fig.text(0.02, 0.98, time_text, fontsize=9, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.tight_layout()
        fname = f"t{k}.png" if k < 10 else f"t{k:02d}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close()
    print(f"Saved {n_frames} figures to {out_dir}/  (IC={initial_condition}, nwd={nwd}, nst={nst}, njp={njp})")
    print(f"  Spectral total: {spectral_total_time:.3f} s, NN total: {nn_time_list[-1]:.4f} s")


if __name__ == "__main__":
    main()
