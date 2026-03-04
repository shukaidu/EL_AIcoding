"""
Compare spectral (traditional) and NN solutions at each timestep.
Runtime comparison (spectral vs NN) is shown on each figure.
Uses same params as gen_data (params_wave_ml): nwd, nst, njp, dt_samp so the
NN input window correctly covers the domain of dependence for the time jump.
"""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pde import wave2d_main
from params_wave_ml import NX, NY, Lx, Ly, dt, TSCREEN, nwd, njp, nst, patch_side

# Compare up to t=1 (same dt_samp as training)
TF_compare = 1.0
# Test case: packet (moving), ring (expanding), flower, collide, random_white, random_band
initial_condition = "ring"
rng_seed = 42

_script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_script_dir, "ML", "data_wave_model.pth")


def integrate_nn_uv(model, u, v, nwd, nst, device):
    """One NN step: (u,v) -> (u_nn, v_nn). Periodic extension, patch-based."""
    nx, ny = u.shape
    u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
    u_ext = np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])
    v_ext = np.hstack([v[:, -nst:], v, v[:, :nst]])
    v_ext = np.vstack([v_ext[-nst:], v_ext, v_ext[:nst]])
    u_nn = np.zeros_like(u)
    v_nn = np.zeros_like(v)
    for ii in range(nx // nwd):
        for jj in range(ny // nwd):
            pu = u_ext[ii * nwd : ii * nwd + patch_side, jj * nwd : jj * nwd + patch_side].astype(np.float32)
            pv = v_ext[ii * nwd : ii * nwd + patch_side, jj * nwd : jj * nwd + patch_side].astype(np.float32)
            inp = np.concatenate([pu.ravel(), pv.ravel()])
            inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp_t)
            out = out.cpu().numpy().ravel()
            n = nwd * nwd
            u_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[:n].reshape(nwd, nwd)
            v_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[n : 2 * n].reshape(nwd, nwd)
    return u_nn, v_nn


def main():
    if not os.path.isfile(model_path):
        print("Train the model first: python gen_data.py && python ML/train.py")
        return
    out_dir = os.path.join(_script_dir, "compare_spectral_nn")
    os.makedirs(out_dir, exist_ok=True)

    # Spectral trajectory (same grid/time as training data), with timing
    t0_spectral = time.perf_counter()
    t_hist, u_hist, v_hist, xx, yy, _ = wave2d_main(
        NX=NX, NY=NY, Lx=Lx, Ly=Ly, dt=dt, TF=TF_compare, TSCREEN=TSCREEN,
        initial_condition=initial_condition, rng_seed=rng_seed,
    )
    spectral_total_time = time.perf_counter() - t0_spectral
    n_frames = u_hist.shape[2]
    spectral_time_per_frame = [spectral_total_time * k / max(1, n_frames) for k in range(n_frames)]

    # NN: load model (same arch as training: hidden_size, num_layers from checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N_i = 2 * patch_side ** 2
    N_o = 2 * nwd ** 2
    import sys
    sys.path.insert(0, os.path.join(_script_dir, "ML"))
    from nn_model import myNN
    from snapshot import load_checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    hidden_size = ckpt.get("hidden_size", 256)
    num_layers = ckpt.get("num_layers", 5)
    model = myNN(N_i, N_o, hidden_size=hidden_size, num_layers=num_layers).to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    # With njp=1: one NN step = one frame, so we run one step per frame and compare at all frames.
    u_nn = u_hist[:, :, 0].copy()
    v_nn = v_hist[:, :, 0].copy()
    nn_u_states = [u_nn.copy()]
    nn_v_states = [v_nn.copy()]
    nn_time_list = [0.0]
    t0_nn = time.perf_counter()
    for _ in range(n_frames - 1):
        u_nn, v_nn = integrate_nn_uv(model, u_nn, v_nn, nwd, nst, device)
        nn_u_states.append(u_nn.copy())
        nn_v_states.append(v_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)
    # Compare at every frame (nn_u_states[k] = NN state at frame k)
    nn_u_frames = nn_u_states
    nn_v_frames = nn_v_states

    all_nn_u = np.array(nn_u_frames)
    all_nn_v = np.array(nn_v_frames)
    u_min = min(u_hist.min(), all_nn_u.min())
    u_max = max(u_hist.max(), all_nn_u.max())
    v_min = min(v_hist.min(), all_nn_v.min())
    v_max = max(v_hist.max(), all_nn_v.max())
    pad_u = 0.05 * (u_max - u_min or 1)
    pad_v = 0.05 * (v_max - v_min or 1)
    clim_u = (u_min - pad_u, u_max + pad_u)
    clim_v = (v_min - pad_v, v_max + pad_v)

    for k in range(n_frames):
        t_val = t_hist[k]
        time_spec = spectral_time_per_frame[k]
        time_nn = nn_time_list[k]
        speedup = time_spec / time_nn if time_nn > 0 else float("inf")

        u_spec = u_hist[:, :, k]
        v_spec = v_hist[:, :, k]
        u_nn_k = nn_u_frames[k]
        v_nn_k = nn_v_frames[k]
        # Layout: row 0 = Spectral, row 1 = NN, row 2 = error |Spectral - NN|
        err_u = np.abs(u_spec - u_nn_k)
        err_v = np.abs(v_spec - v_nn_k)
        err_u_max = max(err_u.max(), 1e-12)
        err_v_max = max(err_v.max(), 1e-12)
        clim_err_u = (0, err_u_max)
        clim_err_v = (0, err_v_max)

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        fig.suptitle(f"IC: {initial_condition}  t = {t_val:.3f}", fontsize=12)
        # Row 0: Spectral
        ax = axes[0, 0]
        im = ax.pcolormesh(xx, yy, u_spec, shading="auto")
        ax.set_title("Spectral u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_u)
        plt.colorbar(im, ax=ax)
        ax = axes[0, 1]
        im = ax.pcolormesh(xx, yy, v_spec, shading="auto")
        ax.set_title("Spectral v")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_v)
        plt.colorbar(im, ax=ax)
        # Row 1: NN
        ax = axes[1, 0]
        im = ax.pcolormesh(xx, yy, u_nn_k, shading="auto")
        ax.set_title("NN u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_u)
        plt.colorbar(im, ax=ax)
        ax = axes[1, 1]
        im = ax.pcolormesh(xx, yy, v_nn_k, shading="auto")
        ax.set_title("NN v")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_v)
        plt.colorbar(im, ax=ax)
        # Row 2: Error |Spectral - NN|
        ax = axes[2, 0]
        im = ax.pcolormesh(xx, yy, err_u, shading="auto")
        ax.set_title("|Spectral u − NN u|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_err_u)
        plt.colorbar(im, ax=ax)
        ax = axes[2, 1]
        im = ax.pcolormesh(xx, yy, err_v, shading="auto")
        ax.set_title("|Spectral v − NN v|")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        im.set_clim(clim_err_v)
        plt.colorbar(im, ax=ax)
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
    print(f"Saved {n_frames} figures to {out_dir}/  (all frames; IC={initial_condition}, nwd={nwd}, nst={nst}, njp={njp})")
    print(f"  Spectral total: {spectral_total_time:.3f} s, NN total: {nn_time_list[-1]:.4f} s")


if __name__ == "__main__":
    main()
