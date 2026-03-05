"""
Single comparison entry. Run from repo root:
  python compare.py --problem burgers_1d
  python compare.py --problem wave_2d_linear
  python compare.py --problem wave_2d_nonlinear
"""
import os
import sys
import re
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)


def _compare_burgers_1d(data_dir, out_dir):
    from pde.burgers_1d import gen_dist, build_diffusion_matrix, integrate_burger
    import config.burgers_1d_config as cfg
    from common.models import MLP
    from ml.snapshot import load_checkpoint

    L = cfg.L
    nx, dx, dt = cfg.nx, cfg.dx, cfg.dt
    njp, nst, nwd = cfg.njp, cfg.nst, cfg.nwd
    alpha, u_mean, nu = cfg.alpha, cfg.u_mean, cfg.nu
    times = list(np.linspace(0, 2, 10))
    n_times = len(times)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0
    np.random.seed(42)
    u0, _ = gen_dist(nx, alpha)
    u0 = u0 + u_mean

    u_fv = u0.copy()
    A = build_diffusion_matrix(nx, dt, dx, nu)
    t_fv = 0.0
    fv_list = [u_fv.copy()] + [None] * (n_times - 1)
    fv_time_list = [0.0] * n_times
    next_time_idx = 1
    t0_fv = time.perf_counter()
    nt_needed = int(np.ceil(2.0 / dt))
    for _ in range(nt_needed):
        u_fv = integrate_burger(u_fv, dt, dx, nu, A=A)
        t_fv += dt
        while next_time_idx < n_times and t_fv >= times[next_time_idx] - 0.5 * dt:
            fv_time_list[next_time_idx] = time.perf_counter() - t0_fv
            fv_list[next_time_idx] = u_fv.copy()
            next_time_idx += 1
        if next_time_idx >= n_times:
            break

    dt_nn = njp * dt
    n_nn_steps = int(round(2.0 / dt_nn))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem burgers_1d")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    hidden_size = ckpt.get("hidden_size", 256)
    N_i, N_o = 2 * nst + nwd, nwd
    model = MLP(N_i, N_o, hidden_size=hidden_size, num_layers=ckpt.get("num_hidden_layers", 6), activation="relu").to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    def integrate_nn(model, u, nwd, nst, device):
        unn_ext = np.concatenate([u[-nst:], u, u[:nst]])
        u_nn = np.zeros_like(u)
        for i in range(nx // nwd):
            patch = unn_ext[i * nwd : i * nwd + 2 * nst + nwd].astype(np.float32)
            patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(patch_t)
            u_nn[i * nwd : (i + 1) * nwd] = out.cpu().numpy().ravel()
        return u_nn

    u_nn = u0.copy()
    nn_list = [u_nn.copy()]
    nn_time_list = [0.0]
    t0_nn = time.perf_counter()
    for _ in range(n_nn_steps):
        u_nn = integrate_nn(model, u_nn, nwd, nst, device)
        nn_list.append(u_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)

    all_vals = [a for a in fv_list + nn_list if a is not None]
    y_min = min(np.min(a) for a in all_vals)
    y_max = max(np.max(a) for a in all_vals)
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_times):
        u_fv_t, u_nn_t = fv_list[i], None
        nn_idx = min(round(times[i] / dt_nn), len(nn_list) - 1)
        u_nn_t = nn_list[nn_idx] if nn_idx < len(nn_list) else None
        if u_fv_t is None or u_nn_t is None:
            continue
        time_fv = fv_time_list[i]
        time_nn = nn_time_list[nn_idx]
        speedup = time_fv / time_nn if time_nn > 0 else float("inf")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xc, u_fv_t, label="FV", linewidth=1.5)
        ax.plot(xc, u_nn_t, label="NN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"t = {times[i]:.2f}")
        ax.legend()
        ax.set_ylim(y_lim)
        ax.text(0.02, 0.98, f"FV runtime: {time_fv:.3f} s\nNN runtime: {time_nn:.4f} s\nNN speedup: {speedup:.1f}×",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{i}.png"), dpi=150)
        plt.close()
    print(f"Saved {n_times} figures to {out_dir}/  FV total: {fv_time_list[-1]:.3f} s  NN total: {nn_time_list[-1]:.4f} s")


def _compare_wave_2d_linear(data_dir, out_dir):
    from pde.wave_2d_linear import wave2d_main
    import config.wave_2d_linear_config as cfg
    from common.models import MLP
    from ml.snapshot import load_checkpoint

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem wave_2d_linear")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    N_i = 2 * cfg.patch_side ** 2
    N_o = 2 * cfg.nwd ** 2
    model = MLP(N_i, N_o, hidden_size=ckpt.get("hidden_size", 256), num_layers=ckpt.get("num_layers", 5), activation="identity").to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    t0_spec = time.perf_counter()
    t_hist, u_hist, v_hist, xx, yy, _ = wave2d_main(
        cfg.NX, cfg.NY, cfg.Lx, cfg.Ly, cfg.dt, 1.0, cfg.TSCREEN,
        c=cfg.c, initial_condition="random_white", rng_seed=42,
    )
    spectral_time = time.perf_counter() - t0_spec
    n_frames = u_hist.shape[2]
    spectral_time_per_frame = [spectral_time * k / max(1, n_frames) for k in range(n_frames)]

    def boundary_ext(u, nst):
        u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
        return np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])

    def one_nn_step(u2d, v2d, model, nwd, nst, patch_side, device):
        NX, NY = u2d.shape
        u_ext = boundary_ext(u2d, nst)
        v_ext = boundary_ext(v2d, nst)
        u_nn = np.zeros_like(u2d)
        v_nn = np.zeros_like(v2d)
        for ii in range(NX // nwd):
            for jj in range(NY // nwd):
                ind1 = slice(ii * nwd, ii * nwd + patch_side)
                ind2 = slice(jj * nwd, jj * nwd + patch_side)
                inp = np.stack([u_ext[ind1, ind2], v_ext[ind1, ind2]], axis=0).astype(np.float32)
                inp = inp.reshape(-1)  # (2*patch_side^2,)
                inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                out = out.cpu().numpy().squeeze(0)
                n = cfg.nwd ** 2
                u_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[:n].reshape(cfg.nwd, cfg.nwd)
                v_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[n:].reshape(cfg.nwd, cfg.nwd)
        return u_nn, v_nn

    u_nn = u_hist[:, :, 0].copy()
    v_nn = v_hist[:, :, 0].copy()
    nn_frames_u = [u_nn.copy()]
    nn_frames_v = [v_nn.copy()]
    nn_time_list = [0.0]
    t0_nn = time.perf_counter()
    for _ in range(n_frames - 1):
        u_nn, v_nn = one_nn_step(u_nn, v_nn, model, cfg.nwd, cfg.nst, cfg.patch_side, device)
        nn_frames_u.append(u_nn.copy())
        nn_frames_v.append(v_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)

    comp_names = ["u", "v"]
    clims = []
    for c, name in enumerate(comp_names):
        spec_u = u_hist if c == 0 else v_hist
        nn_u = nn_frames_u if c == 0 else nn_frames_v
        lo = min(spec_u.min(), np.array(nn_u).min())
        hi = max(spec_u.max(), np.array(nn_u).max())
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    os.makedirs(out_dir, exist_ok=True)
    for k in range(n_frames):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for c in range(2):
            spec_f = u_hist[:, :, k] if c == 0 else v_hist[:, :, k]
            nn_f = nn_frames_u[k] if c == 0 else nn_frames_v[k]
            pc0 = axes[0, c].pcolormesh(xx, yy, spec_f.T, shading="auto")
            axes[0, c].set_title(f"Spectral {comp_names[c]}")
            pc1 = axes[1, c].pcolormesh(xx, yy, nn_f.T, shading="auto")
            axes[1, c].set_title(f"NN {comp_names[c]}")
            pc0.set_clim(clims[c])
            pc1.set_clim(clims[c])
        time_spec = spectral_time_per_frame[k]
        time_nn = nn_time_list[k]
        speedup = time_spec / time_nn if time_nn > 0 else float("inf")
        fig.text(0.02, 0.98, f"Spectral: {time_spec:.3f} s\nNN: {time_nn:.4f} s\nSpeedup: {speedup:.1f}×",
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.suptitle(f"t = {t_hist[k]:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{k}.png"), dpi=120)
        plt.close()
    print(f"Saved {n_frames} figures to {out_dir}/  Spectral: {spectral_time:.3f} s  NN: {nn_time_list[-1]:.4f} s")


def _compare_wave_2d_nonlinear(data_dir, out_dir):
    from pde.wave_2d_nonlinear import wave2d_spectral
    import config.wave_2d_nonlinear_config as cfg
    from common.models import ShrinkCNN
    from ml.snapshot import load_checkpoint

    nwd = cfg.nwd
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem wave_2d_nonlinear")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    max_idx = max(int(re.match(r"net\.(\d+)\.", k).group(1)) for k in sd if re.match(r"net\.(\d+)\.", k))
    patch_side = nwd + 4 * max_idx
    nst = (patch_side - nwd) // 2
    base = ckpt.get("base", 32)
    model = ShrinkCNN(Cin=3, Cout=3, base=base, Nx=patch_side, nx=nwd).to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    def boundary_ext(u, nst):
        u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
        return np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])

    def integrate_nn_cnn(model, U0, nwd, nst, patch_side, device):
        nx, ny, _ = U0.shape
        U_nn = np.zeros_like(U0)
        u1 = boundary_ext(U0[:, :, 0], nst)
        u2 = boundary_ext(U0[:, :, 1], nst)
        u3 = boundary_ext(U0[:, :, 2], nst)
        for ii in range(nx // nwd):
            for jj in range(ny // nwd):
                ind1 = slice(ii * nwd, ii * nwd + patch_side)
                ind2 = slice(jj * nwd, jj * nwd + patch_side)
                tmp = np.stack([u1[ind1, ind2], u2[ind1, ind2], u3[ind1, ind2]], axis=0)
                inp = torch.tensor(tmp, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                out = out.cpu().numpy().squeeze(0)
                U_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd, :] = np.transpose(out, (1, 2, 0))
        return U_nn

    t0_spectral = time.perf_counter()
    t_hist, U_hist, xx, yy, _, _, _, _ = wave2d_spectral(
        cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, 1.0, cfg.TSCREEN,
        g=cfg.g, h0=cfg.h0, frot0=cfg.frot0, nu_h=cfg.nu_h, nu_q=cfg.nu_q,
        initial_condition="random", rng_seed=42,
    )
    spectral_time = time.perf_counter() - t0_spectral
    n_frames = U_hist.shape[3]
    spectral_time_per_frame = [spectral_time * k / max(1, n_frames) for k in range(n_frames)]

    U_nn = U_hist[:, :, :, 0].copy()
    nn_frames = [U_nn.copy()]
    nn_time_list = [0.0]
    t0_nn = time.perf_counter()
    for _ in range(n_frames - 1):
        U_nn = integrate_nn_cnn(model, U_nn, nwd, nst, patch_side, device)
        nn_frames.append(U_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)

    err_per_frame = [np.abs(U_hist[:, :, :, k] - nn_frames[k]).mean() for k in range(n_frames)]
    print(f"  L1 error mean: {np.mean(err_per_frame):.6f}, max frame: {np.max(err_per_frame):.6f}")

    comp_names = ["h - h0", "qx", "qy"]
    clims = []
    for c in range(3):
        lo = min(U_hist[:, :, c, :].min(), np.array([f[:, :, c].min() for f in nn_frames]).min())
        hi = max(U_hist[:, :, c, :].max(), np.array([f[:, :, c].max() for f in nn_frames]).max())
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    os.makedirs(out_dir, exist_ok=True)
    for k in range(n_frames):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for c in range(3):
            pc0 = axes[0, c].pcolormesh(xx, yy, U_hist[:, :, c, k].T, shading="auto")
            axes[0, c].set_title(f"Spectral {comp_names[c]}")
            pc1 = axes[1, c].pcolormesh(xx, yy, nn_frames[k][:, :, c].T, shading="auto")
            axes[1, c].set_title(f"NN {comp_names[c]}")
            pc0.set_clim(clims[c])
            pc1.set_clim(clims[c])
        time_spec = spectral_time_per_frame[k]
        time_nn = nn_time_list[k]
        speedup = time_spec / time_nn if time_nn > 0 else float("inf")
        fig.text(0.02, 0.98, f"Spectral: {time_spec:.3f} s\nNN: {time_nn:.4f} s\nSpeedup: {speedup:.1f}×",
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.suptitle(f"t = {t_hist[k]:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{k}.png"), dpi=120)
        plt.close()
    print(f"Saved {n_frames} figures to {out_dir}/  Spectral: {spectral_time:.3f} s  NN: {nn_time_list[-1]:.4f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    args = parser.parse_args()
    problem = args.problem
    data_dir = os.path.join(_repo_root, "data", problem)
    out_dir = os.path.join(data_dir, "compare")
    if problem == "burgers_1d":
        _compare_burgers_1d(data_dir, out_dir)
    elif problem == "wave_2d_linear":
        _compare_wave_2d_linear(data_dir, out_dir)
    else:
        _compare_wave_2d_nonlinear(data_dir, out_dir)


if __name__ == "__main__":
    main()
