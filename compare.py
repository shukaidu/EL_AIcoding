"""Compare: python compare.py --problem burgers_1d|wave_2d_linear|wave_2d_nonlinear"""
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

from ml.train_loop import get_device


def _speedup_str(time_ref, time_nn):
    if time_nn > 0 and time_ref >= 0:
        return f"{time_ref / time_nn:.1f}×"
    return "N/A (t=0)"


def _boundary_ext_2d_periodic(u, nst):
    u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
    return np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])

def _compare_burgers_1d(data_dir, out_dir):
    from pde.burgers_1d import integrate_burger, setup_burger
    import config.burgers_1d_config as cfg
    from ml.models import MLP
    from ml.snapshot import load_checkpoint

    L = cfg.L
    nx, dx, dt = cfg.nx, cfg.dx, cfg.dt
    njp, nst, nwd = cfg.njp, cfg.nst, cfg.nwd
    alpha, u_mean, nu = cfg.alpha, cfg.u_mean, cfg.nu
    dt_nn = njp * dt
    n_nn_steps = int(round(cfg.compare_t_end / dt_nn))

    u0, xc, A = setup_burger(nx, dx, dt, L, nu, alpha, u_mean, cfg.compare_seed)

    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem burgers_1d")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    N_i, N_o = 2 * nst + nwd, nwd
    model = MLP(
        N_i, N_o,
        hidden_size=ckpt.get("hidden_size", 256),
        num_layers=ckpt.get("num_layers", 6),
        activation=ckpt.get("activation", getattr(cfg, "activation", "relu")),
    ).to(device)
    load_checkpoint(model, model_path)
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

    u_fv = u0.copy()
    u_nn = u0.copy()
    fv_list = [u_fv.copy()]
    nn_list = [u_nn.copy()]
    fv_time_list = [0.0]
    nn_time_list = [0.0]
    fv_elapsed = nn_elapsed = 0.0
    for _ in range(n_nn_steps):
        t0 = time.perf_counter()
        for _ in range(njp):
            u_fv = integrate_burger(u_fv, dt, dx, nu, A=A)
        fv_elapsed += time.perf_counter() - t0
        fv_list.append(u_fv.copy())
        fv_time_list.append(fv_elapsed)

        t0 = time.perf_counter()
        u_nn = integrate_nn(model, u_nn, nwd, nst, device)
        nn_elapsed += time.perf_counter() - t0
        nn_list.append(u_nn.copy())
        nn_time_list.append(nn_elapsed)

    all_vals = fv_list + nn_list
    y_min = min(np.min(a) for a in all_vals)
    y_max = max(np.max(a) for a in all_vals)
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    plot_indices = np.linspace(0, len(nn_list) - 1, cfg.compare_n_times, dtype=int)
    os.makedirs(out_dir, exist_ok=True)
    for idx, i in enumerate(plot_indices):
        t_val = i * dt_nn
        speedup_str = _speedup_str(fv_time_list[i], nn_time_list[i])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xc, fv_list[i], label="FV", linewidth=1.5)
        ax.plot(xc, nn_list[i], label="NN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"t = {t_val:.2f}")
        ax.legend()
        ax.set_ylim(y_lim)
        ax.text(0.02, 0.98, f"FV runtime: {fv_time_list[i]:.3f} s\nNN runtime: {nn_time_list[i]:.4f} s\nNN speedup: {speedup_str}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=150)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  FV total: {fv_time_list[-1]:.3f} s  NN total: {nn_time_list[-1]:.4f} s")


def _compare_wave_2d_linear(data_dir, out_dir):
    from pde.wave_2d_linear import setup_wave2d, advance_tscreen
    import config.wave_2d_linear_config as cfg
    from ml.models import MLP
    from ml.snapshot import load_checkpoint

    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem wave_2d_linear")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    N_i = 2 * cfg.patch_side ** 2
    N_o = 2 * cfg.nwd ** 2
    model = MLP(
        N_i, N_o,
        hidden_size=ckpt.get("hidden_size", 256),
        num_layers=ckpt.get("num_layers", 5),
        activation=ckpt.get("activation", getattr(cfg, "activation", "relu")),
    ).to(device)
    load_checkpoint(model, model_path)
    model.eval()

    uhat, vhat, omega2, xx, yy, u0, v0 = setup_wave2d(
        cfg.NX, cfg.NY, cfg.Lx, cfg.Ly,
        c=cfg.c, initial_condition=cfg.compare_ic, rng_seed=cfg.compare_seed,
    )

    steps_per_nn = cfg.TSCREEN * cfg.njp
    n_nn_steps = int(round(cfg.compare_TF / (steps_per_nn * cfg.dt)))

    def one_nn_step(u2d, v2d, model, nwd, nst, patch_side, device):
        NX, NY = u2d.shape
        u_ext = _boundary_ext_2d_periodic(u2d, nst)
        v_ext = _boundary_ext_2d_periodic(v2d, nst)
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

    u_nn = u0.copy()
    v_nn = v0.copy()
    spec_list = [u0.copy()]
    spec_v_list = [v0.copy()]
    nn_frames_u = [u_nn.copy()]
    nn_frames_v = [v_nn.copy()]
    spec_time_list = [0.0]
    nn_time_list = [0.0]
    spec_elapsed = nn_elapsed = 0.0
    for _ in range(n_nn_steps):
        t0 = time.perf_counter()
        uhat, vhat, u, v = advance_tscreen(uhat, vhat, omega2, cfg.dt, steps_per_nn)
        spec_elapsed += time.perf_counter() - t0
        spec_list.append(u)
        spec_v_list.append(v)
        spec_time_list.append(spec_elapsed)

        t0 = time.perf_counter()
        u_nn, v_nn = one_nn_step(u_nn, v_nn, model, cfg.nwd, cfg.nst, cfg.patch_side, device)
        nn_elapsed += time.perf_counter() - t0
        nn_frames_u.append(u_nn.copy())
        nn_frames_v.append(v_nn.copy())
        nn_time_list.append(nn_elapsed)

    comp_names = ["u", "v"]
    clims = []
    for c_idx in range(2):
        lo = min(np.array(spec_list if c_idx == 0 else spec_v_list).min(),
                 np.array(nn_frames_u if c_idx == 0 else nn_frames_v).min())
        hi = max(np.array(spec_list if c_idx == 0 else spec_v_list).max(),
                 np.array(nn_frames_u if c_idx == 0 else nn_frames_v).max())
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    os.makedirs(out_dir, exist_ok=True)
    dt_nn = steps_per_nn * cfg.dt
    plot_indices = np.linspace(0, len(nn_frames_u) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        t_val = i * dt_nn
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for c_idx in range(2):
            spec_f = spec_list[i] if c_idx == 0 else spec_v_list[i]
            nn_f = nn_frames_u[i] if c_idx == 0 else nn_frames_v[i]
            pc0 = axes[0, c_idx].pcolormesh(xx, yy, spec_f.T, shading="auto")
            axes[0, c_idx].set_title(f"Spectral {comp_names[c_idx]}")
            pc1 = axes[1, c_idx].pcolormesh(xx, yy, nn_f.T, shading="auto")
            axes[1, c_idx].set_title(f"NN {comp_names[c_idx]}")
            pc0.set_clim(clims[c_idx])
            pc1.set_clim(clims[c_idx])
            fig.colorbar(pc0, ax=[axes[0, c_idx], axes[1, c_idx]], label=comp_names[c_idx])
        speedup_str = _speedup_str(spec_time_list[i], nn_time_list[i])
        fig.text(0.02, 0.98, f"Spectral: {spec_time_list[i]:.3f} s\nNN: {nn_time_list[i]:.4f} s\nSpeedup: {speedup_str}",
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.suptitle(f"t = {t_val:.3f}")
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=120)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  Spectral: {spec_time_list[-1]:.3f} s  NN: {nn_time_list[-1]:.4f} s")


def _compare_wave_2d_nonlinear(data_dir, out_dir):
    from pde.wave_2d_nonlinear import setup_wave2d_nonlinear, advance_tscreen
    import config.wave_2d_nonlinear_config as cfg
    from ml.models import CNN
    from ml.snapshot import load_checkpoint

    nwd = cfg.nwd
    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem wave_2d_nonlinear")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    nst = cfg.nst
    patch_side = cfg.patch_side
    base = ckpt.get("base", 32)
    model = CNN(Cin=3, Cout=3, base=base, Nx=patch_side, nx=nwd).to(device)
    load_checkpoint(model, model_path)
    model.eval()

    h, qx, qy, rhs, dt, xx, yy = setup_wave2d_nonlinear(
        cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
        g=cfg.g, h0=cfg.h0, f_coriolis=cfg.f_coriolis, nu_h=cfg.nu_h, nu_q=cfg.nu_q,
        initial_condition=cfg.compare_ic, rng_seed=cfg.compare_seed,
    )

    steps_per_nn = cfg.TSCREEN * cfg.njp
    n_nn_steps = int(round(cfg.compare_TF / (steps_per_nn * dt)))

    def integrate_nn_cnn(model, U0, nwd, nst, patch_side, device):
        nx, ny, _ = U0.shape
        U_nn = np.zeros_like(U0)
        u1 = _boundary_ext_2d_periodic(U0[:, :, 0], nst)
        u2 = _boundary_ext_2d_periodic(U0[:, :, 1], nst)
        u3 = _boundary_ext_2d_periodic(U0[:, :, 2], nst)
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

    # warm up spectral solver to T=warmup_T
    warmup_steps = int(round(cfg.warmup_T / (cfg.TSCREEN * dt)))
    for _ in range(warmup_steps):
        h, qx, qy = advance_tscreen(h, qx, qy, rhs, dt, cfg.TSCREEN)

    initial_frame = np.stack([h - cfg.h0, qx, qy], axis=-1)
    U_nn = initial_frame.copy()
    spec_frames = [initial_frame]
    nn_frames = [U_nn.copy()]
    spec_time_list = [0.0]
    nn_time_list = [0.0]
    spec_elapsed = nn_elapsed = 0.0
    for _ in range(n_nn_steps):
        t0 = time.perf_counter()
        h, qx, qy = advance_tscreen(h, qx, qy, rhs, dt, steps_per_nn)
        spec_elapsed += time.perf_counter() - t0
        spec_frames.append(np.stack([h - cfg.h0, qx, qy], axis=-1))
        spec_time_list.append(spec_elapsed)

        t0 = time.perf_counter()
        U_nn = integrate_nn_cnn(model, U_nn, nwd, nst, patch_side, device)
        nn_elapsed += time.perf_counter() - t0
        nn_frames.append(U_nn.copy())
        nn_time_list.append(nn_elapsed)

    err_per_frame = [np.abs(spec_frames[i] - nn_frames[i]).mean() for i in range(len(nn_frames))]
    print(f"  L1 error mean: {np.mean(err_per_frame):.6f}, max frame: {np.max(err_per_frame):.6f}")

    comp_names = ["h - h0", "qx", "qy"]
    clims = []
    for c in range(3):
        lo = min(np.array([f[:, :, c].min() for f in spec_frames]).min(),
                 np.array([f[:, :, c].min() for f in nn_frames]).min())
        hi = max(np.array([f[:, :, c].max() for f in spec_frames]).max(),
                 np.array([f[:, :, c].max() for f in nn_frames]).max())
        pad = 0.05 * (hi - lo or 1)
        clims.append((lo - pad, hi + pad))

    os.makedirs(out_dir, exist_ok=True)
    dt_nn = steps_per_nn * dt
    plot_indices = np.linspace(0, len(nn_frames) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        t_val = i * dt_nn
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for c in range(3):
            pc0 = axes[0, c].pcolormesh(xx, yy, spec_frames[i][:, :, c].T, shading="auto")
            axes[0, c].set_title(f"Spectral {comp_names[c]}")
            pc1 = axes[1, c].pcolormesh(xx, yy, nn_frames[i][:, :, c].T, shading="auto")
            axes[1, c].set_title(f"NN {comp_names[c]}")
            pc0.set_clim(clims[c])
            pc1.set_clim(clims[c])
            fig.colorbar(pc0, ax=[axes[0, c], axes[1, c]], label=comp_names[c])
        speedup_str = _speedup_str(spec_time_list[i], nn_time_list[i])
        fig.text(0.02, 0.98, f"Spectral: {spec_time_list[i]:.3f} s\nNN: {nn_time_list[i]:.4f} s\nSpeedup: {speedup_str}",
                 fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.suptitle(f"t = {t_val:.3f}")
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=120)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  Spectral: {spec_time_list[-1]:.3f} s  NN: {nn_time_list[-1]:.4f} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    problem = p.parse_args().problem
    data_dir = os.path.join(_repo_root, "data", problem)
    out_dir = os.path.join(data_dir, "compare")
    fns = {"burgers_1d": _compare_burgers_1d, "wave_2d_linear": _compare_wave_2d_linear, "wave_2d_nonlinear": _compare_wave_2d_nonlinear}
    fns[problem](data_dir, out_dir)


if __name__ == "__main__":
    main()


