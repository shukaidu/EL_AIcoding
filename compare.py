"""Compare: python compare.py --problem burgers_1d|wave_2d_linear|wave_2d_nonlinear"""
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)

from ml.train_loop import get_device


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _speedup_str(time_ref, time_nn):
    if time_nn > 0 and time_ref >= 0:
        return f"{time_ref / time_nn:.1f}×"
    return "N/A (t=0)"


def _boundary_ext_2d_periodic(u, nst):
    u_ext = np.hstack([u[:, -nst:], u, u[:, :nst]])
    return np.vstack([u_ext[-nst:], u_ext, u_ext[:nst]])


def _timing_annotation(ax_or_fig, ref_t, nn_t, use_fig=False):
    """在图上添加 runtime / speedup 文字框。"""
    txt = f"Ref: {ref_t:.3f} s\nNN: {nn_t:.4f} s\nSpeedup: {_speedup_str(ref_t, nn_t)}"
    kw = dict(fontsize=9, verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    extra = {} if use_fig else {"transform": ax_or_fig.transAxes}
    ax_or_fig.text(0.02, 0.98, txt, **kw, **extra)


def _load_mlp(model_path, N_i, N_o, cfg, device):
    """加载 MLP checkpoint，返回已 eval 的 model。"""
    from ml.models import MLP
    from ml.snapshot import load_checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    model = MLP(
        N_i, N_o,
        hidden_size=ckpt.get("hidden_size", 256),
        num_layers=ckpt.get("num_layers", 6),
        activation=ckpt.get("activation", getattr(cfg, "activation", "relu")),
    ).to(device)
    load_checkpoint(model, model_path)
    model.eval()
    return model


def _check_model_file(model_path, problem):
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}. Run: python -m ml.train --problem {problem}")
        return False
    return True


# ---------------------------------------------------------------------------
# Burgers 1D
# ---------------------------------------------------------------------------

def _integrate_nn_burgers(model, u, nx, nwd, nst, device):
    """Run one NN step for 1D Burgers (sliding window over tiles)."""
    u_ext = np.concatenate([u[-nst:], u, u[:nst]])
    u_out = np.zeros_like(u)
    for i in range(nx // nwd):
        patch = u_ext[i * nwd : i * nwd + 2 * nst + nwd].astype(np.float32)
        inp = torch.tensor(patch).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)
        u_out[i * nwd : (i + 1) * nwd] = out.cpu().numpy().ravel()
    return u_out


def _compare_burgers_1d(data_dir, out_dir):
    from pde.burgers_1d import integrate_burger, setup_burger
    import config.burgers_1d_config as cfg

    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not _check_model_file(model_path, "burgers_1d"):
        return

    nx, dx, dt = cfg.nx, cfg.dx, cfg.dt
    njp, nst, nwd = cfg.njp, cfg.nst, cfg.nwd
    dt_nn = njp * cfg.TSCREEN * dt
    n_nn_steps = int(round(cfg.compare_t_end / dt_nn))

    u0, xc, A = setup_burger(nx, dx, dt, cfg.L, cfg.nu, cfg.alpha, cfg.u_mean, cfg.compare_seed)
    model = _load_mlp(model_path, 2 * nst + nwd, nwd, cfg, device)

    u_fv, u_nn = u0.copy(), u0.copy()
    fv_list, nn_list = [u_fv.copy()], [u_nn.copy()]
    fv_times, nn_times = [0.0], [0.0]
    fv_elapsed = nn_elapsed = 0.0

    for _ in tqdm(range(n_nn_steps), desc="burgers_1d rollout"):
        t0 = time.perf_counter()
        for _ in range(njp * cfg.TSCREEN):
            u_fv = integrate_burger(u_fv, dt, dx, cfg.nu, A=A)
        fv_elapsed += time.perf_counter() - t0
        fv_list.append(u_fv.copy())
        fv_times.append(fv_elapsed)

        t0 = time.perf_counter()
        u_nn = _integrate_nn_burgers(model, u_nn, nx, nwd, nst, device)
        nn_elapsed += time.perf_counter() - t0
        nn_list.append(u_nn.copy())
        nn_times.append(nn_elapsed)

    all_vals = fv_list + nn_list
    y_min = min(np.min(a) for a in all_vals)
    y_max = max(np.max(a) for a in all_vals)
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    os.makedirs(out_dir, exist_ok=True)
    plot_indices = np.linspace(0, len(nn_list) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xc, fv_list[i], label="FV", linewidth=1.5)
        ax.plot(xc, nn_list[i], label="NN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"t = {i * dt_nn:.2f}")
        ax.legend()
        ax.set_ylim(y_lim)
        _timing_annotation(ax, fv_times[i], nn_times[i])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=150)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  FV total: {fv_times[-1]:.3f} s  NN total: {nn_times[-1]:.4f} s")


# ---------------------------------------------------------------------------
# Wave 2D linear
# ---------------------------------------------------------------------------

def _integrate_nn_wave2d(u2d, v2d, model, nwd, nst, patch_side, device):
    """Run one NN step for 2D linear wave (tiled over patches)."""
    NX, NY = u2d.shape
    u_ext = _boundary_ext_2d_periodic(u2d, nst)
    v_ext = _boundary_ext_2d_periodic(v2d, nst)
    u_nn = np.zeros_like(u2d)
    v_nn = np.zeros_like(v2d)
    n = nwd ** 2
    for ii in range(NX // nwd):
        for jj in range(NY // nwd):
            s1 = slice(ii * nwd, ii * nwd + patch_side)
            s2 = slice(jj * nwd, jj * nwd + patch_side)
            inp = np.stack([u_ext[s1, s2], v_ext[s1, s2]], axis=0).reshape(-1).astype(np.float32)
            inp = torch.tensor(inp).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp).cpu().numpy().squeeze(0)
            u_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[:n].reshape(nwd, nwd)
            v_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd] = out[n:].reshape(nwd, nwd)
    return u_nn, v_nn


def _compare_wave_2d_linear(data_dir, out_dir):
    from pde.wave_2d_linear import setup_wave2d, advance_tscreen
    import config.wave_2d_linear_config as cfg

    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    if not _check_model_file(model_path, "wave_2d_linear"):
        return

    model = _load_mlp(model_path, 2 * cfg.patch_side ** 2, 2 * cfg.nwd ** 2, cfg, device)
    uhat, vhat, omega2, xx, yy, u0, v0 = setup_wave2d(
        cfg.NX, cfg.NY, cfg.Lx, cfg.Ly,
        c=cfg.c, initial_condition=cfg.compare_ic, rng_seed=cfg.compare_seed,
    )

    steps_per_nn = cfg.TSCREEN * cfg.njp
    dt_nn = steps_per_nn * cfg.dt
    n_nn_steps = int(round(cfg.compare_TF / dt_nn))

    u_nn, v_nn = u0.copy(), v0.copy()
    spec_u, spec_v = [u0.copy()], [v0.copy()]
    nn_u, nn_v = [u_nn.copy()], [v_nn.copy()]
    spec_times, nn_times = [0.0], [0.0]
    spec_elapsed = nn_elapsed = 0.0

    for _ in tqdm(range(n_nn_steps), desc="wave_2d_linear rollout"):
        t0 = time.perf_counter()
        uhat, vhat, u, v = advance_tscreen(uhat, vhat, omega2, cfg.dt, steps_per_nn)
        spec_elapsed += time.perf_counter() - t0
        spec_u.append(u)
        spec_v.append(v)
        spec_times.append(spec_elapsed)

        t0 = time.perf_counter()
        u_nn, v_nn = _integrate_nn_wave2d(u_nn, v_nn, model, cfg.nwd, cfg.nst, cfg.patch_side, device)
        nn_elapsed += time.perf_counter() - t0
        nn_u.append(u_nn.copy())
        nn_v.append(v_nn.copy())
        nn_times.append(nn_elapsed)

    # Compute color limits from spectral reference
    clims = [
        _symm_clim(spec_u),
        _symm_clim(spec_v),
    ]
    os.makedirs(out_dir, exist_ok=True)
    plot_indices = np.linspace(0, len(nn_u) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for c_idx, (s_frames, n_frames, name) in enumerate(zip([spec_u, spec_v], [nn_u, nn_v], ["u", "v"])):
            _pcolor_row(fig, axes[:, c_idx], xx, yy, s_frames[i], n_frames[i], name, clims[c_idx])
        _timing_annotation(fig, spec_times[i], nn_times[i], use_fig=True)
        plt.suptitle(f"t = {i * dt_nn:.3f}")
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=120)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  Spectral: {spec_times[-1]:.3f} s  NN: {nn_times[-1]:.4f} s")


# ---------------------------------------------------------------------------
# Wave 2D nonlinear
# ---------------------------------------------------------------------------

def _integrate_nn_cnn(model, U0, nwd, nst, patch_side, device, ch_mean=None, ch_std=None, residual=False):
    """Run one NN step for 2D nonlinear shallow water (CNN/UNet, tiled)."""
    nx, ny, _ = U0.shape
    u_exts = [_boundary_ext_2d_periodic(U0[:, :, c], nst) for c in range(3)]
    U_nn = np.zeros_like(U0)

    normalise = ch_mean is not None
    if normalise:
        mean_t = torch.tensor(ch_mean, dtype=torch.float32).view(1, -1, 1, 1).to(device)
        std_t = torch.tensor(ch_std, dtype=torch.float32).view(1, -1, 1, 1).to(device)

    for ii in range(nx // nwd):
        for jj in range(ny // nwd):
            s1 = slice(ii * nwd, ii * nwd + patch_side)
            s2 = slice(jj * nwd, jj * nwd + patch_side)
            patch = np.stack([e[s1, s2] for e in u_exts], axis=0)
            inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
            inp_norm = (inp - mean_t) / std_t if normalise else inp
            with torch.no_grad():
                out_norm = model(inp_norm)
            if residual:
                out_norm = inp_norm[:, :, nst:nst + nwd, nst:nst + nwd] + out_norm
            out = out_norm * std_t + mean_t if normalise else out_norm
            U_nn[ii * nwd : (ii + 1) * nwd, jj * nwd : (jj + 1) * nwd, :] = \
                np.transpose(out.cpu().numpy().squeeze(0), (1, 2, 0))
    return U_nn


def _compare_wave_2d_nonlinear(data_dir, out_dir, model=None):
    from pde.wave_2d_nonlinear import setup_wave2d_nonlinear
    import config.wave_2d_nonlinear_config as cfg
    from ml.models import CNN, UNet
    from ml.snapshot import load_checkpoint

    device = get_device()
    model_path = os.path.join(data_dir, cfg.model_pth)
    ch_mean = ch_std = None
    residual = False

    if model is None:
        if not _check_model_file(model_path, "wave_2d_nonlinear"):
            return
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        base = ckpt.get("base", 32)
        model_type = ckpt.get("model_type", getattr(cfg, "model_type", "cnn")).lower()
        if model_type == "unet":
            pooling = ckpt.get("pooling", getattr(cfg, "pooling", "max"))
            model = UNet(Cin=3, Cout=3, base=base, Nx=cfg.patch_side, nx=cfg.nwd, pooling=pooling).to(device)
        else:
            model = CNN(Cin=3, Cout=3, base=base, Nx=cfg.patch_side, nx=cfg.nwd).to(device)
        load_checkpoint(model, model_path)
        model.eval()
        ch_mean = ckpt.get("ch_mean")
        ch_std = ckpt.get("ch_std")
        residual = ckpt.get("residual", False)

    h, qx, qy, _, advance_fn, dt, xx, yy = setup_wave2d_nonlinear(
        cfg.Lx, cfg.Ly, cfg.nx, cfg.ny,
        g=cfg.g, h0=cfg.h0, f_coriolis=cfg.f_coriolis, nu_h=cfg.nu_h, nu_q=cfg.nu_q,
        nudging_coeff=cfg.nudging_coeff,
        initial_condition=cfg.compare_ic, rng_seed=cfg.compare_seed,
        integrator=cfg.integrator, dt=cfg.dt_internal,
    )

    steps_per_nn = cfg.TSCREEN * cfg.njp
    dt_nn = steps_per_nn * dt
    n_nn_steps = int(round(cfg.compare_TF / dt_nn))

    # Warm up spectral solver
    warmup_steps = int(round(cfg.warmup_T / (cfg.TSCREEN * dt)))
    for _ in range(warmup_steps):
        h, qx, qy = advance_fn(h, qx, qy, dt, cfg.TSCREEN)

    initial_frame = np.stack([h - cfg.h0, qx, qy], axis=-1)
    U_nn = initial_frame.copy()
    spec_frames = [initial_frame]
    nn_frames = [U_nn.copy()]
    spec_times, nn_times = [0.0], [0.0]
    spec_elapsed = nn_elapsed = 0.0

    for _ in tqdm(range(n_nn_steps), desc="wave_2d_nonlinear rollout"):
        t0 = time.perf_counter()
        h, qx, qy = advance_fn(h, qx, qy, dt, steps_per_nn)
        spec_elapsed += time.perf_counter() - t0
        spec_frames.append(np.stack([h - cfg.h0, qx, qy], axis=-1))
        spec_times.append(spec_elapsed)

        t0 = time.perf_counter()
        U_nn = _integrate_nn_cnn(model, U_nn, cfg.nwd, cfg.nst, cfg.patch_side, device,
                                  ch_mean=ch_mean, ch_std=ch_std, residual=residual)
        nn_elapsed += time.perf_counter() - t0
        nn_frames.append(U_nn.copy())
        nn_times.append(nn_elapsed)

    # Error summary
    err_per_frame = [np.abs(spec_frames[i] - nn_frames[i]).mean() for i in range(len(nn_frames))]
    print(f"  L1 error mean: {np.mean(err_per_frame):.6f}, max frame: {np.max(err_per_frame):.6f}")
    ch_names = ["h-h0", "qx", "qy"]
    for c, name in enumerate(ch_names):
        ch_err = np.mean([np.abs(spec_frames[i][:, :, c] - nn_frames[i][:, :, c]).mean()
                          for i in range(len(nn_frames))])
        ch_scale = np.mean([np.abs(spec_frames[i][:, :, c]).mean() for i in range(1, len(spec_frames))])
        print(f"    {name}: abs={ch_err:.6f}, scale={ch_scale:.6f}, rel={ch_err / max(ch_scale, 1e-10):.4f}")

    comp_names = ["h - h0", "qx", "qy"]
    clims = [_symm_clim([f[:, :, c] for f in spec_frames]) for c in range(3)]

    os.makedirs(out_dir, exist_ok=True)
    plot_indices = np.linspace(0, len(nn_frames) - 1, cfg.compare_n_times, dtype=int)
    for idx, i in enumerate(plot_indices):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for c in range(3):
            _pcolor_row(fig, axes[:, c], xx, yy,
                        spec_frames[i][:, :, c], nn_frames[i][:, :, c], comp_names[c], clims[c])
        _timing_annotation(fig, spec_times[i], nn_times[i], use_fig=True)
        plt.suptitle(f"t = {i * dt_nn:.3f}")
        plt.savefig(os.path.join(out_dir, f"t{idx}.png"), dpi=120)
        plt.close()
    print(f"Saved {len(plot_indices)} figures to {out_dir}/  Spectral: {spec_times[-1]:.3f} s  NN: {nn_times[-1]:.4f} s")
    return float(np.mean(err_per_frame)), float(np.max(err_per_frame)), spec_elapsed, nn_elapsed


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _symm_clim(frames):
    """Symmetric color limits [-vmax, vmax] from a list of 2D arrays."""
    vmax = max(np.abs(f).max() for f in frames)
    return (-max(vmax, 1e-6), max(vmax, 1e-6))


def _pcolor_row(fig, ax_pair, xx, yy, spec_f, nn_f, name, clim):
    """Fill a (spectral, NN) pair of pcolormesh axes with shared colorbar."""
    pc0 = ax_pair[0].pcolormesh(xx, yy, spec_f.T, shading="auto", cmap="RdBu_r")
    ax_pair[0].set_title(f"Spectral {name}")
    pc1 = ax_pair[1].pcolormesh(xx, yy, nn_f.T, shading="auto", cmap="RdBu_r")
    ax_pair[1].set_title(f"NN {name}")
    for pc in (pc0, pc1):
        pc.set_clim(clim)
    fig.colorbar(pc0, ax=list(ax_pair), label=name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    problem = p.parse_args().problem
    data_dir = os.path.join(_repo_root, "data", problem)
    out_dir = os.path.join(data_dir, "compare")
    fns = {
        "burgers_1d": _compare_burgers_1d,
        "wave_2d_linear": _compare_wave_2d_linear,
        "wave_2d_nonlinear": _compare_wave_2d_nonlinear,
    }
    fns[problem](data_dir, out_dir)


if __name__ == "__main__":
    main()
