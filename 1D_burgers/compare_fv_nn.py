"""
Compare FV (traditional) and NN solutions from t=0 to t=2 in steps of 0.1.
Generates one figure per time: both solutions on the same plot, with runtime comparison.
Run after: python gen_data.py && python ml/train.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pde import (
    set_param,
    gen_dist,
    build_diffusion_matrix,
    integrate_burger,
)

# Import model from ml package (run from project root)
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "ml"))
from nn_model import myNN
from snapshot import load_checkpoint


def integrate_burger_nn(model, u, nwd, nst, device):
    """Advance full state u by one NN macro step (periodic BCs)."""
    nx = u.size
    unn_ext = np.concatenate([u[-nst:], u, u[:nst]])
    u_nn = np.zeros_like(u)
    for i in range(nx // nwd):
        patch = unn_ext[i * nwd : i * nwd + 2 * nst + nwd].astype(np.float32)
        patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(patch_t)
        u_nn[i * nwd : (i + 1) * nwd] = out.cpu().numpy().ravel()
    return u_nn


def main(output_dir="compare_fv_nn"):
    prm = set_param()
    L = prm["L"]
    nx = prm["nx"]
    dx = prm["dx"]
    dt = prm["dt"]
    njp = prm["njp"]
    nst = prm["nst"]
    nwd = prm["nwd"]
    alpha = prm["alpha"]
    u_mean = prm["u_mean"]
    nu = prm["nu"]

    # Times to record: t=0 to t=2, 共 10 个时刻
    times = list(np.linspace(0, 2, 10))
    n_times = len(times)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0

    # Same initial condition for both（固定种子便于复现）
    np.random.seed(42)
    u0, _ = gen_dist(nx, alpha)
    u0 = u0 + u_mean

    # --- FV: run to t=2 and record at each time，并记录累计运行时间 ---
    u_fv = u0.copy()
    A = build_diffusion_matrix(nx, dt, dx, nu)
    t_fv = 0.0
    fv_list = [None] * n_times
    fv_time_list = [0.0] * n_times  # 到达该时刻的累计 CPU 时间 (s)
    fv_list[0] = u_fv.copy()
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

    # --- NN: 按 prm 计算每步时间与步数，加载与训练时一致的网络结构 ---
    dt_nn = njp * dt
    n_nn_steps = int(round(2.0 / dt_nn))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(_script_dir, "ml", "data_res_model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Train the model first: run gen_data.py then python ml/train.py. Expected: {model_path}"
        )
    N_i = 2 * nst + nwd
    N_o = nwd
    ckpt = torch.load(model_path, map_location="cpu")
    hidden_size = ckpt.get("hidden_size", 256)
    num_hidden_layers = ckpt.get("num_hidden_layers", 6)
    model = myNN(
        N_i, N_o,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    ).to(device)
    load_checkpoint(model, None, model_path)
    model.eval()

    u_nn = u0.copy()
    nn_list = [u_nn.copy()]
    nn_time_list = [0.0]  # 到达该步的累计运行时间 (s)
    t0_nn = time.perf_counter()
    for _ in range(n_nn_steps):
        u_nn = integrate_burger_nn(model, u_nn, nwd, nst, device)
        nn_list.append(u_nn.copy())
        nn_time_list.append(time.perf_counter() - t0_nn)

    # --- Global y-axis limits (same for all plots) ---
    all_vals = fv_list + nn_list
    y_min = min(np.min(a) for a in all_vals if a is not None)
    y_max = max(np.max(a) for a in all_vals if a is not None)
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_lim = (y_min - pad, y_max + pad)

    # --- One figure per time (共 10 张)，图中标注 FV 与 NN 运行时间 ---
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_times):
        u_fv_t = fv_list[i]
        nn_idx = min(round(times[i] / dt_nn), len(nn_list) - 1)
        u_nn_t = nn_list[nn_idx] if nn_idx < len(nn_list) else None
        if u_fv_t is None or u_nn_t is None:
            continue
        t_val = times[i]
        time_fv = fv_time_list[i]
        time_nn = nn_time_list[nn_idx]
        speedup = time_fv / time_nn if time_nn > 0 else float("inf")

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xc, u_fv_t, label="FV (traditional)", linewidth=1.5)
        ax.plot(xc, u_nn_t, label="NN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"t = {t_val:.2f}")
        ax.legend()
        ax.set_ylim(y_lim)
        # 在图上标出运行时间
        time_text = f"FV runtime: {time_fv:.3f} s\nNN runtime: {time_nn:.4f} s\nNN speedup: {speedup:.1f}×"
        ax.text(0.02, 0.98, time_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        plt.tight_layout()
        fname = f"t{i}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    print(f"Saved {n_times} comparison plots to {output_dir}/ (t=0 to t=2, dt_nn={dt_nn:.4f}, n_nn_steps={n_nn_steps})")


if __name__ == "__main__":
    main()
