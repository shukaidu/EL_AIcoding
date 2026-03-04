import os
import numpy as np
from scipy.io import savemat

from pde import set_param, gen_dist, build_diffusion_matrix, integrate_burger


def _run_one_trajectory(seed: int, prm: dict):
    """单条轨迹：初值由 seed 决定，返回 u_history (nx, nt+1)."""
    nx = prm["nx"]
    dt = prm["dt"]
    nt = prm["nt"]
    alpha = prm["alpha"]
    u_mean = prm["u_mean"]
    nu = prm["nu"]
    dx = prm["dx"]
    np.random.seed(seed)
    u0, _ = gen_dist(nx, alpha)
    u0 = u0 + u_mean
    u = u0.copy()
    u_history = np.zeros((nx, nt + 1), dtype=float)
    u_history[:, 0] = u
    A = build_diffusion_matrix(nx, dt, dx, nu)
    for n in range(nt):
        u = integrate_burger(u, dt, dx, nu, A=A)
        u_history[:, n + 1] = u
    return u_history


def generate_dataset(
    filename: str = "data_res.mat",
    nsamp: int = 6000,
    n_trajectories: int = 5,
    seed_base: int = 42,
):
    """
    多轨迹生成数据，提高多样性；写入 'input_tensor' 与 'output_tensor'。
    """
    prm = set_param()
    nx = prm["nx"]
    nt = prm["nt"]
    njp = prm["njp"]
    nst = prm["nst"]
    nwd = prm["nwd"]

    rng = np.random.RandomState(seed_base)
    seeds = [rng.randint(0, 2**31) for _ in range(n_trajectories)]
    all_inputs = []
    all_outputs = []
    per_traj = (nsamp + n_trajectories - 1) // n_trajectories

    for seed in seeds:
        u_history = _run_one_trajectory(seed, prm)
        n_avail = (nx - 2 * nst - nwd + 1) * max(0, nt - njp + 1)
        n_take = min(per_traj, n_avail)
        if n_take <= 0:
            continue
        i0_pool = np.random.randint(0, nx - 2 * nst - nwd + 1, size=n_take)
        j0_pool = np.random.randint(0, nt - njp + 1, size=n_take)
        for k in range(n_take):
            i0, j0 = i0_pool[k], j0_pool[k]
            inp = u_history[i0 : i0 + 2 * nst + nwd, j0]
            out = u_history[i0 + nst : i0 + nst + nwd, j0 + njp]
            all_inputs.append(inp)
            all_outputs.append(out)

    input_arr = np.array(all_inputs, dtype=float).T  # (2*nst+nwd, N)
    output_arr = np.array(all_outputs, dtype=float).T  # (nwd, N)
    savemat(filename, {"input_tensor": input_arr, "output_tensor": output_arr})


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(_script_dir, "ml", "data_res.mat")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("Generating data_res.mat (multi-trajectory, 6000 samples)...")
    generate_dataset(out_path, nsamp=6000, n_trajectories=5, seed_base=42)
    print("Saved", out_path)

