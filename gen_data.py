"""
Single data generation entry. Run from repo root:
  python gen_data.py --problem burgers_1d
  python gen_data.py --problem wave_2d_linear
  python gen_data.py --problem wave_2d_nonlinear
"""
import os
import sys
import argparse
import numpy as np
from scipy.io import savemat

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)


def run_burgers_1d(data_dir):
    from pde.burgers_1d import set_param, gen_dist, build_diffusion_matrix, integrate_burger
    import config.burgers_1d_config as cfg

    prm = set_param()
    nx, nt, njp, nst, nwd = prm["nx"], prm["nt"], prm["njp"], prm["nst"], prm["nwd"]
    rng = np.random.RandomState(cfg.seed_base)
    seeds = [rng.randint(0, 2**31) for _ in range(cfg.n_trajectories)]
    all_inputs, all_outputs = [], []
    per_traj = (cfg.nsamp + cfg.n_trajectories - 1) // cfg.n_trajectories

    for seed in seeds:
        np.random.seed(seed)
        u0, _ = gen_dist(nx, prm["alpha"])
        u0 = u0 + prm["u_mean"]
        u = u0.copy()
        u_history = np.zeros((nx, nt + 1), dtype=float)
        u_history[:, 0] = u
        A = build_diffusion_matrix(nx, prm["dt"], prm["dx"], prm["nu"])
        for n in range(nt):
            u = integrate_burger(u, prm["dt"], prm["dx"], prm["nu"], A=A)
            u_history[:, n + 1] = u
        n_avail = (nx - 2 * nst - nwd + 1) * max(0, nt - njp + 1)
        n_take = min(per_traj, n_avail)
        if n_take <= 0:
            continue
        i0_pool = np.random.randint(0, nx - 2 * nst - nwd + 1, size=n_take)
        j0_pool = np.random.randint(0, nt - njp + 1, size=n_take)
        for k in range(n_take):
            i0, j0 = i0_pool[k], j0_pool[k]
            all_inputs.append(u_history[i0 : i0 + 2 * nst + nwd, j0])
            all_outputs.append(u_history[i0 + nst : i0 + nst + nwd, j0 + njp])

    input_arr = np.array(all_inputs, dtype=float).T
    output_arr = np.array(all_outputs, dtype=float).T
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, cfg.data_mat)
    savemat(out_path, {"input_tensor": input_arr, "output_tensor": output_arr})
    print(f"Saved {out_path}  input {input_arr.shape}  output {output_arr.shape}")


def run_wave_2d_linear(data_dir):
    from pde.wave_2d_linear import wave2d_main
    import config.wave_2d_linear_config as cfg

    input_arr_u = np.zeros((cfg.patch_side, cfg.patch_side, cfg.nsamp), dtype=np.float32)
    input_arr_v = np.zeros((cfg.patch_side, cfg.patch_side, cfg.nsamp), dtype=np.float32)
    output_arr_u = np.zeros((cfg.nwd, cfg.nwd, cfg.nsamp), dtype=np.float32)
    output_arr_v = np.zeros((cfg.nwd, cfg.nwd, cfg.nsamp), dtype=np.float32)
    cnt = 0
    n_per_run = cfg.nsamp // cfg.ntest
    for run in range(cfg.ntest):
        ic = cfg.ic_list[run % len(cfg.ic_list)]
        seed = cfg.rng_seeds[run]
        t_hist, u_hist, v_hist, xx, yy, _ = wave2d_main(
            NX=cfg.NX, NY=cfg.NY, Lx=cfg.Lx, Ly=cfg.Ly, dt=cfg.dt, TF=cfg.TF, TSCREEN=cfg.TSCREEN,
            initial_condition=ic, rng_seed=seed,
        )
        n_frames = u_hist.shape[2]
        for _ in range(n_per_run):
            i0 = np.random.randint(0, cfg.NX - cfg.patch_side + 1)
            j0 = np.random.randint(0, cfg.NY - cfg.patch_side + 1)
            t0 = np.random.randint(0, max(1, n_frames - cfg.njp))
            t1 = t0 + cfg.njp
            if t1 >= n_frames:
                continue
            input_arr_u[:, :, cnt] = u_hist[i0 : i0 + cfg.patch_side, j0 : j0 + cfg.patch_side, t0]
            input_arr_v[:, :, cnt] = v_hist[i0 : i0 + cfg.patch_side, j0 : j0 + cfg.patch_side, t0]
            output_arr_u[:, :, cnt] = u_hist[i0 + cfg.nst : i0 + cfg.nst + cfg.nwd, j0 + cfg.nst : j0 + cfg.nst + cfg.nwd, t1]
            output_arr_v[:, :, cnt] = v_hist[i0 + cfg.nst : i0 + cfg.nst + cfg.nwd, j0 + cfg.nst : j0 + cfg.nst + cfg.nwd, t1]
            cnt += 1
            if cnt >= cfg.nsamp:
                break
        if cnt >= cfg.nsamp:
            break
    input_arr_u = input_arr_u[:, :, :cnt].reshape((cfg.patch_side ** 2, cnt))
    input_arr_v = input_arr_v[:, :, :cnt].reshape((cfg.patch_side ** 2, cnt))
    output_arr_u = output_arr_u[:, :, :cnt].reshape((cfg.nwd ** 2, cnt))
    output_arr_v = output_arr_v[:, :, :cnt].reshape((cfg.nwd ** 2, cnt))
    input_tensor = np.vstack([input_arr_u, input_arr_v])
    output_tensor = np.vstack([output_arr_u, output_arr_v])
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, cfg.data_mat)
    savemat(out_path, {"input_tensor": input_tensor, "output_tensor": output_tensor})
    print(f"Saved {out_path}  input {input_tensor.shape}  output {output_tensor.shape}")


def _wave2d_nonlinear_single_run(args):
    """Top-level for multiprocessing pickle on Windows. args = (seed, ic, TF, TSCREEN, nx, ny)."""
    seed, ic, TF, TSCREEN, nx, ny = args
    from pde.wave_2d_nonlinear import wave2d_spectral
    t_hist, U_hist, _, _, _, _, _, _ = wave2d_spectral(
        initial_condition=ic, TF=TF, TSCREEN=TSCREEN, nx=nx, ny=ny, rng_seed=seed
    )
    return t_hist, U_hist


def run_wave_2d_nonlinear(data_dir):
    from multiprocessing import Pool, cpu_count
    import config.wave_2d_nonlinear_config as cfg

    run_configs = [
        (i + 1, cfg.ic_list[i % len(cfg.ic_list)], cfg.TF, cfg.TSCREEN, cfg.nx, cfg.ny)
        for i in range(cfg.ntest)
    ]
    n_workers = min(cfg.ntest, max(1, cpu_count() - 1))
    if n_workers <= 1:
        results = [_wave2d_nonlinear_single_run(c) for c in run_configs]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_wave2d_nonlinear_single_run, run_configs)

    t_hist, U_hist = results[0]
    nx_, ny_, _, nt = U_hist.shape
    dt_samp = float(t_hist[1] - t_hist[0]) if len(t_hist) > 1 else cfg.TSCREEN * (0.5 * (2 * np.pi / cfg.nx) / (9.8**0.5))
    nst = cfg.nst_from_dt_samp(dt_samp)
    halo = 2 * nst
    halo = ((halo + 3) // 4) * 4
    nst = halo // 2
    patch_side = cfg.nwd + 2 * nst
    print(f"nst={nst}, patch_side={patch_side}")

    U_historys = np.zeros((nx_, ny_, 3, nt, cfg.ntest), dtype=np.float64)
    for ii, (_, U_hist) in enumerate(results):
        U_historys[:, :, :, :, ii] = U_hist

    rng = np.random.default_rng(42)
    input_tensor = np.zeros((cfg.nsamp, 3, patch_side, patch_side), dtype=np.float32)
    output_tensor = np.zeros((cfg.nsamp, 3, cfg.nwd, cfg.nwd), dtype=np.float32)
    n_per_run = (cfg.nsamp + cfg.ntest - 1) // cfg.ntest
    cnt = 0
    for ii in range(cfg.ntest):
        U = U_historys[:, :, :, :, ii]
        for _ in range(n_per_run):
            if cnt >= cfg.nsamp:
                break
            t0 = rng.integers(0, nt - cfg.njp) if nt > cfg.njp else 0
            i0 = rng.integers(0, nx_ - 2 * nst - cfg.nwd + 1) if nx_ > 2 * nst + cfg.nwd else 0
            j0 = rng.integers(0, ny_ - 2 * nst - cfg.nwd + 1) if ny_ > 2 * nst + cfg.nwd else 0
            in_patch = U[i0 : i0 + 2 * nst + cfg.nwd, j0 : j0 + 2 * nst + cfg.nwd, :, t0]
            out_patch = U[i0 + nst : i0 + nst + cfg.nwd, j0 + nst : j0 + nst + cfg.nwd, :, t0 + cfg.njp]
            input_tensor[cnt] = np.transpose(in_patch, (2, 0, 1))
            output_tensor[cnt] = np.transpose(out_patch, (2, 0, 1))
            cnt += 1
        if cnt >= cfg.nsamp:
            break
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, cfg.data_mat)
    import scipy.io as sio
    sio.savemat(out_path, {"input_tensor": input_tensor, "output_tensor": output_tensor}, do_compression=True)
    print(f"Saved {out_path}  input {input_tensor.shape}  output {output_tensor.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    args = parser.parse_args()
    problem = args.problem
    data_dir = os.path.join(_repo_root, "data", problem)

    if problem == "burgers_1d":
        run_burgers_1d(data_dir)
    elif problem == "wave_2d_linear":
        run_wave_2d_linear(data_dir)
    else:
        run_wave_2d_nonlinear(data_dir)


if __name__ == "__main__":
    main()
