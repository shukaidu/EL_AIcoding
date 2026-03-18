"""Gen data: python gen_data.py --problem burgers_1d|wave_2d_linear|wave_2d_nonlinear"""
import os
import sys
import argparse
import numpy as np
from scipy.io import savemat
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)


def _burgers_1d_single_trajectory(args):
    """Top-level for multiprocessing pickle. args = (seed, per_traj). Returns (inputs_list, outputs_list)."""
    seed, per_traj = args
    import config.burgers_1d_config as cfg
    from pde.burgers_1d import burgers_1d_main

    TF = cfg.nt * cfg.dt
    t_hist, u_history, xc = burgers_1d_main(
        cfg.nx, cfg.dx, cfg.dt, cfg.L, cfg.nu, cfg.alpha, cfg.u_mean,
        TF=TF, TSCREEN=cfg.TSCREEN, rng_seed=seed,
    )
    # u_history: (nx, n_frames)，相邻帧间距 = TSCREEN 步
    n_frames = u_history.shape[1]
    rng = np.random.default_rng(seed)
    inputs_list, outputs_list = [], []
    for _ in range(per_traj):
        i0 = rng.integers(0, cfg.nx - 2 * cfg.nst - cfg.nwd + 1)
        j0 = rng.integers(0, n_frames - cfg.njp)
        inputs_list.append(u_history[i0 : i0 + 2 * cfg.nst + cfg.nwd, j0])
        outputs_list.append(u_history[i0 + cfg.nst : i0 + cfg.nst + cfg.nwd, j0 + cfg.njp])
    return inputs_list, outputs_list


def run_burgers_1d(data_dir):
    import config.burgers_1d_config as cfg

    rng = np.random.RandomState(cfg.seed_base)
    seeds = [rng.randint(0, 2**31) for _ in range(cfg.n_trajectories)]
    run_configs = [(seed, cfg.nsamp // cfg.n_trajectories) for seed in seeds]
    n_workers = min(cfg.n_trajectories, max(1, cpu_count() - 1))
    if n_workers <= 1:
        results = [_burgers_1d_single_trajectory(c) for c in tqdm(run_configs, desc="burgers_1d trajectories")]
    else:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_burgers_1d_single_trajectory, run_configs),
                                total=cfg.n_trajectories, desc="burgers_1d trajectories"))

    all_inputs = []
    all_outputs = []
    for inputs_list, outputs_list in results:
        all_inputs.extend(inputs_list)
        all_outputs.extend(outputs_list)
    input_arr = np.array(all_inputs, dtype=float).T
    output_arr = np.array(all_outputs, dtype=float).T
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, cfg.data_mat)
    savemat(out_path, {"input_tensor": input_arr, "output_tensor": output_arr})
    print(f"Saved {out_path}  input {input_arr.shape}  output {output_arr.shape}")


def _wave2d_linear_single_run(args):
    """Top-level for multiprocessing pickle. args = (ic, seed, NX, NY, Lx, Ly, dt, TF, TSCREEN, c). Returns (u_hist, v_hist)."""
    ic, seed, NX, NY, Lx, Ly, dt, TF, TSCREEN, c = args
    from pde.wave_2d_linear import wave2d_main
    t_hist, u_hist, v_hist, xx, yy, _ = wave2d_main(
        NX, NY, Lx, Ly, dt, TF, TSCREEN,
        c=c, initial_condition=ic, rng_seed=seed,
    )
    return u_hist, v_hist


def run_wave_2d_linear(data_dir):
    import config.wave_2d_linear_config as cfg

    run_configs = [
        (cfg.ic_list[run % len(cfg.ic_list)], cfg.rng_seeds[run],
         cfg.NX, cfg.NY, cfg.Lx, cfg.Ly, cfg.dt, cfg.TF, cfg.TSCREEN, cfg.c)
        for run in range(cfg.ntest)
    ]
    n_workers = min(cfg.ntest, max(1, cpu_count() - 1))
    if n_workers <= 1:
        results = [_wave2d_linear_single_run(c) for c in tqdm(run_configs, desc="wave_2d_linear runs")]
    else:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_wave2d_linear_single_run, run_configs),
                                total=cfg.ntest, desc="wave_2d_linear runs"))

    input_arr_u = np.zeros((cfg.patch_side, cfg.patch_side, cfg.nsamp), dtype=np.float32)
    input_arr_v = np.zeros((cfg.patch_side, cfg.patch_side, cfg.nsamp), dtype=np.float32)
    output_arr_u = np.zeros((cfg.nwd, cfg.nwd, cfg.nsamp), dtype=np.float32)
    output_arr_v = np.zeros((cfg.nwd, cfg.nwd, cfg.nsamp), dtype=np.float32)
    cnt = 0
    n_per_run = cfg.nsamp // cfg.ntest
    rng = np.random.default_rng(cfg.sample_seed)
    for run_idx, (u_hist, v_hist) in enumerate(results):
        n_frames = u_hist.shape[2]
        for _ in range(n_per_run):
            if cnt >= cfg.nsamp:
                break
            i0 = rng.integers(0, cfg.NX - cfg.patch_side + 1)
            j0 = rng.integers(0, cfg.NY - cfg.patch_side + 1)
            t0 = rng.integers(0, max(1, n_frames - cfg.njp))
            t1 = t0 + cfg.njp

            input_arr_u[:, :, cnt] = u_hist[i0 : i0 + cfg.patch_side, j0 : j0 + cfg.patch_side, t0]
            input_arr_v[:, :, cnt] = v_hist[i0 : i0 + cfg.patch_side, j0 : j0 + cfg.patch_side, t0]
            output_arr_u[:, :, cnt] = u_hist[i0 + cfg.nst : i0 + cfg.nst + cfg.nwd, j0 + cfg.nst : j0 + cfg.nst + cfg.nwd, t1]
            output_arr_v[:, :, cnt] = v_hist[i0 + cfg.nst : i0 + cfg.nst + cfg.nwd, j0 + cfg.nst : j0 + cfg.nst + cfg.nwd, t1]
            cnt += 1
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
    """Top-level for multiprocessing pickle on Windows. args = (seed, ic, TF, TSCREEN, nx, ny, Lx, Ly, g, h0, f_coriolis, nu_h, nu_q, nudging_coeff, integrator). Returns (t_hist, U_hist)."""
    seed, ic, TF, TSCREEN, nx, ny, Lx, Ly, g, h0, f_coriolis, nu_h, nu_q, nudging_coeff, integrator = args
    from pde.wave_2d_nonlinear import wave2d_spectral
    t_hist, U_hist, _, _, _, _, _, _ = wave2d_spectral(
        Lx, Ly, nx, ny, TF, TSCREEN,
        g=g, h0=h0, f_coriolis=f_coriolis, nu_h=nu_h, nu_q=nu_q,
        nudging_coeff=nudging_coeff,
        initial_condition=ic, rng_seed=seed, integrator=integrator,
    )
    return t_hist, U_hist


def run_wave_2d_nonlinear(data_dir):
    import config.wave_2d_nonlinear_config as cfg

    run_configs = [
        (i + 1, cfg.ic_list[i % len(cfg.ic_list)], cfg.TF, cfg.TSCREEN, cfg.nx, cfg.ny, cfg.Lx, cfg.Ly,
         cfg.g, cfg.h0, cfg.f_coriolis, cfg.nu_h, cfg.nu_q, cfg.nudging_coeff, cfg.integrator)
        for i in range(cfg.ntest)
    ]
    n_workers = min(cfg.ntest, max(1, cpu_count() - 1))
    if n_workers <= 1:
        results = [_wave2d_nonlinear_single_run(c) for c in tqdm(run_configs, desc="wave_2d_nonlinear runs")]
    else:
        with Pool(n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_wave2d_nonlinear_single_run, run_configs),
                                total=cfg.ntest, desc="wave_2d_nonlinear runs"))

    t_hist, U_hist = results[0]
    nx_, ny_, _, nt = U_hist.shape
    nst = cfg.nst
    patch_side = cfg.patch_side
    print(f"nst={nst}, patch_side={patch_side}")

    U_historys = np.zeros((nx_, ny_, 3, nt, cfg.ntest), dtype=np.float64)
    for ii, (_, U_hist) in enumerate(results):
        U_historys[:, :, :, :, ii] = U_hist

    rng = np.random.default_rng(cfg.sample_seed)
    input_tensor = np.zeros((cfg.nsamp, 3, patch_side, patch_side), dtype=np.float32)
    output_tensor = np.zeros((cfg.nsamp, 3, cfg.nwd, cfg.nwd), dtype=np.float32)
    n_per_run = cfg.nsamp // cfg.ntest
    cnt = 0
    for ii in range(cfg.ntest):
        U = U_historys[:, :, :, :, ii]
        for _ in range(n_per_run):
            if cnt >= cfg.nsamp:
                break
            t_warmup = int(round(cfg.warmup_T / cfg.dt_samp))
            t0 = rng.integers(t_warmup, nt - cfg.njp)
            i0 = rng.integers(0, nx_ - 2 * nst - cfg.nwd + 1)
            j0 = rng.integers(0, ny_ - 2 * nst - cfg.nwd + 1)
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
    p = argparse.ArgumentParser()
    p.add_argument("--problem", required=True, choices=["burgers_1d", "wave_2d_linear", "wave_2d_nonlinear"])
    problem = p.parse_args().problem
    data_dir = os.path.join(_repo_root, "data", problem)
    {"burgers_1d": run_burgers_1d, "wave_2d_linear": run_wave_2d_linear, "wave_2d_nonlinear": run_wave_2d_nonlinear}[problem](data_dir)


if __name__ == "__main__":
    main()
