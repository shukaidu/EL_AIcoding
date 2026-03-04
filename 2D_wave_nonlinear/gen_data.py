"""Generate ML/data_wave.mat from Python spectral solver (multi-trajectory, multi-IC, patch sampling)."""
import os
import numpy as np
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from pde import wave2d_spectral
from params_wave_ml import TF, TSCREEN, nwd, njp, nx, ny, nst_from_dt_samp

# 多轨迹 + 多初值，提升多样性（对齐 2D_wave_linear 思路）
ntest = 10
nsamp = 6000
ic_list = ["random", "ring"]  # pde 支持的初值


def _run_one_trajectory(args):
    """Run one spectral trajectory; returns (t_hist, U_hist). seed, ic."""
    seed, ic = args
    t_hist, U_hist, _, _, _, _, _, _ = wave2d_spectral(
        initial_condition=ic,
        TF=TF, TSCREEN=TSCREEN, nx=nx, ny=ny, rng_seed=seed
    )
    return t_hist, U_hist


def main():
    n_workers = min(ntest, max(1, cpu_count() - 1))
    # 交替 random / ring，多种子
    run_configs = [(i + 1, ic_list[i % len(ic_list)]) for i in range(ntest)]
    if n_workers <= 1:
        results = [_run_one_trajectory(cfg) for cfg in run_configs]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_run_one_trajectory, run_configs)

    t_hist, U_hist = results[0]
    nx_, ny_, _, nt = U_hist.shape
    dt_samp = float(t_hist[1] - t_hist[0]) if len(t_hist) > 1 else TSCREEN * (0.5 * (2 * np.pi / nx) / np.sqrt(9.8))
    nst = nst_from_dt_samp(dt_samp)
    halo = 2 * nst
    halo = ((halo + 3) // 4) * 4
    nst = halo // 2
    patch_side = nwd + 2 * nst
    print(f"nst={nst}, patch_side={patch_side} (patch_side-nwd={patch_side - nwd})")

    U_historys = np.zeros((nx_, ny_, 3, nt, ntest), dtype=np.float64)
    for ii, (_, U_hist) in enumerate(results):
        U_historys[:, :, :, :, ii] = U_hist

    rng = np.random.default_rng(42)
    input_tensor = np.zeros((nsamp, 3, patch_side, patch_side), dtype=np.float32)
    output_tensor = np.zeros((nsamp, 3, nwd, nwd), dtype=np.float32)

    n_per_run = (nsamp + ntest - 1) // ntest
    cnt = 0
    for ii in range(ntest):
        U = U_historys[:, :, :, :, ii]
        for _ in range(n_per_run):
            if cnt >= nsamp:
                break
            t0 = rng.integers(0, nt - njp) if nt > njp else 0
            i0 = rng.integers(0, nx_ - 2 * nst - nwd + 1) if nx_ > 2 * nst + nwd else 0
            j0 = rng.integers(0, ny_ - 2 * nst - nwd + 1) if ny_ > 2 * nst + nwd else 0
            in_patch = U[i0 : i0 + 2 * nst + nwd, j0 : j0 + 2 * nst + nwd, :, t0]
            out_patch = U[i0 + nst : i0 + nst + nwd, j0 + nst : j0 + nst + nwd, :, t0 + njp]
            input_tensor[cnt] = np.transpose(in_patch, (2, 0, 1))
            output_tensor[cnt] = np.transpose(out_patch, (2, 0, 1))
            cnt += 1
        if cnt >= nsamp:
            break

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(_script_dir, "ML")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data_wave.mat")
    sio.savemat(out_path, {"input_tensor": input_tensor, "output_tensor": output_tensor}, do_compression=True)
    print("Saved", out_path, "input", input_tensor.shape, "output", output_tensor.shape)


if __name__ == "__main__":
    main()
