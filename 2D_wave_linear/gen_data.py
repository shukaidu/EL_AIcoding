"""
Generate training data for the 2D wave NN: run spectral solver, sample (u,v) patches
at t and t+njp*dt_samp, save ML/data_wave.mat (input_tensor, output_tensor).
nst is set from njp and dt_samp so the extended window covers the domain of dependence.
"""
import os
import numpy as np
from scipy.io import savemat
from pde import wave2d_main
from params_wave_ml import (
    NX, NY, Lx, Ly, dt, TF, TSCREEN,
    nwd, njp, nst, patch_side,
)

nsamp = 6000
# 多轨迹 + 多初值类型，提升多样性
ntest = 10
ic_list = ["random_white", "packet", "ring"]
rng_seeds = list(range(1, 21))


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "ML")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data_wave.mat")

    input_arr_u = np.zeros((patch_side, patch_side, nsamp), dtype=np.float32)
    input_arr_v = np.zeros((patch_side, patch_side, nsamp), dtype=np.float32)
    output_arr_u = np.zeros((nwd, nwd, nsamp), dtype=np.float32)
    output_arr_v = np.zeros((nwd, nwd, nsamp), dtype=np.float32)

    cnt = 0
    n_per_run = nsamp // ntest
    for run in range(ntest):
        ic = ic_list[run % len(ic_list)]
        seed = rng_seeds[run]
        t_hist, u_hist, v_hist, xx, yy, _ = wave2d_main(
            NX=NX, NY=NY, Lx=Lx, Ly=Ly, dt=dt, TF=TF, TSCREEN=TSCREEN,
            initial_condition=ic, rng_seed=seed,
        )
        n_frames = u_hist.shape[2]
        for _ in range(n_per_run):
            i0 = np.random.randint(0, NX - patch_side + 1)
            j0 = np.random.randint(0, NY - patch_side + 1)
            t0 = np.random.randint(0, max(1, n_frames - njp))
            t1 = t0 + njp
            if t1 >= n_frames:
                continue
            input_arr_u[:, :, cnt] = u_hist[i0 : i0 + patch_side, j0 : j0 + patch_side, t0]
            input_arr_v[:, :, cnt] = v_hist[i0 : i0 + patch_side, j0 : j0 + patch_side, t0]
            output_arr_u[:, :, cnt] = u_hist[i0 + nst : i0 + nst + nwd, j0 + nst : j0 + nst + nwd, t1]
            output_arr_v[:, :, cnt] = v_hist[i0 + nst : i0 + nst + nwd, j0 + nst : j0 + nst + nwd, t1]
            cnt += 1
            if cnt >= nsamp:
                break
        if cnt >= nsamp:
            break
        print(f"Run {run+1}/{ntest} (IC={ic}, seed={seed}) done, samples {cnt}")

    input_arr_u = input_arr_u[:, :, :cnt].reshape((patch_side ** 2, cnt))
    input_arr_v = input_arr_v[:, :, :cnt].reshape((patch_side ** 2, cnt))
    output_arr_u = output_arr_u[:, :, :cnt].reshape((nwd ** 2, cnt))
    output_arr_v = output_arr_v[:, :, :cnt].reshape((nwd ** 2, cnt))
    input_tensor = np.vstack([input_arr_u, input_arr_v])
    output_tensor = np.vstack([output_arr_u, output_arr_v])

    savemat(out_path, {"input_tensor": input_tensor, "output_tensor": output_tensor})
    print(f"Saved {out_path}  input {input_tensor.shape}  output {output_tensor.shape}  (nwd={nwd}, nst={nst}, njp={njp})")


if __name__ == "__main__":
    main()
