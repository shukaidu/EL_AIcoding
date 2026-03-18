"""验证 IMEX Strang splitting 正确性：一致性、质量守恒、收敛阶。"""
import numpy as np
from pde.wave_2d_nonlinear import setup_wave2d_nonlinear

COMMON = dict(Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
              g=9.8, h0=1.0, f_coriolis=0.0, nu_h=0.0, nu_q=0.0,
              nudging_coeff=0.0, initial_condition="random", rng_seed=42)

_dx = COMMON["Lx"] / COMMON["nx"]
_c  = np.sqrt(COMMON["g"] * COMMON["h0"])
DT_TEST = 0.5 * _dx / _c   # RK4 CFL dt for this test grid

def make(integrator, dt=DT_TEST):
    return setup_wave2d_nonlinear(**COMMON, integrator=integrator, dt=dt)

# ── 1. IMEX vs RK4 一致性（小 dt）────────────────────────────────────────────
h_i, qx_i, qy_i, _, adv_i, dt, _, _ = make("imex")
h_r, qx_r, qy_r, _, adv_r, _,  _, _ = make("rk4")
h_i, qx_i, qy_i = adv_i(h_i, qx_i, qy_i, dt, 20)
h_r, qx_r, qy_r = adv_r(h_r, qx_r, qy_r, dt, 20)
rel_err = np.linalg.norm(h_i - h_r) / np.linalg.norm(h_r)
print(f"[1] IMEX vs RK4 consistency:  rel L2 = {rel_err:.2e}  (expect < 1e-3)")
assert rel_err < 1e-3, f"FAILED: {rel_err}"
print("    PASSED")

# ── 2. 质量守恒 ──────────────────────────────────────────────────────────────
h, qx, qy, _, adv, dt, _, _ = make("imex")
mass0 = np.mean(h)
h, qx, qy = adv(h, qx, qy, dt, 100)
drift = abs(np.mean(h) - mass0)
print(f"[2] Mass conservation:        drift  = {drift:.2e}  (expect < 1e-10)")
assert drift < 1e-10, f"FAILED: {drift}"
print("    PASSED")

# ── 3. 收敛阶 ────────────────────────────────────────────────────────────────
# 用 ring IC（小振幅，线性区），参考解用 dt/16
CONV = dict(Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
            g=9.8, h0=1.0, f_coriolis=0.0, nu_h=0.0, nu_q=0.0,
            nudging_coeff=0.0, initial_condition="ring", rng_seed=1)

_dx_conv = CONV["Lx"] / CONV["nx"]
_c_conv  = np.sqrt(CONV["g"] * CONV["h0"])
dt_base  = 0.5 * _dx_conv / _c_conv
T = 5 * dt_base  # 短时间，保证在渐近区

def run_conv(n_steps):
    dt_run = T / n_steps
    h, qx, qy, _, adv, _, _, _ = setup_wave2d_nonlinear(**CONV, integrator="imex", dt=dt_run)
    return adv(h, qx, qy, dt_run, n_steps)[0]

h_ref = run_conv(640)   # 参考解
ns = [5, 10, 20, 40, 80, 160]
errs = [np.linalg.norm(run_conv(n) - h_ref) for n in ns]
print("[3] Convergence order:")
print(f"    {'n':>6}  {'err':>10}  {'ratio':>7}  {'order':>6}")
for i, (n, err) in enumerate(zip(ns, errs)):
    if i == 0:
        print(f"    {n:6d}  {err:10.3e}  {'---':>7}  {'---':>6}")
    else:
        ratio = errs[i-1] / err
        order = np.log(ratio) / np.log(2)
        print(f"    {n:6d}  {err:10.3e}  {ratio:7.2f}  {order:6.2f}")
ratio_check = errs[1] / errs[2]  # n=10 -> n=20
assert ratio_check > 3.0, f"FAILED: ratio={ratio_check:.2f}"
print("    PASSED")

print("\nAll tests passed.")
