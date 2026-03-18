"""验证 Burgers 1D 求解器：质量守恒、能量耗散、收敛阶。"""
import numpy as np
from pde.burgers_1d import setup_burger, integrate_burger, build_diffusion_matrix

nx = 128
L  = 2 * np.pi
dx = L / nx
dt = 0.5 * dx          # CFL ~ 0.5 (u_max ~ 1)
nu = 1e-3

def run(n_steps, rng_seed=7):
    u, _, A = setup_burger(nx, dx, dt, L, nu, alpha=2.5, u_mean=0.0, rng_seed=rng_seed)
    for _ in range(n_steps):
        u = integrate_burger(u, dt, dx, nu, A)
    return u

# ── 1. 质量守恒（周期边界，平均值守恒）────────────────────────────────────────
u0, _, A = setup_burger(nx, dx, dt, L, nu, alpha=2.5, u_mean=0.0, rng_seed=7)
mass0 = np.mean(u0)
u = u0.copy()
for _ in range(200):
    u = integrate_burger(u, dt, dx, nu, A)
drift = abs(np.mean(u) - mass0)
print(f"[1] Mass conservation:   drift = {drift:.2e}  (expect < 1e-12)")
assert drift < 1e-12, f"FAILED: {drift}"
print("    PASSED")

# ── 2. 能量耗散（nu > 0 时 ||u|| 应下降）──────────────────────────────────────
e0 = np.linalg.norm(u0)
e1 = np.linalg.norm(run(500))
print(f"[2] Energy dissipation:  ||u||: {e0:.4f} -> {e1:.4f}  (expect decrease)")
assert e1 < e0, f"FAILED: energy increased"
print("    PASSED")

# ── 3. 收敛阶（线性扩散主导，小振幅，1阶时间格式）──────────────────────────────
# 用纯扩散极限验证：小振幅 u，nu 大，对流可忽略
nx2 = 128; dx2 = L / nx2; nu2 = 0.1
T = 0.5

def run_conv(n_steps, seed=99):
    dt_r = T / n_steps
    A_r = build_diffusion_matrix(nx2, dt_r, dx2, nu2)
    u, _, _ = setup_burger(nx2, dx2, dt_r, L, nu2, alpha=3.0, u_mean=0.0, rng_seed=seed)
    u *= 0.01   # 小振幅压制对流
    for _ in range(n_steps):
        u = integrate_burger(u, dt_r, dx2, nu2, A_r)
    return u

u_ref = run_conv(2000)
ns = [25, 50, 100, 200]
errs = [np.linalg.norm(run_conv(n) - u_ref) for n in ns]
print("[3] Convergence order:")
print(f"    {'n':>6}  {'err':>10}  {'ratio':>7}  {'order':>6}")
for i, (n, err) in enumerate(zip(ns, errs)):
    if i == 0:
        print(f"    {n:6d}  {err:10.3e}  {'---':>7}  {'---':>6}")
    else:
        ratio = errs[i-1] / err
        order = np.log(ratio) / np.log(2)
        print(f"    {n:6d}  {err:10.3e}  {ratio:7.2f}  {order:6.2f}")
ratio_check = errs[0] / errs[1]
assert ratio_check > 1.5, f"FAILED: ratio={ratio_check:.2f}"
print("    PASSED")

print("\nAll tests passed.")
