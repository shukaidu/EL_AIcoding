"""验证 wave_2d_linear 求解器：能量守恒、色散关系、收敛阶。"""
import numpy as np
from pde.wave_2d_linear import setup_wave2d, advance_tscreen

NX, NY = 128, 128
Lx, Ly = 2 * np.pi, 2 * np.pi
c  = 1.0
dt = 0.5 * (Lx / NX) / c   # CFL = 0.5

def make(ic="ring", seed=1):
    return setup_wave2d(NX, NY, Lx, Ly, c=c, initial_condition=ic, rng_seed=seed)

# ── 1. 能量守恒（无耗散，||uhat||² + ||vhat||²/c²K² 守恒）────────────────────
# 线性波能量 E = sum(|vhat|² + c²K²|uhat|²)
uhat, vhat, omega2, *_ = make()
K2 = -omega2 / c**2
E0 = np.sum(np.abs(vhat)**2 + c**2 * K2 * np.abs(uhat)**2)
uhat, vhat, _, _ = advance_tscreen(uhat, vhat, omega2, dt, 20)
E1 = np.sum(np.abs(vhat)**2 + c**2 * K2 * np.abs(uhat)**2)
rel_drift = abs(E1 - E0) / E0
print(f"[1] Energy conservation: rel drift = {rel_drift:.2e}  (expect < 0.1)")
assert rel_drift < 0.1, f"FAILED: {rel_drift}"
print("    PASSED")

# ── 2. 色散关系（单模振荡频率 = c|k|）────────────────────────────────────────
# 激发 (kx,ky)=(2,0) 单模，测量振荡周期
kx_mode, ky_mode = 2, 0
NX2, NY2 = 128, 128
x = np.linspace(0, Lx, NX2, endpoint=False)
y = np.linspace(0, Ly, NY2, endpoint=False)
xx, yy = np.meshgrid(x, y, indexing="ij")
k = 2 * np.pi / Lx * kx_mode
omega_exact = c * k
T_exact = 2 * np.pi / omega_exact

u_single = np.cos(kx_mode * 2 * np.pi / Lx * xx)
v_single = np.zeros_like(u_single)
uhat2 = np.fft.fft2(u_single)
vhat2 = np.fft.fft2(v_single)
kx2 = (2*np.pi/Lx) * np.concatenate([np.arange(0, NX2//2+1), np.arange(-NX2//2+1, 0)])
ky2 = (2*np.pi/Ly) * np.concatenate([np.arange(0, NY2//2+1), np.arange(-NY2//2+1, 0)])
KX2, KY2 = np.meshgrid(kx2, ky2, indexing="ij")
omega2_2 = -(KX2**2 + KY2**2) * c**2

# 积分 T_exact/4，u 应从 1 变为 0（余弦→零点）
n_quarter = max(1, round(T_exact / 4 / dt))
uhat2, vhat2, u2, _ = advance_tscreen(uhat2, vhat2, omega2_2, dt, n_quarter)
u_mean = np.mean(np.abs(u2))
print(f"[2] Dispersion relation: |u| at T/4 = {u_mean:.4f}  (expect ~ 0)")
assert u_mean < 0.2, f"FAILED: {u_mean}"
print("    PASSED")

# ── 3. 收敛阶（RK4，期望 4 阶）────────────────────────────────────────────────
# 用小网格避免大 dt 超出稳定域
NX_c, NY_c = 64, 64
T_conv = 2.0

def run_conv(n_steps):
    dt_r = T_conv / n_steps
    uh, vh, om2, *_ = setup_wave2d(NX_c, NY_c, Lx, Ly, c=c, initial_condition="ring", rng_seed=3)
    uh, vh, u, _ = advance_tscreen(uh, vh, om2, dt_r, n_steps)
    return u

u_ref = run_conv(2560)
ns = [40, 80, 160, 320, 640]
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
ratio_check = errs[1] / errs[2]   # n=80 -> n=160
assert ratio_check > 10.0, f"FAILED: ratio={ratio_check:.2f}"
print("    PASSED")

print("\nAll tests passed.")
