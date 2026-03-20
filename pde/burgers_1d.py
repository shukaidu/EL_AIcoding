import time as _time
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def gen_dist_1d(N: int, alpha: float, rng):
    """Random smooth 1D field with Fourier decay ~ 1/(1+|k|^alpha). Returns real 1D array (N,)."""
    k = np.arange(-N // 2, N // 2)
    decay = 1.0 + np.abs(k) ** alpha
    Y = np.fft.ifftshift((rng.standard_normal(N) + 1j * rng.standard_normal(N)) / decay)
    f = np.real(np.fft.ifft(Y)) * N
    f -= f.mean()
    max_abs = np.max(np.abs(f))
    if max_abs > 0:
        f = 0.9 * f / max_abs
    return f


def build_diffusion_matrix(nx: int, dt: float, dx: float, nu: float):
    """
    Construct the sparse matrix A = I - (nu * dt / dx^2) * L
    where L is the 1D Laplacian with periodic boundary conditions.
    This matches the MATLAB construction in integrateBurger.m.
    """
    e = np.ones(nx)
    Lmat = diags([e, -2.0 * e, e], [-1, 0, 1], shape=(nx, nx), format="lil")
    Lmat[0, -1] = 1.0
    Lmat[-1, 0] = 1.0
    return eye(nx, format="csr") - (nu * dt / dx**2) * Lmat.tocsr()


def _godunov_flux(u_ext):
    """Vectorised Godunov flux for Burgers' equation. u_ext has length nx+2."""
    u_L = u_ext[:-1]
    u_R = u_ext[1:]
    # Rarefaction: min of f at both sides; shock: upwind based on sign of s
    f_L = 0.5 * u_L ** 2
    f_R = 0.5 * u_R ** 2
    # Shock: s = (f_R - f_L)/(u_R - u_L) = (u_L + u_R)/2
    s = 0.5 * (u_L + u_R)
    shock = u_L > u_R          # entropy-satisfying shock
    raref = ~shock             # rarefaction fan
    F = np.where(shock, np.where(s >= 0, f_L, f_R),
                 np.where(u_L >= 0, f_L, np.where(u_R <= 0, f_R, 0.0)))
    return F


def integrate_burger(u, dt: float, dx: float, nu: float, A=None):
    """
    One time step of the viscous Burgers' equation using:
    - Explicit Godunov flux for convection
    - Implicit diffusion via sparse linear solve
    This mirrors integrateBurger.m.
    """
    u = np.asarray(u, dtype=float)
    nx = u.size
    u_ext = np.concatenate([u[-1:], u, u[:1]])
    F = _godunov_flux(u_ext)
    u_star = u - (dt / dx) * (F[1:] - F[:-1])
    if A is None:
        A = build_diffusion_matrix(nx, dt, dx, nu)
    return spsolve(A, u_star)


def setup_burger(nx, dx, dt, L, nu, alpha, u_mean, rng_seed):
    """初始化 1D Burgers 状态。返回 (u0, xc, A)。"""
    rng = np.random.default_rng(rng_seed)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0
    u0 = gen_dist_1d(nx, alpha, rng) + u_mean
    A = build_diffusion_matrix(nx, dt, dx, nu)
    return u0, xc, A


def burgers_1d_main(nx, dx, dt, L, nu, alpha, u_mean, TF, TSCREEN, rng_seed, verbose=False):
    """运行 Burgers 求解器，返回 (t_history, u_history, xc)。
    u_history shape: (nx, n_frames)，每 TSCREEN 步保存一帧。
    """
    u, xc, A = setup_burger(nx, dx, dt, L, nu, alpha, u_mean, rng_seed)
    n_steps = int(TF / dt)
    n_frames = n_steps // TSCREEN + 1
    t_history = np.zeros(n_frames)
    u_history = np.zeros((nx, n_frames))
    u_history[:, 0] = u
    t0 = _time.perf_counter()
    for frame in range(1, n_frames):
        for _ in range(TSCREEN):
            u = integrate_burger(u, dt, dx, nu, A)
        u_history[:, frame] = u
        t_history[frame] = frame * TSCREEN * dt
        if verbose:
            elapsed = _time.perf_counter() - t0
            print(f"  [burgers] frame {frame:4d}/{n_frames-1}  t={t_history[frame]:.4f}  "
                  f"elapsed={elapsed:.1f}s  |u|_max={np.max(np.abs(u)):.4f}", flush=True)
    return t_history, u_history, xc


if __name__ == "__main__":
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config import burgers_1d_config as cfg
    TF = cfg.nt * cfg.dt
    t_hist, u_hist, xc = burgers_1d_main(
        cfg.nx, cfg.dx, cfg.dt, cfg.L, cfg.nu, cfg.alpha, cfg.u_mean,
        TF=TF, TSCREEN=cfg.njp, rng_seed=42, verbose=True,
    )
    print(f"nx={cfg.nx}  n_frames={u_hist.shape[1]}  t_end={t_hist[-1]:.4f}")
    print(f"u0 mean={u_hist[:, 0].mean():.6f}  u_final mean={u_hist[:, -1].mean():.6f}")
