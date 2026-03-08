import time as _time
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def gen_dist_1d(N: int, alpha: float, rng):
    """Random smooth 1D field with Fourier decay ~ 1/(1+|k|^alpha). Returns real 1D array (N,)."""
    k = np.arange(-N // 2, N // 2)
    decay = 1.0 + np.abs(k) ** alpha
    Y = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / decay
    Y = np.fft.ifftshift(Y)
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
    diagonals = [e, -2.0 * e, e]
    offsets = [-1, 0, 1]

    Lmat = diags(diagonals, offsets, shape=(nx, nx), format="csr")
    # periodic BCs
    Lmat = Lmat.tolil()
    Lmat[0, -1] = 1.0
    Lmat[-1, 0] = 1.0
    Lmat = Lmat.tocsr()

    A = eye(nx, format="csr") - (nu * dt / dx**2) * Lmat
    return A


def integrate_burger(u, dt: float, dx: float, nu: float, A=None):
    """
    One time step of the viscous Burgers' equation using:
    - Explicit Godunov flux for convection
    - Implicit diffusion via sparse linear solve
    This mirrors integrateBurger.m.
    """
    u = np.asarray(u, dtype=float)
    nx = u.size

    # Godunov flux (explicit convection)
    F = np.zeros(nx + 1)
    u_ext = np.concatenate([u[-1:], u, u[:1]])

    for i in range(nx + 1):
        u_L = u_ext[i]
        u_R = u_ext[i + 1]

        if u_L <= u_R:
            if u_L >= 0:
                F[i] = 0.5 * u_L**2
            elif u_R <= 0:
                F[i] = 0.5 * u_R**2
            else:
                F[i] = 0.0
        else:
            s = (0.5 * u_R**2 - 0.5 * u_L**2) / (u_R - u_L)
            if s >= 0:
                F[i] = 0.5 * u_L**2
            else:
                F[i] = 0.5 * u_R**2

    u_star = u - dt / dx * (F[1:] - F[:-1])

    # Implicit diffusion: (I - nu*dt/dx^2 * L) u^{n+1} = u_star
    if A is None:
        A = build_diffusion_matrix(nx, dt, dx, nu)

    u_next = spsolve(A, u_star)
    return u_next


def setup_burger(nx, dx, dt, L, nu, alpha, u_mean, rng_seed):
    """初始化 1D Burgers 状态。返回 (u0, xc, A)。"""
    rng = np.random.default_rng(rng_seed)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0
    u0 = gen_dist_1d(nx, alpha, rng) + u_mean
    A = build_diffusion_matrix(nx, dt, dx, nu)
    return u0, xc, A


def run_reference_solver():
    """
    Run the full FV solver with parameters from config.burgers_1d_config.
    Returns:
        xc : cell centers in [0, L]
        u0 : initial condition
        u  : final solution after t_end
    """
    from config import burgers_1d_config as cfg
    L = cfg.L
    nx, dx, dt, nt = cfg.nx, cfg.dx, cfg.dt, cfg.nt
    alpha, u_mean, nu = cfg.alpha, cfg.u_mean, cfg.nu

    rng = np.random.default_rng(42)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0
    u0 = gen_dist_1d(nx, alpha, rng) + u_mean

    u = u0.copy()
    A = build_diffusion_matrix(nx, dt, dx, nu)
    _t0 = _time.perf_counter()
    print_every = max(1, nt // 20)
    for n in range(nt):
        u = integrate_burger(u, dt, dx, nu, A=A)
        if (n + 1) % print_every == 0 or n + 1 == nt:
            elapsed = _time.perf_counter() - _t0
            t_phys = (n + 1) * dt
            eta = elapsed / (n + 1) * (nt - n - 1)
            print(f"  [burgers_1d] step {n+1:5d}/{nt}  t={t_phys:.4f}  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  |u|_max={np.max(np.abs(u)):.4f}", flush=True)

    return xc, u0, u


if __name__ == "__main__":
    import os
    import sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from config import burgers_1d_config as cfg
    xc, u0, u = run_reference_solver()
    print(f"nx = {cfg.nx}, nt = {cfg.nt}")
    print(f"u0 mean = {u0.mean():.6f}, u mean = {u.mean():.6f}")

