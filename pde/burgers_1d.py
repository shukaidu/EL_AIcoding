import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def set_param():
    """
    Python translation of the MATLAB setParam.m.
    Returns a dict with all parameters needed for the solver and data generation.
    """
    CFL = 0.5
    umax = 1.0

    L = 2.0
    nx = 2000
    dx = L / nx

    t_end = 4.0
    dt = CFL * dx / umax
    nt = int(round(t_end / dt))

    # 预测时间步：每步 njp*dt，减小 njp 使单步更容易学（短时预测更准）
    njp = 80
    nst = int(np.floor(njp * CFL)) + 1
    nwd = 100

    alpha = 3.0
    u_mean = 0.0

    # Viscosity; set to 0 for inviscid Burgers as requested
    nu = 0.0

    return {
        "CFL": CFL,
        "umax": umax,
        "L": L,
        "nx": nx,
        "dx": dx,
        "t_end": t_end,
        "dt": dt,
        "nt": nt,
        "njp": njp,
        "nst": nst,
        "nwd": nwd,
        "alpha": alpha,
        "u_mean": u_mean,
        "nu": nu,
    }


def gen_dist(N: int, alpha: float):
    """
    Python translation of genDist.m.
    Generates a random smooth initial condition using a Fourier series
    with coefficient decay ~ (1 + k^2)^(-alpha/2).
    """
    x = np.linspace(1.0 / (2 * N), 1.0 - 1.0 / (2 * N), N)
    f = np.zeros_like(x)

    for k in range(0, N + 1):
        add = np.random.randn() * np.cos(2 * np.pi * k * x) + np.random.randn() * np.sin(
            2 * np.pi * k * x
        )
        add /= (1.0 + k**2) ** (alpha / 2.0)
        f += add

    f -= f.mean()
    max_abs = np.max(np.abs(f))
    if max_abs > 0:
        f = 0.9 * f / max_abs

    return f, x


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


def run_reference_solver():
    """
    Convenience function: run the full FV solver with parameters
    translated from MATLAB's test_Burger.m using integrateBurger.m.
    Returns:
        xc  : cell centers in [0, L]
        u0  : initial condition
        u   : final solution after t_end
        prm : parameter dict from set_param()
    """
    prm = set_param()
    L = prm["L"]
    nx = prm["nx"]
    dx = prm["dx"]
    dt = prm["dt"]
    nt = prm["nt"]
    alpha = prm["alpha"]
    u_mean = prm["u_mean"]
    nu = prm["nu"]

    # Spatial grid (cell centers)
    xc = np.linspace(0.0, L, nx, endpoint=False) + dx / 2.0

    # Initial condition (match genData/test_Burger logic)
    u0, _ = gen_dist(nx, alpha)
    u0 = u0 + u_mean

    u = u0.copy()
    A = build_diffusion_matrix(nx, dt, dx, nu)

    for _ in range(nt):
        u = integrate_burger(u, dt, dx, nu, A=A)

    return xc, u0, u, prm


if __name__ == "__main__":
    xc, u0, u, prm = run_reference_solver()
    print(f"nx = {prm['nx']}, nt = {prm['nt']}")
    print(f"u0 mean = {u0.mean():.6f}, u mean = {u.mean():.6f}")
