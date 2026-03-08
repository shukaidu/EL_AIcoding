"""
2D linear wave equation on a periodic domain:
  u_tt = c^2 (u_xx + u_yy)

Space: Fourier pseudo-spectral (periodic)
Time:  RK4 in Fourier space.

Python port of wave2d_main.m.
"""
import time as _time
import numpy as np

def _wrap(z, L):
    """Wrap to [-L/2, L/2)."""
    return np.mod(z + L / 2, L) - L / 2


def _build_ic(ic_type, xx, yy, Lx, Ly, Nx, Ny, K2, rng_seed, c):
    """Build initial u0 and v0. ic_type: packet, collide, ring, flower, random_band, random_white, custom."""
    rng = np.random.default_rng(rng_seed)
    ic_type = ic_type.lower()

    if ic_type == "packet":
        k0, theta, A = 8, np.pi / 6, 1.0
        sigma = 0.15 * min(Lx, Ly)
        x0, y0 = Lx / 3, Ly / 2
        kx, ky = k0 * np.cos(theta), k0 * np.sin(theta)
        phase = kx * (xx - x0) + ky * (yy - y0)
        dx = _wrap(xx - x0, Lx)
        dy = _wrap(yy - y0, Ly)
        env = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        u0 = A * env * np.cos(phase)
        v0 = A * c * k0 * env * np.sin(phase)

    elif ic_type == "collide":
        k0, A = 8, 1.0
        sigma = 0.15 * min(Lx, Ly)
        x1, y1, th1 = Lx / 3, Ly / 2, 0.0
        x2, y2, th2 = 2 * Lx / 3, Ly / 2, np.pi
        ph1 = k0 * np.cos(th1) * (xx - x1) + k0 * np.sin(th1) * (yy - y1)
        ph2 = k0 * np.cos(th2) * (xx - x2) + k0 * np.sin(th2) * (yy - y2)
        dx1 = _wrap(xx - x1, Lx)
        dy1 = _wrap(yy - y1, Ly)
        dx2 = _wrap(xx - x2, Lx)
        dy2 = _wrap(yy - y2, Ly)
        env1 = np.exp(-(dx1**2 + dy1**2) / (2 * sigma**2))
        env2 = np.exp(-(dx2**2 + dy2**2) / (2 * sigma**2))
        u0 = A * (env1 * np.cos(ph1) + env2 * np.cos(ph2))
        v0 = A * c * k0 * (env1 * np.sin(ph1) + env2 * np.sin(ph2))

    elif ic_type == "ring":
        A = 1.0
        x0, y0 = Lx / 2, Ly / 2
        R0 = 0.25 * min(Lx, Ly)
        w = 0.05 * min(Lx, Ly)
        dx = _wrap(xx - x0, Lx)
        dy = _wrap(yy - y0, Ly)
        r = np.sqrt(dx**2 + dy**2)
        g = np.exp(-((r - R0) ** 2) / (2 * w**2))
        u0 = A * g * np.cos(6 * (r - R0))
        v0 = A * c * ((r - R0) / (w**2 + 1e-12)) * g * np.cos(6 * (r - R0))

    elif ic_type == "flower":
        A, m = 1.0, 8
        sigma = 2 * 0.15 * min(Lx, Ly)
        dx = _wrap(xx - Lx / 2, Lx)
        dy = _wrap(yy - Ly / 2, Ly)
        theta_grid = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        env = np.exp(-(r**2) / (2 * sigma**2))
        u0 = A * env * np.cos(m * theta_grid)
        v0 = np.zeros_like(u0)

    elif ic_type == "random_band":
        A = 1.0
        kmin = 10 * (2 * np.pi / min(Lx, Ly))
        kmax = 14 * (2 * np.pi / min(Lx, Ly))
        K = np.sqrt(K2)
        mask = (K >= kmin) & (K <= kmax)
        Urand = (rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))) * mask
        Urand[0, 0] = 0
        u0 = np.real(np.fft.ifft2(Urand))
        u0 = A * u0 / (np.max(np.abs(u0)) + 1e-12)
        v0 = np.zeros_like(u0)

    elif ic_type == "random_white":
        smooth = 0.1
        u0 = rng.standard_normal((Nx, Ny))
        if smooth > 0:
            Ghat = np.exp(-(smooth**2) * K2)
            u0 = np.real(np.fft.ifft2(np.fft.fft2(u0) * Ghat))
        u0 = u0 / (np.max(np.abs(u0)) + 1e-12)
        v0 = np.zeros_like(u0)

    elif ic_type == "custom":
        dx = _wrap(xx - Lx / 2, Lx)
        dy = _wrap(yy - Ly / 2, Ly)
        u0 = np.exp(-(dx**2 + dy**2) / (2 * (0.2 * min(Lx, Ly)) ** 2))
        v0 = np.zeros_like(u0)

    else:
        raise ValueError(f"Unknown initial_condition: {ic_type}")

    return u0, v0


def advance_tscreen(uhat, vhat, omega2, dt, TSCREEN):
    """Advance TSCREEN RK4 steps in Fourier space. Returns (uhat, vhat, u, v)."""
    for _ in range(TSCREEN):
        k1u, k1v = vhat, omega2 * uhat
        k2u, k2v = vhat + 0.5 * dt * k1v, omega2 * (uhat + 0.5 * dt * k1u)
        k3u, k3v = vhat + 0.5 * dt * k2v, omega2 * (uhat + 0.5 * dt * k2u)
        k4u, k4v = vhat + dt * k3v, omega2 * (uhat + dt * k3u)
        uhat = uhat + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
        vhat = vhat + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    u = np.real(np.fft.ifft2(uhat))
    v = np.real(np.fft.ifft2(vhat))
    return uhat, vhat, u, v


def setup_wave2d(NX, NY, Lx, Ly, *, c, initial_condition, rng_seed):
    """Initialize grid, wavenumbers, and IC. Returns (uhat, vhat, omega2, xx, yy, u0, v0)."""
    x = np.linspace(0, Lx, NX + 1)[:-1]
    y = np.linspace(0, Ly, NY + 1)[:-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    kx = (2 * np.pi / Lx) * np.concatenate([np.arange(0, NX // 2 + 1), np.arange(-NX // 2 + 1, 0)])
    ky = (2 * np.pi / Ly) * np.concatenate([np.arange(0, NY // 2 + 1), np.arange(-NY // 2 + 1, 0)])
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    omega2 = -K2 * c**2

    u0, v0 = _build_ic(initial_condition, xx, yy, Lx, Ly, NX, NY, K2, rng_seed, c)
    uhat = np.fft.fft2(u0)
    vhat = np.fft.fft2(v0)
    return uhat, vhat, omega2, xx, yy, u0, v0


def wave2d_main(
    NX,
    NY,
    Lx,
    Ly,
    dt,
    TF,
    TSCREEN,
    *,
    c,
    initial_condition,
    rng_seed,
    verbose=False,
):
    """
    Run 2D linear wave equation (Fourier spectral, RK4 time stepping).
    All parameters (grid, time, c, initial_condition, rng_seed) from config; no defaults here.
    Returns: t_history, u_history, v_history, xx, yy, initial_condition.
    u_history shape (NX, NY, n_frames), v_history same.
    """
    uhat, vhat, omega2, xx, yy, u0, v0 = setup_wave2d(
        NX, NY, Lx, Ly, c=c, initial_condition=initial_condition, rng_seed=rng_seed
    )

    n_frames = (int(np.ceil(TF / dt)) // TSCREEN) + 1
    u_history = np.zeros((NX, NY, n_frames), dtype=float)
    v_history = np.zeros((NX, NY, n_frames), dtype=float)
    t_history = np.zeros(n_frames)

    u_history[:, :, 0] = u0
    v_history[:, :, 0] = v0
    t_history[0] = 0.0

    t = 0.0
    _t0 = _time.perf_counter()
    for frame in range(1, n_frames):
        uhat, vhat, u, v = advance_tscreen(uhat, vhat, omega2, dt, TSCREEN)
        t += dt * TSCREEN
        u_history[:, :, frame] = u
        v_history[:, :, frame] = v
        t_history[frame] = t
        if verbose:
            elapsed = _time.perf_counter() - _t0
            fps = frame / elapsed
            eta = (n_frames - 1 - frame) / fps if fps > 0 else float("inf")
            print(f"  [wave2d_linear] frame {frame:4d}/{n_frames-1}  t={t:.4f}  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  |u|_max={np.max(np.abs(u)):.4f}", flush=True)

    return t_history, u_history, v_history, xx, yy, initial_condition
