"""
2D rotating shallow water on periodic domain (Fourier pseudo-spectral + RK4).
State: U = (h, qx=hu, qy=hv). Python port of wave2d_spectral.m.
"""
import time as _time
import numpy as np


def _gen_dist_2d(xgrid, ygrid, alpha, rng):
    """Random smooth 2D field with Fourier decay ~ 1/(1+|k|^alpha). Returns real 2D array (m,n)."""
    m, n = len(xgrid), len(ygrid)
    km = np.arange(-m // 2, m // 2)
    kn = np.arange(-n // 2, n // 2)
    decay = 1.0 + np.sqrt(km[:, None] ** 2 + kn[None, :] ** 2) ** alpha
    Y = (rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))) / decay
    Y = Y * m * n
    Y = np.fft.ifftshift(Y)
    X = np.real(np.fft.ifft2(Y))
    X -= np.mean(X)
    max_abs = np.max(np.abs(X))
    if max_abs > 0:
        X = 0.9 * X / max_abs
    return X


def _build_ic(initial_condition, xx, yy, xgrid, ygrid, Lx, Ly, h0, rng, Dx_lin, Dy_lin):
    """Build initial h, qx, qy. ic_type: ring, random."""
    if initial_condition == "ring":
        h = h0 + 0.1 * np.exp(-50 * ((xx - Lx / 3) ** 2 + (yy - Ly / 2) ** 2))
        psi = 0.5 * _gen_dist_2d(xgrid, ygrid, 3.0, rng)
    elif initial_condition == "random":
        psi = 0.5 * _gen_dist_2d(xgrid, ygrid, 2.5, rng)
        h = h0 + 0.0 * psi
    else:
        raise ValueError(f"Unknown initial_condition: {initial_condition}")
    u = Dy_lin(psi)
    v = -Dx_lin(psi)
    return h, h * u, h * v


def advance_tscreen(h, qx, qy, rhs, dt, TSCREEN):
    """Advance TSCREEN RK4 steps in physical space. Returns (h, qx, qy)."""
    for _ in range(TSCREEN):
        k1h, k1qx, k1qy = rhs(h, qx, qy)
        k2h, k2qx, k2qy = rhs(h + 0.5 * dt * k1h, qx + 0.5 * dt * k1qx, qy + 0.5 * dt * k1qy)
        k3h, k3qx, k3qy = rhs(h + 0.5 * dt * k2h, qx + 0.5 * dt * k2qx, qy + 0.5 * dt * k2qy)
        k4h, k4qx, k4qy = rhs(h + dt * k3h, qx + dt * k3qx, qy + dt * k3qy)
        h  = h  + (dt / 6) * (k1h  + 2 * k2h  + 2 * k3h  + k4h)
        qx = qx + (dt / 6) * (k1qx + 2 * k2qx + 2 * k3qx + k4qx)
        qy = qy + (dt / 6) * (k1qy + 2 * k2qy + 2 * k3qy + k4qy)
    return h, qx, qy


def setup_wave2d_nonlinear(Lx, Ly, nx, ny, *, g, h0, f_coriolis, nu_h, nu_q, initial_condition, rng_seed):
    """初始化网格、算子、初始条件。返回 (h, qx, qy, rhs, dt, xx, yy)。"""
    rng = np.random.default_rng(rng_seed)
    c = np.sqrt(g * h0)
    dx, dy = Lx / nx, Ly / ny
    dt = 0.5 * min(dx, dy) / c

    xgrid = np.linspace(dx / 2, Lx - dx / 2, nx)
    ygrid = np.linspace(dy / 2, Ly - dy / 2, ny)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing="ij")

    kx_vec = (2 * np.pi / Lx) * np.concatenate([np.arange(0, nx // 2), np.arange(-nx // 2, 0)])
    ky_vec = (2 * np.pi / Ly) * np.concatenate([np.arange(0, ny // 2), np.arange(-ny // 2, 0)])
    KX, KY = np.meshgrid(kx_vec, ky_vec, indexing="ij")
    ikx = 1j * KX
    iky = 1j * KY
    Lap = -(KX**2 + KY**2)
    kx_cut = (2 / 3) * np.max(np.abs(kx_vec))
    ky_cut = (2 / 3) * np.max(np.abs(ky_vec))
    dealias = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)

    def ifft2r(F):
        return np.real(np.fft.ifft2(F))

    def Dx_lin(f):
        return ifft2r(ikx * np.fft.fft2(f))

    def Dy_lin(f):
        return ifft2r(iky * np.fft.fft2(f))

    def LAP(f):
        return ifft2r(Lap * np.fft.fft2(f))

    def filter_nl(f):
        return ifft2r(np.fft.fft2(f) * dealias)

    h, qx, qy = _build_ic(initial_condition, xx, yy, xgrid, ygrid, Lx, Ly, h0, rng, Dx_lin, Dy_lin)

    def rhs(h, qx, qy):
        h_safe = np.maximum(h, 1e-10)
        u = qx / h_safe
        v = qy / h_safe
        dhdt = -(Dx_lin(qx) + Dy_lin(qy))
        dhdt = dhdt - 1.0 * (h - h0)  # Newtonian height relaxation for stability
        Fxx = qx * u + 0.5 * g * h**2
        Fxy = qx * v
        Gyx = qy * u
        Gyy = qy * v + 0.5 * g * h**2
        dqxdt = -Dx_lin(filter_nl(Fxx)) - Dy_lin(filter_nl(Fxy)) - f_coriolis * h * v
        dqydt = -Dx_lin(filter_nl(Gyx)) - Dy_lin(filter_nl(Gyy)) + f_coriolis * h * u
        if nu_h != 0:
            dhdt = dhdt + nu_h * LAP(h)
        if nu_q != 0:
            dqxdt = dqxdt + nu_q * LAP(qx)
            dqydt = dqydt + nu_q * LAP(qy)
        return dhdt, dqxdt, dqydt

    return h, qx, qy, rhs, dt, xx, yy


def wave2d_spectral(
    Lx,
    Ly,
    nx,
    ny,
    TF,
    TSCREEN,
    *,
    g,
    h0,
    f_coriolis,
    nu_h,
    nu_q,
    initial_condition,
    rng_seed,
    verbose=False,
):
    """
    Run shallow-water solver. All parameters from config; no defaults here.
    Returns: t_history, U_history (nx, ny, 3, n_frames), xx, yy, initial_condition, g, h0, c
    U_history[:,:,0,:] = h - h0, [:,:,1,:] = qx, [:,:,2,:] = qy
    """
    h, qx, qy, rhs, dt, xx, yy = setup_wave2d_nonlinear(
        Lx, Ly, nx, ny,
        g=g, h0=h0, f_coriolis=f_coriolis, nu_h=nu_h, nu_q=nu_q,
        initial_condition=initial_condition, rng_seed=rng_seed,
    )

    n_frames = (int(np.ceil(TF / dt)) // TSCREEN) + 1
    t_history = np.zeros(n_frames)
    U_history = np.zeros((nx, ny, 3, n_frames))

    U_history[:, :, 0, 0] = h - h0
    U_history[:, :, 1, 0] = qx
    U_history[:, :, 2, 0] = qy
    t_history[0] = 0.0

    t = 0.0
    _t0 = _time.perf_counter()
    for frame in range(1, n_frames):
        h, qx, qy = advance_tscreen(h, qx, qy, rhs, dt, TSCREEN)
        t += dt * TSCREEN
        U_history[:, :, 0, frame] = h - h0
        U_history[:, :, 1, frame] = qx
        U_history[:, :, 2, frame] = qy
        t_history[frame] = t
        if verbose:
            elapsed = _time.perf_counter() - _t0
            fps = frame / elapsed
            eta = (n_frames - 1 - frame) / fps if fps > 0 else float("inf")
            print(f"  [wave2d_nonlinear] frame {frame:4d}/{n_frames-1}  t={t:.4f}  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  |h-h0|_max={np.max(np.abs(h - h0)):.4f}", flush=True)

    c = np.sqrt(g * h0)
    return t_history, U_history, xx, yy, initial_condition, g, h0, c
