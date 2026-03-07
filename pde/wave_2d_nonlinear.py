"""
2D rotating shallow water on periodic domain (Fourier pseudo-spectral + RK4).
State: U = (h, qx=hu, qy=hv). Python port of wave2d_spectral.m.
"""
import numpy as np


def gen_dist_2d(xgrid, ygrid, alpha):
    """Random smooth 2D field with Fourier decay ~ 1/(1+|k|^alpha). Returns real 2D array (m,n)."""
    m, n = len(xgrid), len(ygrid)
    km = np.arange(-m // 2, m // 2)
    kn = np.arange(-n // 2, n // 2)
    decay = 1.0 + np.sqrt(km[:, None] ** 2 + kn[None, :] ** 2) ** alpha
    Y = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / decay
    Y = Y * m * n
    Y = np.fft.ifftshift(Y)
    X = np.real(np.fft.ifft2(Y))
    X -= np.mean(X)
    max_abs = np.max(np.abs(X))
    if max_abs > 0:
        X = 0.9 * X / max_abs
    return X


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
    frot0,
    nu_h,
    nu_q,
    initial_condition,
    rng_seed=None,
):
    """
    Run shallow-water solver. All parameters from config; no defaults here.
    Returns: t_history, U_history (nx, ny, 3, n_frames), xx, yy, initial_condition, g, h0, c
    U_history[:,:,0,:] = h - h0, [:,:,1,:] = qx, [:,:,2,:] = qy
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    c = np.sqrt(g * h0)
    dx = Lx / nx
    dy = Ly / ny
    dt = 0.5 * min(dx, dy) / c
    if nu_q is None or nu_q == 0:
        nu_q = 1e-3 * (min(dx, dy) ** 2) / dt

    # Cell-centered grid
    xgrid = np.linspace(dx / 2, Lx - dx / 2, nx)
    ygrid = np.linspace(dy / 2, Ly - dy / 2, ny)
    xx, yy = np.meshgrid(xgrid, ygrid, indexing="ij")

    # Wavenumbers (MATLAB ndgrid order: first dim kx, second dim ky)
    kx_vec = (2 * np.pi / Lx) * np.concatenate([np.arange(0, nx // 2), np.arange(-nx // 2, 0)])
    ky_vec = (2 * np.pi / Ly) * np.concatenate([np.arange(0, ny // 2), np.arange(-ny // 2, 0)])
    KX, KY = np.meshgrid(kx_vec, ky_vec, indexing="ij")
    ikx = 1j * KX
    iky = 1j * KY
    Lap = -(KX**2 + KY**2)
    kx_cut = (2 / 3) * np.max(np.abs(kx_vec))
    ky_cut = (2 / 3) * np.max(np.abs(ky_vec))
    dealias = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)

    def fft2r(f):
        return np.fft.fft2(f)

    def ifft2r(F):
        return np.real(np.fft.ifft2(F))

    def Dx_lin(f):
        return ifft2r(ikx * fft2r(f))

    def Dy_lin(f):
        return ifft2r(iky * fft2r(f))

    def LAP(f):
        return ifft2r(Lap * fft2r(f))

    def filter_nl(f):
        return ifft2r(fft2r(f) * dealias)

    # Initial condition
    if initial_condition == "ring":
        h = h0 + 0.1 * np.exp(-50 * ((xx - Lx / 3) ** 2 + (yy - Ly / 2) ** 2))
        psi = 0.5 * gen_dist_2d(xgrid, ygrid, 3.0)
    elif initial_condition == "random":
        psi = 0.5 * gen_dist_2d(xgrid, ygrid, 2.5)
        h = h0 + 0.0 * psi
    else:
        raise ValueError(f"Unknown initial_condition: {initial_condition}")

    u = Dy_lin(psi)
    v = -Dx_lin(psi)
    qx = h * u
    qy = h * v

    n_steps = int(np.ceil(TF / dt))
    n_frames = n_steps // TSCREEN
    t_history = np.arange(1, n_frames + 1, dtype=float) * TSCREEN * dt
    U_history = np.zeros((nx, ny, 3, n_frames))

    def rhs(h, qx, qy):
        h_safe = np.maximum(h, 1e-10)
        u = qx / h_safe
        v = qy / h_safe
        dhdt = -(Dx_lin(qx) + Dy_lin(qy))
        dhdt = dhdt - 1.0 * (h - h0)
        Fxx = qx * u + 0.5 * g * h**2
        Fxy = qx * v
        Gyx = qy * u
        Gyy = qy * v + 0.5 * g * h**2
        dqxdt = -Dx_lin(filter_nl(Fxx)) - Dy_lin(filter_nl(Fxy)) - frot0 * h * v
        dqydt = -Dx_lin(filter_nl(Gyx)) - Dy_lin(filter_nl(Gyy)) + frot0 * h * u
        if nu_h != 0:
            dhdt = dhdt + nu_h * LAP(h)
        if nu_q != 0:
            dqxdt = dqxdt + nu_q * LAP(qx)
            dqydt = dqydt + nu_q * LAP(qy)
        return dhdt, dqxdt, dqydt

    t = 0.0
    frame = 0
    for step in range(1, n_steps + 1):
        k1h, k1qx, k1qy = rhs(h, qx, qy)
        k2h, k2qx, k2qy = rhs(h + 0.5 * dt * k1h, qx + 0.5 * dt * k1qx, qy + 0.5 * dt * k1qy)
        k3h, k3qx, k3qy = rhs(h + 0.5 * dt * k2h, qx + 0.5 * dt * k2qx, qy + 0.5 * dt * k2qy)
        k4h, k4qx, k4qy = rhs(h + dt * k3h, qx + dt * k3qx, qy + dt * k3qy)
        h = h + (dt / 6) * (k1h + 2 * k2h + 2 * k3h + k4h)
        qx = qx + (dt / 6) * (k1qx + 2 * k2qx + 2 * k3qx + k4qx)
        qy = qy + (dt / 6) * (k1qy + 2 * k2qy + 2 * k3qy + k4qy)
        t += dt
        if step % TSCREEN == 0 and frame < n_frames:
            U_history[:, :, 0, frame] = h - h0
            U_history[:, :, 1, frame] = qx
            U_history[:, :, 2, frame] = qy
            t_history[frame] = t
            frame += 1

    t_history = t_history[:frame]
    U_history = U_history[:, :, :, :frame]
    return t_history, U_history, xx, yy, initial_condition, g, h0, c
