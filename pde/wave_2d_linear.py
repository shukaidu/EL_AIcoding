"""
2D linear wave equation on a periodic domain:
  u_tt = c^2 (u_xx + u_yy)

Space: Fourier pseudo-spectral (periodic)
Time:  Exact modal update in Fourier space (cos/sin per mode).

Python port of wave2d_main.m.
"""
import numpy as np


def wrap(z, L):
    """Wrap to [-L/2, L/2)."""
    return np.mod(z + L / 2, L) - L / 2


def build_ic(ic_type, xx, yy, Lx, Ly, Nx, Ny, K2, ic_params, c):
    """Build initial u0 and v0. ic_type: packet, collide, ring, flower, random_band, random_white, custom."""
    rng = np.random.default_rng(ic_params.get("rng_seed", 1))
    ic_type = ic_type.lower()

    if ic_type == "packet":
        theta = ic_params.get("theta_pkt", np.pi / 6)
        k0 = ic_params.get("k0_packet", 8)
        A = ic_params.get("amplitude", 1.0)
        sigma = ic_params.get("sigma_frac", 0.15) * min(Lx, Ly)
        x0, y0 = Lx / 3, Ly / 2
        kx, ky = k0 * np.cos(theta), k0 * np.sin(theta)
        phase = kx * (xx - x0) + ky * (yy - y0)
        dx = wrap(xx - x0, Lx)
        dy = wrap(yy - y0, Ly)
        env = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        u0 = A * env * np.cos(phase)
        v0 = A * c * k0 * env * np.sin(phase)

    elif ic_type == "collide":
        A = ic_params.get("amplitude", 1.0)
        k0 = ic_params.get("k0_packet", 8)
        sigma = ic_params.get("sigma_frac", 0.15) * min(Lx, Ly)
        x1, y1, th1 = Lx / 3, Ly / 2, 0.0
        x2, y2, th2 = 2 * Lx / 3, Ly / 2, np.pi
        ph1 = k0 * np.cos(th1) * (xx - x1) + k0 * np.sin(th1) * (yy - y1)
        ph2 = k0 * np.cos(th2) * (xx - x2) + k0 * np.sin(th2) * (yy - y2)
        dx1 = wrap(xx - x1, Lx)
        dy1 = wrap(yy - y1, Ly)
        dx2 = wrap(xx - x2, Lx)
        dy2 = wrap(yy - y2, Ly)
        env1 = np.exp(-(dx1**2 + dy1**2) / (2 * sigma**2))
        env2 = np.exp(-(dx2**2 + dy2**2) / (2 * sigma**2))
        u0 = A * (env1 * np.cos(ph1) + env2 * np.cos(ph2))
        v0 = A * c * k0 * (env1 * np.sin(ph1) + env2 * np.sin(ph2))

    elif ic_type == "ring":
        A = ic_params.get("amplitude", 1.0)
        x0, y0 = Lx / 2, Ly / 2
        R0 = 0.25 * min(Lx, Ly)
        w = 0.05 * min(Lx, Ly)
        dx = wrap(xx - x0, Lx)
        dy = wrap(yy - y0, Ly)
        r = np.sqrt(dx**2 + dy**2)
        g = np.exp(-((r - R0) ** 2) / (2 * w**2))
        u0 = A * g * np.cos(6 * (r - R0))
        v0 = A * c * ((r - R0) / (w**2 + 1e-12)) * g * np.cos(6 * (r - R0))

    elif ic_type == "flower":
        A = ic_params.get("amplitude", 1.0)
        m = ic_params.get("m_flower", 8)
        sigma = 2 * ic_params.get("sigma_frac", 0.15) * min(Lx, Ly)
        dx = wrap(xx - Lx / 2, Lx)
        dy = wrap(yy - Ly / 2, Ly)
        theta_grid = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)
        env = np.exp(-(r**2) / (2 * sigma**2))
        u0 = A * env * np.cos(m * theta_grid)
        v0 = np.zeros_like(u0)

    elif ic_type == "random_band":
        A = ic_params.get("amplitude", 1.0)
        kmin = ic_params.get("band_kmin", 10) * (2 * np.pi / min(Lx, Ly))
        kmax = ic_params.get("band_kmax", 14) * (2 * np.pi / min(Lx, Ly))
        K = np.sqrt(K2)
        mask = (K >= kmin) & (K <= kmax)
        Urand = (rng.standard_normal((Nx, Ny)) + 1j * rng.standard_normal((Nx, Ny))) * mask
        Urand[0, 0] = 0
        u0 = np.real(np.fft.ifft2(Urand))
        u0 = A * u0 / (np.max(np.abs(u0)) + 1e-12)
        v0 = np.zeros_like(u0)

    elif ic_type == "random_white":
        A = ic_params.get("amplitude", 1.0)
        smooth = ic_params.get("white_smooth", 0.1)
        u0 = rng.standard_normal((Nx, Ny))
        if smooth > 0:
            Ghat = np.exp(-(smooth**2) * K2)
            u0 = np.real(np.fft.ifft2(np.fft.fft2(u0) * Ghat))
        u0 = A * u0 / (np.max(np.abs(u0)) + 1e-12)
        v0 = np.zeros_like(u0)

    elif ic_type == "custom":
        dx = wrap(xx - Lx / 2, Lx)
        dy = wrap(yy - Ly / 2, Ly)
        u0 = np.exp(-(dx**2 + dy**2) / (2 * (0.2 * min(Lx, Ly)) ** 2))
        v0 = np.zeros_like(u0)

    else:
        raise ValueError(f"Unknown initial_condition: {ic_type}")

    return u0, v0


def wave2d_main(
    c=1.0,
    NX=512,
    NY=512,
    Lx=2 * np.pi,
    Ly=2 * np.pi,
    dt=2e-3,
    TF=4.0,
    TSCREEN=50,
    initial_condition="random_white",
    rng_seed=1,
    ic_params=None,
):
    """
    Run 2D linear wave equation (Fourier spectral, exact time step).
    Returns: t_history, u_history, v_history, xx, yy, initial_condition.
    u_history shape (NX, NY, n_frames), v_history same.
    """
    if ic_params is None:
        ic_params = {}
    ic_params.setdefault("m_flower", 8)
    ic_params.setdefault("k0_packet", 8)
    ic_params.setdefault("theta_pkt", np.pi / 6)
    ic_params.setdefault("sigma_frac", 0.15)
    ic_params.setdefault("band_kmin", 10)
    ic_params.setdefault("band_kmax", 14)
    ic_params.setdefault("white_smooth", 0.1)
    ic_params.setdefault("amplitude", 1.0)
    ic_params["rng_seed"] = rng_seed

    # Grid (periodic, exclude right endpoint)
    x = np.linspace(0, Lx, NX + 1)[:-1]
    y = np.linspace(0, Ly, NY + 1)[:-1]
    xx, yy = np.meshgrid(x, y, indexing="ij")  # (NX, NY)

    # Wavenumbers (MATLAB ordering: 0..NX/2, -NX/2+1..-1)
    kx = (2 * np.pi / Lx) * np.concatenate([np.arange(0, NX // 2 + 1), np.arange(-NX // 2 + 1, 0)])
    ky = (2 * np.pi / Ly) * np.concatenate([np.arange(0, NY // 2 + 1), np.arange(-NY // 2 + 1, 0)])
    KX, KY = np.meshgrid(kx, ky, indexing="ij")  # (NX, NY), KX[i,j]=kx[i], KY[i,j]=ky[j]
    K2 = KX**2 + KY**2
    omega = c * np.sqrt(K2)

    # Precompute trig factors for exact step (omega=0 -> sinw_over_w=dt, sinw=0)
    cosw = np.cos(omega * dt)
    sinw_over_w = np.full_like(omega, dt)
    sinw = np.zeros_like(omega)
    mask = omega > 1e-12
    sinw_over_w[mask] = np.sin(omega[mask] * dt) / omega[mask]
    sinw[mask] = np.sin(omega[mask] * dt)

    # Initial condition
    u0, v0 = build_ic(initial_condition, xx, yy, Lx, Ly, NX, NY, K2, ic_params, c)
    uhat = np.fft.fft2(u0)
    vhat = np.fft.fft2(v0)

    # Time loop
    n_steps = int(np.ceil(TF / dt))
    n_frames = n_steps // TSCREEN + 1
    u_history = np.zeros((NX, NY, n_frames), dtype=float)
    v_history = np.zeros((NX, NY, n_frames), dtype=float)
    t_history = np.zeros(n_frames)

    frame = 0
    u_history[:, :, frame] = u0
    v_history[:, :, frame] = v0
    t_history[frame] = 0.0

    t = 0.0
    for step in range(1, n_steps + 1):
        uhat_new = cosw * uhat + sinw_over_w * vhat
        vhat_new = -sinw * omega * uhat + cosw * vhat
        uhat = uhat_new
        vhat = vhat_new
        t += dt

        if step % TSCREEN == 0 or step == n_steps:
            frame += 1
            if frame < n_frames:
                u = np.real(np.fft.ifft2(uhat))
                v = np.real(np.fft.ifft2(vhat))
                u_history[:, :, frame] = u
                v_history[:, :, frame] = v
                t_history[frame] = t

    t_history = t_history[: frame + 1]
    u_history = u_history[:, :, : frame + 1]
    v_history = v_history[:, :, : frame + 1]
    return t_history, u_history, v_history, xx, yy, initial_condition
