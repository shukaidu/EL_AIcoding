"""Test wave2d_main (linear) and wave2d_spectral (nonlinear), save gifs to pde/test/."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

HERE = os.path.dirname(__file__)


def save_gif(u_history, t_history, Lx, Ly, name):
    fig, ax = plt.subplots(figsize=(5, 5))
    vmax0 = np.max(np.abs(u_history[:, :, 0])) or 1.0
    im = ax.imshow(u_history[:, :, 0].T, origin="lower", cmap="RdBu_r",
                   vmin=-vmax0, vmax=vmax0, extent=[0, Lx, 0, Ly])
    title = ax.set_title(f"t = {t_history[0]:.3f}")
    fig.colorbar(im, ax=ax)

    def update(frame):
        data = u_history[:, :, frame]
        vmax = np.max(np.abs(data)) or 1.0
        im.set_data(data.T)
        im.set_clim(-vmax, vmax)
        title.set_text(f"t = {t_history[frame]:.3f}")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=u_history.shape[2], interval=50, blit=True)
    out = os.path.join(HERE, name)
    ani.save(out, writer="pillow", fps=20)
    plt.close()
    print(f"Saved: {out}")


# --- Linear ---
from pde.wave_2d_linear import wave2d_main
import config.wave_2d_linear_config as lcfg

t_hist, u_hist, _, xx, yy, _ = wave2d_main(
    lcfg.NX, lcfg.NY, lcfg.Lx, lcfg.Ly, lcfg.dt, TF=10.0, TSCREEN=lcfg.TSCREEN,
    c=lcfg.c, initial_condition="ring", rng_seed=42, verbose=True,
)
print(f"[linear] n_frames={u_hist.shape[2]}, t_end={t_hist[-1]:.4f}")
save_gif(u_hist, t_hist, lcfg.Lx, lcfg.Ly, "test_wave2d_linear.gif")

# --- Nonlinear ---
from pde.wave_2d_nonlinear import wave2d_spectral
import config.wave_2d_nonlinear_config as ncfg

t_hist, U_hist, xx, yy, _, _, _, _ = wave2d_spectral(
    ncfg.Lx, ncfg.Ly, ncfg.nx, ncfg.ny, TF=10.0, TSCREEN=ncfg.TSCREEN,
    g=ncfg.g, h0=ncfg.h0, f_coriolis=ncfg.f_coriolis, nu_h=ncfg.nu_h, nu_q=ncfg.nu_q,
    initial_condition="random", rng_seed=42, verbose=True,
)
print(f"[nonlinear] n_frames={U_hist.shape[3]}, t_end={t_hist[-1]:.4f}")
save_gif(U_hist[:, :, 0, :], t_hist, ncfg.Lx, ncfg.Ly, "test_wave2d_nonlinear.gif")
