"""可视化所有 PDE 求解器，保存 GIF 到 pde/test/。

Usage:
  python pde/test/vis_pde.py                  # all
  python pde/test/vis_pde.py burgers          # burgers_1d only
  python pde/test/vis_pde.py linear           # wave_2d_linear only
  python pde/test/vis_pde.py nonlinear        # wave_2d_nonlinear only
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

HERE = os.path.dirname(__file__)
args = sys.argv[1:]
run_burgers   = not args or "burgers"   in args
run_linear    = not args or "linear"    in args
run_nonlinear = not args or "nonlinear" in args


def save_gif_1d(u_history, t_history, xc, name):
    """Save GIF for 1D field: u_history shape (nx, n_frames)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    line, = ax.plot(xc, u_history[:, 0])
    ax.set_xlim(xc[0], xc[-1])
    ymax = np.max(np.abs(u_history)) or 1.0
    ax.set_ylim(-ymax * 1.1, ymax * 1.1)
    title = ax.set_title(f"t = {t_history[0]:.3f}")

    def update(frame):
        line.set_ydata(u_history[:, frame])
        title.set_text(f"t = {t_history[frame]:.3f}")
        return line, title

    ani = animation.FuncAnimation(fig, update, frames=u_history.shape[1], interval=50, blit=True)
    out = os.path.join(HERE, name)
    ani.save(out, writer="pillow", fps=20)
    plt.close()
    print(f"Saved: {out}")


def save_gif_multi(U_history, t_history, Lx, Ly, var_names, name):
    """Save GIF with 1×n_vars subplots for a (nx, ny, n_vars, n_frames) array."""
    nx, ny, n_vars, n_frames = U_history.shape
    fig, axes = plt.subplots(1, n_vars, figsize=(13, 4))
    ims, ax_titles = [], []
    for i, ax in enumerate(axes):
        data0 = U_history[:, :, i, 0]
        vmax0 = np.max(np.abs(data0)) or 1.0
        im = ax.imshow(data0.T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax0, vmax=vmax0, extent=[0, Lx, 0, Ly])
        t = ax.set_title(f"{var_names[i]}  t={t_history[0]:.3f}")
        fig.colorbar(im, ax=ax)
        ims.append(im)
        ax_titles.append(t)

    def update(frame):
        artists = []
        for i, im in enumerate(ims):
            data = U_history[:, :, i, frame]
            vmax = np.max(np.abs(data)) or 1.0
            im.set_data(data.T)
            im.set_clim(-vmax, vmax)
            artists.append(im)
            ax_titles[i].set_text(f"{var_names[i]}  t={t_history[frame]:.3f}")
            artists.append(ax_titles[i])
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    out = os.path.join(HERE, name)
    ani.save(out, writer="pillow", fps=20)
    plt.close()
    print(f"Saved: {out}")


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


if run_burgers:
    from pde.burgers_1d import burgers_1d_main
    import config.burgers_1d_config as bcfg

    print(f"[burgers] nx={bcfg.nx}  dt={bcfg.dt:.5f}  TF={bcfg.compare_t_end}")
    t_history, u_history, xc = burgers_1d_main(
        bcfg.nx, bcfg.dx, bcfg.dt, bcfg.L, bcfg.nu, bcfg.alpha, bcfg.u_mean,
        TF=bcfg.compare_t_end, TSCREEN=bcfg.TSCREEN, rng_seed=bcfg.compare_seed, verbose=True,
    )
    print(f"[burgers] n_frames={u_history.shape[1]}  t_end={t_history[-1]:.4f}")
    save_gif_1d(u_history, t_history, xc, "vis_burgers_1d.gif")

if run_linear:
    from pde.wave_2d_linear import wave2d_main
    import config.wave_2d_linear_config as lcfg

    t_hist, u_hist, _, xx, yy, _ = wave2d_main(
        lcfg.NX, lcfg.NY, lcfg.Lx, lcfg.Ly, lcfg.dt, TF=lcfg.TF, TSCREEN=lcfg.TSCREEN,
        c=lcfg.c, initial_condition=lcfg.compare_ic, rng_seed=lcfg.compare_seed, verbose=True,
    )
    print(f"[linear] n_frames={u_hist.shape[2]}, t_end={t_hist[-1]:.4f}")
    save_gif(u_hist, t_hist, lcfg.Lx, lcfg.Ly, "test_wave2d_linear.gif")

if run_nonlinear:
    from pde.wave_2d_nonlinear import wave2d_spectral
    import config.wave_2d_nonlinear_config as ncfg

    print(f"[nonlinear] integrator={ncfg.integrator}  dt={ncfg.dt_internal:.5f}  TSCREEN={ncfg.TSCREEN}")
    t_hist, U_hist, xx, yy, _, _, _, _ = wave2d_spectral(
        ncfg.Lx, ncfg.Ly, ncfg.nx, ncfg.ny, TF=ncfg.TF, TSCREEN=ncfg.TSCREEN,
        g=ncfg.g, h0=ncfg.h0, f_coriolis=ncfg.f_coriolis, nu_h=ncfg.nu_h, nu_q=ncfg.nu_q,
        nudging_coeff=ncfg.nudging_coeff,
        initial_condition=ncfg.compare_ic, rng_seed=ncfg.compare_seed,
        integrator=ncfg.integrator, dt=ncfg.dt_internal, verbose=True,
    )
    print(f"[nonlinear] n_frames={U_hist.shape[3]}, t_end={t_hist[-1]:.4f}")
    save_gif_multi(U_hist, t_hist, ncfg.Lx, ncfg.Ly,
                   ["h-h0", "qx", "qy"], f"test_wave2d_nonlinear_{ncfg.integrator}.gif")
