"""Shared parameters for 2D nonlinear shallow-water ML (patch size, time step, grid)."""
import numpy as np

# Grid & PDE (match wave2d_spectral)
Lx = Ly = 2 * np.pi
nx = ny = 128
g = 9.8
h0 = 1.0
c = np.sqrt(g * h0)
dx = Lx / nx
dy = Ly / ny

# Time
TF = 10.0
TSCREEN = 20
# dt = 0.5 * min(dx, dy) / c  (computed in pde)

# Prediction window: njp=1 使「1 NN 步 = 1 帧」，对比时时间对齐（与 2D_wave_linear 一致）
nwd = 32
njp = 1


def nst_from_dt_samp(dt_samp):
    return int(4 * np.floor(c * njp * dt_samp / (4 * dx) + 1))


def patch_side(nst):
    return nwd + 2 * nst
