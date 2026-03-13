"""Config for 2D nonlinear shallow-water: data gen, train, compare."""
import numpy as np

# Grid & PDE (match wave2d_spectral)
Lx = Ly = 2 * np.pi
nx = ny = 256
g = 9.8
h0 = 1.0
c = np.sqrt(g * h0)
dx = Lx / nx
dy = Ly / ny
f_coriolis = 0.0
nu_h = 0.0

# Time (solver: dt = 0.5*min(dx,dy)/c, frames every TSCREEN steps)
TF = 10.0
TSCREEN = 10
dt_internal = 0.5 * min(dx, dy) / c   # must match pde/wave_2d_nonlinear.py
nu_q = 1e-3 * (min(dx, dy) ** 2) / dt_internal
dt_samp = TSCREEN * dt_internal        # time between saved frames

# Prediction window: one NN step = njp frames (1 = lighter/faster; 2 = heavier/slower)
nwd = 32
njp = 2

# Patch stencil: halo per side from CFL (wave travels ~c over njp*dt_samp); CNN needs (patch_side - nwd) divisible by 4 and >= 4
nst_min = c * njp * dt_samp / dx
nst = max(2, int(np.ceil(nst_min / 2)) * 2)   # smallest even integer >= nst_min, at least 2 for CNN
patch_side = nwd + 2 * nst


# Data generation
nsamp = 6000
ntest = 10
ic_list = ["random", "ring"]
#ic_list = ["random"]

# Training
b_size = 128
num_epochs = 150
base = 48        # CNN base channels (keep moderate size)
lr_schedule = [(60, 3e-4), (110, 1e-4), (140, 3e-5), (150, 1e-5)]
smooth_weight = 0.05   # TV regularization weight
smooth_mode = "absolute"   # "absolute" | "relative"

warmup_T = 4.0   # frames before this time are excluded from training

# Compare (reference vs NN rollout)
compare_TF = 3.0
compare_ic = "random"
compare_seed = 42
compare_n_times = 8

# Data file names (under data/wave_2d_nonlinear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
