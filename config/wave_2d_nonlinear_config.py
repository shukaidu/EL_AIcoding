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
frot0 = 0.0
nu_h = 0.0
nu_q = None  # set in solver when None

# Time (solver: dt = 0.5*min(dx,dy)/c, frames every TSCREEN steps)
TF = 10.0
TSCREEN = 10
dt_internal = 0.5 * min(dx, dy) / c   # must match pde/wave_2d_nonlinear.py
dt_samp = TSCREEN * dt_internal        # time between saved frames

# Prediction window: one NN step = njp frames (1 = lighter/faster; 2 = heavier/slower)
nwd = 32
njp = 2

# Patch stencil: halo per side from CFL (wave travels ~c over njp*dt_samp); ShrinkCNN needs (patch_side - nwd) divisible by 4 and >= 4
nst_min = c * njp * dt_samp / dx
nst = max(2, int(np.ceil(nst_min / 2)) * 2)   # smallest even integer >= nst_min, at least 2 for ShrinkCNN
patch_side = nwd + 2 * nst


# Data generation
nsamp = 6000
ntest = 10
#ic_list = ["random", "ring"]
ic_list = ["random"]

# Training
b_size = 128
num_epochs = 80   # fewer epochs, better schedule → similar or better error, faster
base = 32        # ShrinkCNN base channels (keep moderate size)
lr_schedule = [(40, 3e-4), (70, 1e-4), (80, 1e-5)]

# Compare (reference vs NN rollout)
compare_TF = 1.0
compare_ic = "random"
compare_seed = 42

# Data file names (under data/wave_2d_nonlinear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
