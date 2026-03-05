"""Config for 2D nonlinear shallow-water: data gen, train, compare."""
import numpy as np

# Grid & PDE (match wave2d_spectral)
Lx = Ly = 2 * np.pi
nx = ny = 128
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
TSCREEN = 20
dt_internal = 0.5 * min(dx, dy) / c   # must match pde/wave_2d_nonlinear.py
dt_samp = TSCREEN * dt_internal        # time between saved frames

# Prediction window: one NN step = njp frames (1 = lighter/faster; 2 = heavier/slower)
nwd = 32
njp = 1

# Patch stencil from dt_samp (halo rounded to multiple of 4 for ShrinkCNN)
_nst_raw = int(4 * np.floor(c * njp * dt_samp / (4 * dx) + 1))
_halo = 2 * _nst_raw
_halo = ((_halo + 3) // 4) * 4
nst = _halo // 2
patch_side = nwd + 2 * nst


# Data generation
nsamp = 6000
ntest = 10
ic_list = ["random", "ring"]

# Training
b_size = 128
num_epochs = 80   # fewer epochs, better schedule → similar or better error, faster
base = 32        # ShrinkCNN base channels (keep moderate size)
lr_schedule = [(40, 3e-4), (70, 1e-4), (80, 1e-5)]  # slightly lower start, smooth decay

# Regularization: TV can over-smooth; gradient matching (Sobolev) often helps PDE surrogates more
lam_tv = [1e-3, 0.0, 0.0]   # TV only on h, reduced (was 1e-2); 0 = off
lam_sob = 0.05              # weight for gradient-matching loss (match d/dx, d/dy of output to target)

# Compare (reference vs NN rollout)
compare_TF = 1.0
compare_ic = "random"
compare_seed = 42

# Data file names (under data/wave_2d_nonlinear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
