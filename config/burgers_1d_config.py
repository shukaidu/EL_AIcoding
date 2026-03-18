"""Config for 1D Burgers: data gen, train, compare."""
import numpy as np

# ---------------------------------------------------------------------------
# PDE & solver (used by gen_data and compare)
# ---------------------------------------------------------------------------
CFL = 0.5
umax = 1.0
L = 2.0
nx = 2000
dx = L / nx
t_end = 4.0
dt = CFL * dx / umax
nt = int(round(t_end / dt))
# Frames: save every TSCREEN solver steps
TSCREEN = 50
dt_samp = TSCREEN * dt
# NN prediction window: one NN step = njp saved frames
njp = 2
nwd = 100
nst = int(np.floor(njp * dt_samp / dx)) + 1
# Initial condition
alpha = 2.5
u_mean = 0.0
nu = 1e-3

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
nsamp = 5000
n_trajectories = 10
seed_base = 42

# ---------------------------------------------------------------------------
# NN prediction (model & training, for ml/train.py)
# ---------------------------------------------------------------------------
b_size = 128
num_epochs = 2500
hidden_size = 128
num_layers = 5
activation = "relu"  # relu|tanh|gelu|linear
lr_schedule = [(1000, 1e-3), (2000, 1e-4), (2500, 1e-5)]

# ---------------------------------------------------------------------------
# Compare (reference vs NN rollout)
# ---------------------------------------------------------------------------
compare_seed = 42
sample_seed = 123
compare_t_end = 2.0
compare_n_times = 6

# ---------------------------------------------------------------------------
# Data file names (under data/burgers_1d/)
# ---------------------------------------------------------------------------
data_mat = "data_res.mat"
model_pth = "data_res_model.pth"
error_mat = "data_res_error.mat"



