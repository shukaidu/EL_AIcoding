"""Config for 2D linear wave: data gen, train, compare."""
import numpy as np

# PDE
c = 1.0

# Grid and time (must match between gen_data and compare)
NX = NY = 256
Lx = Ly = 2 * np.pi
dt = 2e-3
TF = 4.0
TSCREEN = 50

# Prediction window: one NN step = njp frames
nwd = 32
njp = 2

dx = Lx / NX
dt_samp = TSCREEN * dt
nst = int(np.floor(njp * dt_samp / dx)) + 1
patch_side = 2 * nst + nwd

# Data generation
nsamp = 5000
ntest = 10
ic_list = ["random_white", "packet", "ring"]
rng_seeds = list(range(1, 21))

# Training
b_size = 128
num_epochs = 2500
hidden_size = 256
num_layers = 1
activation = "linear"  # relu|tanh|gelu|linear
lr_schedule = [(1000, 1e-3), (2000, 1e-4), (2500, 1e-5)]

# Compare (reference vs NN rollout)
compare_TF = 2.0
compare_ic = "ring"
compare_seed = 42
sample_seed = 123
compare_n_times = 6

# Data file names (under data/wave_2d_linear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"



