"""Config for 2D linear wave: data gen, train, compare."""
import numpy as np

# PDE
c = 1.0
ic_param_defaults = {
    "m_flower": 8,
    "k0_packet": 8,
    "theta_pkt": np.pi / 6,
    "sigma_frac": 0.15,
    "band_kmin": 10,
    "band_kmax": 14,
    "white_smooth": 0.1,
    "amplitude": 1.0,
}

# Grid and time (must match between gen_data and compare)
NX = NY = 256
Lx = Ly = 2 * np.pi
dt = 2e-3
TF = 4.0
TSCREEN = 50

# Prediction window: njp=1 so one NN step = one frame
nwd = 32
njp = 1

dx = Lx / NX
dt_samp = TSCREEN * dt
nst = int(np.floor(njp * dt_samp / dx)) + 1
patch_side = 2 * nst + nwd

# Data generation
nsamp = 6000
ntest = 10
ic_list = ["random_white", "packet", "ring"]
rng_seeds = list(range(1, 21))

# Training
b_size = 128
num_epochs = 100  # quick test; use 2500 for full run
hidden_size = 256
num_layers = 5
lr_schedule = [(50, 1e-3), (80, 1e-4), (100, 1e-5)]

# Data file names (under data/wave_2d_linear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
