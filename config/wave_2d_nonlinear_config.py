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

# Time
TF = 10.0
TSCREEN = 20

# Prediction window: njp=1 so one NN step = one frame
nwd = 32
njp = 1


def nst_from_dt_samp(dt_samp):
    return int(4 * np.floor(c * njp * dt_samp / (4 * dx) + 1))


# Data generation
nsamp = 6000
ntest = 10
ic_list = ["random", "ring"]

# Training
b_size = 128
num_epochs = 50
base = 32  # ShrinkCNN base channels (smaller = faster)
lr_schedule = [(25, 1e-3), (40, 1e-4), (50, 1e-5)]

# Data file names (under data/wave_2d_nonlinear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
