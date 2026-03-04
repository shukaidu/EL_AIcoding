"""Config for 1D Burgers: data gen, train, compare."""
import numpy as np


def get_param():
    """Return param dict for solver and data generation (same as pde.burgers_1d.set_param)."""
    CFL = 0.5
    umax = 1.0
    L = 2.0
    nx = 2000
    dx = L / nx
    t_end = 4.0
    dt = CFL * dx / umax
    nt = int(round(t_end / dt))
    njp = 80
    nst = int(np.floor(njp * CFL)) + 1
    nwd = 100
    alpha = 3.0
    u_mean = 0.0
    nu = 0.0
    return {
        "CFL": CFL,
        "umax": umax,
        "L": L,
        "nx": nx,
        "dx": dx,
        "t_end": t_end,
        "dt": dt,
        "nt": nt,
        "njp": njp,
        "nst": nst,
        "nwd": nwd,
        "alpha": alpha,
        "u_mean": u_mean,
        "nu": nu,
    }


# Data generation
nsamp = 6000
n_trajectories = 5
seed_base = 42

# Training (for ml/train.py)
b_size = 128
num_epochs = 2500
hidden_size = 256
num_hidden_layers = 6
lr_schedule = [(1000, 1e-3), (2000, 1e-4), (2500, 1e-5)]

# Data file names (under data/burgers_1d/)
data_mat = "data_res.mat"
model_pth = "data_res_model.pth"
error_mat = "data_res_error.mat"
