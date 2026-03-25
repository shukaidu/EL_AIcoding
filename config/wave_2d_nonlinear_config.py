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
nudging_coeff = 1  # h nudging toward h0 (0 = disabled)

# Time integrator: "imex" (Strang splitting, implicit gravity waves) or "rk4" (explicit)
integrator = "imex"

# Time (solver: dt = 0.5*min(dx,dy)/c, frames every TSCREEN steps)
TF = 12.0
dt_internal = 0.5 * min(dx, dy) if integrator == "imex" else 0.5 * min(dx, dy) / c
TSCREEN = 10
TSCREEN = TSCREEN if integrator == "imex" else round(TSCREEN * c)
nu_h = 0.0
nu_q = 1e-3 * (min(dx, dy) ** 2) / dt_internal
dt_samp = TSCREEN * dt_internal        # time between saved frames

# Prediction window: one NN step = njp frames (1 = lighter/faster; 2 = heavier/slower)
full_domain = False   # True: input/output = whole domain; False: patch-based element learning
nwd = 32
njp = 2

# Patch stencil: halo per side from CFL (wave travels ~c over njp*dt_samp); 2*nst must be divisible by 4 => nst divisible by 2
if full_domain:
    nwd = nx
    nst = 0
    patch_side = nx
else:
    _nst_min = round(c * njp * dt_samp / dx)   # ≈ njp*TSCREEN/2 (exact int, round guards float drift)
    nst = 2 * (_nst_min // 2 + 1)
    patch_side = nwd + 2 * nst


# Data generation
nsamp = 2500
ntest = 5
#ic_list = ["random", "ring"]
ic_list = ["random"]
ic_alpha_ring   = 3.0   # spectral decay exponent for ring stream function (larger = smoother)
ic_alpha_random = 2.5   # spectral decay exponent for random stream function (larger = smoother)

# Training
b_size = 100   # reduce for full_domain mode (256x256 samples are 64x larger than 32x32 patches)
test_split = 0.2
num_epochs = 150
base = 32        # model base channels
model_type = "unet"   # "cnn" | "unet"
pooling = "avg"       # "max" | "avg" | "stride"
residual = False       # predict delta (next - current) instead of next state
lr_schedule = [(80, 1e-3), (100, 5e-4), (120, 1e-4), (150, 5e-5)]
smooth_weight = [0, 1e-1, 1e-1]  # per-channel [h-h0, qx, qy]
smooth_mode = "absolute"   # "absolute" | "relative"
param_ratio   = [1.0, 1.0, 1.0]     # h 通道数据损失×10

warmup_T = 4.0   # frames before this time are excluded from training

# Compare (reference vs NN rollout)
compare_TF = 8.0
compare_ic = "random"
compare_seed = 17
sample_seed = 123
compare_n_times = 8

# Data file names (under data/wave_2d_nonlinear/)
data_mat = "data_wave.mat"
model_pth = "data_wave_model.pth"
error_mat = "data_wave_error.mat"
