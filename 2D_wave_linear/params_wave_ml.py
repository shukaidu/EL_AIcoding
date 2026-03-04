"""
Shared parameters for 2D wave ML: data generation and comparison.
Relation (from predWinParam_uv.m): the extended halo nst must cover the
domain of dependence. Over time jump njp*dt_samp, the wave travels at most
c * (njp*dt_samp) in space (c=1). So in grid points we need
  nst >= (njp * dt_samp) / dx  ->  nst = floor(njp*dt_samp/dx) + 1
so the input patch (2*nst+nwd)^2 contains all info needed to predict
the interior nwd^2 at the next time.
"""
import numpy as np

# Grid and time (must match between gen_data and compare)
NX = NY = 256
Lx = Ly = 2 * np.pi
dt = 2e-3
TF = 4.0
TSCREEN = 50

# Prediction window: nwd must divide NX and NY. njp=1 so one NN step = one frame (compare at all frames).
nwd = 32
njp = 1

dx = Lx / NX
dt_samp = TSCREEN * dt
# Halo size so patch covers domain of dependence for time jump njp*dt_samp
nst = int(np.floor(njp * dt_samp / dx)) + 1
patch_side = 2 * nst + nwd

# Sanity: patch must fit in grid
assert NX % nwd == 0 and NY % nwd == 0, "NX, NY must be divisible by nwd"
assert NX >= patch_side and NY >= patch_side, "Grid too small for patch_side"
