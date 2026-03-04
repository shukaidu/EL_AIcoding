# 2D wave: NN vs spectral comparison

Compare the **spectral (traditional) solver** and the **trained NN** at each timestep. One figure per time: spectral u,v vs NN u,v (same color scale).

## Files (comparison pipeline only)

- **pde.py** – Spectral solver for the 2D wave equation.
- **params_wave_ml.py** – Shared ML params (nwd, nst, njp, grid); nst is set from the time jump so the prediction window covers the domain of dependence.
- **gen_data.py** – Builds **ML/data_wave.mat** (for training).
- **ML/train.py**, **ML/nn_model.py**, **ML/snapshot.py** – Train and load the NN.
- **compare_spectral_nn.py** – Runs one trajectory with spectral and with NN, saves **compare_spectral_nn/t0.png, t1.png, …** (spectral u,v vs NN u,v per time).

## Run the comparison

If the model is already trained (ML/data_wave_model.pth exists):

```bash
cd 2D_wave_linear
python compare_spectral_nn.py
```

To (re)generate data and (re)train first:

```bash
pip install torch scipy scikit-learn matplotlib numpy
python gen_data.py
python ML/train.py
python compare_spectral_nn.py
```

Use `OMP_NUM_THREADS=1 python ML/train.py` if you see an OpenMP error.

## Dependencies

NumPy, matplotlib, PyTorch, scipy, scikit-learn.
