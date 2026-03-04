# Burgers: FV vs NN comparison

This project compares the **traditional finite-volume (FV) solver** and a **neural network (NN)** surrogate for the 1D Burgers equation. It generates data from the PDE, trains a patch-based NN, then plots both solutions side by side for t = 0, 0.1, …, 2.

## Files

- `pde.py` – core PDE solver and parameters (used by `gen_data.py` and `compare_fv_nn.py`)
- `gen_data.py` – generates training data and saves `data_res.mat`
- `ml/train.py` – trains the NN on `data_res.mat`, saves `ml/data_res_model.pth`
- `ml/nn_model.py` – NN architecture and data loader
- `ml/snapshot.py` – checkpoint save/load
- `compare_fv_nn.py` – runs FV and NN from t=0 to t=2, plots both on the same axes per time
- `compare_fv_nn/` – output folder: one PNG per time (FV vs NN, same y-scale)
- `requirements.txt` – Python dependencies

## Run the comparison pipeline

From the **1D_burgers** folder (this folder):

```bash
pip install -r requirements.txt
python gen_data.py
python ml/train.py
python compare_fv_nn.py
```

If you see an OpenMP error, run with:

```bash
OMP_NUM_THREADS=1 python ml/train.py
OMP_NUM_THREADS=1 python compare_fv_nn.py
```

Results are in `compare_fv_nn/`: each figure shows the FV solution and the NN solution at the same time (t = 0, 0.1, 0.2, …, 2).
