import sys
import torch
from NN_EL import ShrinkCNN, tv_isotropic_per_channel
from IO_EL import mat2pyt
from SLmodel_EL import save_checkpoint
from scipy.io import savemat


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on: {device}")
    return device


def setup_problem(dataname, b_size, device, base=64):
    dataloader_train, dataloader_test, N_samp, C_in, C_out, Nx, Ny, nx, ny = mat2pyt(
        dataname, device, b_size, test_split=0.2
    )
    model = ShrinkCNN(Cin=C_in, Cout=C_out, base=base, Nx=Nx, nx=nx).to(device)
    lam_tv = torch.tensor([1e-2, 0, 0], device=device)
    param_ratio = torch.tensor([10, 1, 1], device=device)
    criterion = torch.nn.L1Loss(reduction="none")
    return dataloader_train, dataloader_test, model, lam_tv, param_ratio, criterion


def train_and_eval(
    model, optimizer, dataloader_train, dataloader_test,
    lam_tv, param_ratio, criterion, num_epochs, start_epoch=0,
    train_loss_history=None, test_loss_history=None, modelname=None, training_error=None,
    component_weights=None,
):
    """component_weights: (C_out,) tensor, 1/std per channel to balance loss. If None, use param_ratio for data term."""
    if train_loss_history is None:
        train_loss_history = []
    if test_loss_history is None:
        test_loss_history = []
    total_epochs = start_epoch + num_epochs
    epoch_idx = start_epoch
    w = component_weights if component_weights is not None else param_ratio

    for _ in range(num_epochs):
        epoch_idx += 1
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        model.train()
        for inputs, targets in dataloader_train:
            outputs = model(inputs)
            diff = criterion(outputs, targets)
            data_loss_vec = diff.mean(dim=(0, 2, 3))
            tv_vec = tv_isotropic_per_channel(outputs)
            loss = (data_loss_vec * w + lam_tv * tv_vec).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(dataloader_train)
        train_loss_history.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            for inputs, targets in dataloader_test:
                outputs = model(inputs)
                diff = criterion(outputs, targets)
                data_loss_vec = diff.mean(dim=(0, 2, 3))
                tv_vec = tv_isotropic_per_channel(outputs)
                loss = (data_loss_vec * w + lam_tv * tv_vec).sum()
                epoch_test_loss += loss.item()
        avg_test_loss = epoch_test_loss / len(dataloader_test)
        test_loss_history.append(avg_test_loss)
        print(f"Epoch [{epoch_idx}/{total_epochs}], Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}")
        sys.stdout.flush()

    if modelname is not None:
        save_checkpoint(model, optimizer, epoch_idx, train_loss_history, test_loss_history, modelname)
    if training_error is not None:
        savemat(training_error, {"train_err": train_loss_history, "test_err": test_loss_history})
    return model, optimizer, epoch_idx, train_loss_history, test_loss_history
