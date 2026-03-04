import torch


def save_checkpoint(
    model, optimizer, epoch, train_loss_history, test_loss_history, filename,
    hidden_size=None, num_layers=None,
):
    d = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
    }
    if hidden_size is not None:
        d["hidden_size"] = hidden_size
    if num_layers is not None:
        d["num_layers"] = num_layers
    torch.save(d, filename)


def load_checkpoint(model, optimizer, filename):
    ckpt = torch.load(filename, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, ckpt.get("epoch", 0), ckpt.get("train_loss_history", []), ckpt.get("test_loss_history", [])
