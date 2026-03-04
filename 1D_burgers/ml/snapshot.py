import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss_history,
    test_loss_history,
    filename,
    hidden_size=None,
    num_hidden_layers=None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss_history": train_loss_history,
        "test_loss_history": test_loss_history,
    }
    if hidden_size is not None:
        checkpoint["hidden_size"] = hidden_size
    if num_hidden_layers is not None:
        checkpoint["num_hidden_layers"] = num_hidden_layers
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    train_loss_history = checkpoint.get("train_loss_history", [])
    test_loss_history = checkpoint.get("test_loss_history", [])
    return model, optimizer, epoch, train_loss_history, test_loss_history
