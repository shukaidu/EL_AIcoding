"""Save/load checkpoint; extra kwargs stored for loading."""
import torch


def save_checkpoint(model, optimizer, epoch, hist_tr, hist_te, path, **extra):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss_history": hist_tr,
        "test_loss_history": hist_te,
    }
    ckpt.update({k: v for k, v in extra.items() if v is not None})
    torch.save(ckpt, path)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
