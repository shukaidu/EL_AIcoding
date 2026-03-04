import torch

def save_checkpoint(model, optimizer, epoch, train_loss_history, test_loss_history, filename, base=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history
    }
    if base is not None:
        checkpoint['base'] = base
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    train_loss_history = checkpoint.get('train_loss_history', [])
    test_loss_history = checkpoint.get('test_loss_history', [])
    base = checkpoint.get('base', 32)
    return model, optimizer, epoch, train_loss_history, test_loss_history, base
