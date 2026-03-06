from pathlib import Path

import torch


def save_model_eager(model, path="./artifacts/tinycnn_cifar10.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model state_dict to: {path.resolve()}")


def load_model_eager(model, path="./artifacts/tinycnn_cifar10.pt", device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model state_dict from: {path.resolve()}")
