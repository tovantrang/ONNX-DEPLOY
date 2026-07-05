import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_model_eager(model, path="./artifacts/tinycnn_cifar10.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved model state_dict to %s", path)


def load_model_eager(model, path="./artifacts/tinycnn_cifar10.pt", device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded model state_dict from %s", path)
