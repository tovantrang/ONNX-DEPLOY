import random

import numpy as np
import torch
import torch.nn as nn

from src.data.dataloader import generate_dataloaders
from src.export.onnx import export_onnx
from src.export.onnx import export_sanity_forward
from src.export.onnx import verify_onnxruntime
from src.models.tinycnn import TinyCNN
from src.training.training_loop import evaluate
from src.training.training_loop import train

num_epochs = 1
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, test_dl = generate_dataloaders(batch_size=64)

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dl, opt, criterion, device)
        evaluate(model, val_dl, criterion, device)

    evaluate(model, test_dl, criterion, device, name="Test")

    print("Training and evaluation complete.")

    export_sanity_forward(model, device)

    export_onnx(model, device)

    verify_onnxruntime(model, device)


if __name__ == "__main__":
    main()
