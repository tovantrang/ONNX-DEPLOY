from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

num_epochs = 1
seed = 42
g = torch.Generator().manual_seed(seed)


def generate_dataloaders(batch_size=64):
    full_dataset_train = datasets.CIFAR10(root="../data", train=True, download=True)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset_train, [0.8, 0.2], generator=g
    )
    imgs = train_ds.dataset.data[train_ds.indices]
    mean = imgs.mean(axis=(0, 1, 2)) / 255.0
    std = imgs.std(axis=(0, 1, 2)) / 255.0

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    train_ds.dataset.transform = transform
    val_ds.dataset.transform = transform

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_ds = datasets.CIFAR10(
        root="../data", train=False, download=True, transform=transform
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 10),
        )

    def forward(self, x):
        return self.net(x)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion, device, name="Validation"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            preds = torch.argmax(y_pred, dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    print(f"{name} Loss: {avg_loss}, Accuracy: {avg_accuracy}")


def export_sanity_forward(model, device):
    model.eval()
    x = torch.zeros(1, 3, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"SANITY forward: in {tuple(x.shape)}, out {tuple(y.shape)}")
    assert y.shape == (1, 10), f"Unexpected output shape: {tuple(y.shape)}"


def export_onnx(model, device, out_path="../artifacts/tinycnn_cifar10.onnx"):
    """
    Minimal ONNX export.
    TODO later: add dynamic axes + onnxruntime verification.
    """
    model.eval()
    dummy = torch.zeros(1, 3, 32, 32, device=device)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["logits"],
    )
    print(f"Exported ONNX to: {out_path.resolve()}")
    assert out_path.exists(), "ONNX export failed: file not found"


def verify_onnxruntime(model, device, onnx_path="../artifacts/tinycnn_cifar10.onnx"):
    """
    Minimal ONNXRuntime check: PyTorch logits vs ONNXRuntime logits on the same input.
    TODO later: test with a real CIFAR batch (not just zeros).
    """
    import onnxruntime as ort

    model.eval()
    x = torch.zeros(1, 3, 32, 32, device=device)

    with torch.no_grad():
        pt_logits = model(x).cpu().numpy()

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],  # keep it simple for now
    )
    ort_out = sess.run(["logits"], {"input": x.cpu().numpy()})[0]

    max_abs_diff = float(np.max(np.abs(pt_logits - ort_out)))
    print(
        "ONNXRuntime verify:",
        "pt",
        pt_logits.shape,
        "ort",
        ort_out.shape,
        "max_abs_diff",
        max_abs_diff,
    )

    assert (
        ort_out.shape == pt_logits.shape
    ), "Shape mismatch between PyTorch and ONNXRuntime"
    assert max_abs_diff < 1e-4, f"Outputs differ too much (max_abs_diff={max_abs_diff})"


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
