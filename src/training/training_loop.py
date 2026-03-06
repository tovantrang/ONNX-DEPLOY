import torch
from tqdm import tqdm


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
