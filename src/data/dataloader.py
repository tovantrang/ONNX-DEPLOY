import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def generate_dataloaders(batch_size=64):
    full_dataset_train = datasets.CIFAR10(root="./data", train=True, download=True)
    train_ds, val_ds = torch.utils.data.random_split(full_dataset_train, [0.8, 0.2])
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
        root="./data", train=False, download=True, transform=transform
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl
