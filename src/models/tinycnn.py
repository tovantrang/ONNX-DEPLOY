from torch import nn


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
