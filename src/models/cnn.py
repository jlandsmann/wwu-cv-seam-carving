from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 5, kernel_size=3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            
            nn.Conv2d(5, 1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),
            
            nn.Linear(9409, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits