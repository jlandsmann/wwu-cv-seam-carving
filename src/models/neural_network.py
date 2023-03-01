from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
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
            
            nn.Linear(97*97, 512),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits