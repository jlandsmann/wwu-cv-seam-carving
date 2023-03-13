from torch import nn

class BlockTypeE(nn.Module):
    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()
        self.block_stack = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_dimensions, out_dimensions),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.block_stack(x)