from torch import nn
from .block_type_a import BlockTypeA

class BlockTypeB(nn.Module):
    def __init__(self, dimensions: int):
        super().__init__()
        self.block_stack = nn.Sequential(
            BlockTypeA(dimensions, dimensions),

            nn.Conv2d(dimensions, dimensions, kernel_size=1),
            # nn.BatchNorm2d(dimensions),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block_stack(x) + x