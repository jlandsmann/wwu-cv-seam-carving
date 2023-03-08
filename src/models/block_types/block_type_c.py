from torch import nn, concat
from .block_type_a import BlockTypeA
        
class BlockTypeC(nn.Module):
    dimensions: int

    def __init__(self, dimensions: int):
        super().__init__()
        self.a = BlockTypeA(dimensions, dimensions)
        self.b = BlockTypeA(dimensions, 2*dimensions)
        self.block_stack = nn.Sequential(
            nn.Conv2d(4*dimensions, dimensions, kernel_size=1),
            nn.BatchNorm2d(dimensions),
            nn.ReLU(),
        )

    def forward(self, x):
        tmpA = concat([x, self.a(x)], 1)
        tmpB = concat([tmpA, self.b(x)], 1)
        return self.block_stack(tmpB) + x