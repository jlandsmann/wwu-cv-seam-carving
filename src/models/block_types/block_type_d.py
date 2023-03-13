from torch import nn
from .block_type_a import BlockTypeA

class BlockTypeD(nn.Module):
    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()
        self.block_stack = nn.Sequential(
            BlockTypeA(in_dimensions, out_dimensions),
            BlockTypeA(out_dimensions, out_dimensions),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block_stack_second = nn.Sequential(
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=1, stride=2),
            # nn.BatchNorm2d(out_dimensions),
        )

    def forward(self, x):
        return self.block_stack(x) + self.block_stack_second(x)