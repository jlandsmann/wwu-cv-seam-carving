from torch import nn

class BlockTypeA(nn.Module):
    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()
        self.block_stack = nn.Sequential(
            nn.Conv2d(in_dimensions, out_dimensions, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dimensions),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block_stack(x)