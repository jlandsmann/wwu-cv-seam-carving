from torch import nn

from . import block_types as bt

class DCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            bt.BlockTypeA(1, 16),
            
            bt.BlockTypeB(16),
            
            bt.BlockTypeC(16),
            
            bt.BlockTypeD(16, 32),

            bt.BlockTypeE(32, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)