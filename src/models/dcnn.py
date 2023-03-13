from torch import nn

from . import block_types as bt

class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            bt.BlockTypeA(1, 16),
            bt.BlockTypeA(16, 16),
            
            bt.BlockTypeB(16),
            bt.BlockTypeB(16),
            bt.BlockTypeB(16),
            
            bt.BlockTypeC(16),
            bt.BlockTypeC(16),
            
            bt.BlockTypeD(16, 32),
            bt.BlockTypeD(32, 64),
            bt.BlockTypeD(64, 128),

            bt.BlockTypeE(128, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)