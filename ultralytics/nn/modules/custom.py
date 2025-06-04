import torch
import torch.nn as nn

class AddBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sum(x)
