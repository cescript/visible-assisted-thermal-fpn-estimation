import torch
import torch.nn as nn

class MeanIdentityMapping(nn.Module):
    def __init__(self):
        super(MeanIdentityMapping, self).__init__()

    def forward(self, x):
        # simply returns averaged input
        return torch.mean(x, dim=0)
