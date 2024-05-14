import torch
import torch.nn as nn

class ConvAggregator(nn.Module):
    def __init__(self, ic, oc):
        super(ConvAggregator, self).__init__()

        # convolutional layer to apply along along the AGGREGATED x CHANNELS
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # concatenate the tensors along the channel dimension (x: [AGGREGATED x BATCH x CHANNELS x HEIGHT x WIDTH])
        concatenated = torch.cat([xa for xa in x], dim=1)
        
        # return convolved input
        return self.conv(concatenated)
