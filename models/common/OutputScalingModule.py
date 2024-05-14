import torch
import torch.nn as nn

class OutputScalingModule(nn.Module):
    def __init__(self, ch1_scalar_offset, ch2_scalar_offset):
        super(OutputScalingModule, self).__init__()
        
        self.ch1_scalar_offset = ch1_scalar_offset
        self.ch2_scalar_offset = ch2_scalar_offset
    
    def forward(self, x):
        # assuming x is a batch of 2-channel images of shape [N, 2, H, W]
        
        # scale alpha channel
        alpha = x[:, 0, :, :]
        alpha_s = self.ch1_scalar_offset[0] * alpha + self.ch1_scalar_offset[1]
        
        # scale second channel
        beta = x[:, 1, :, :]
        beta_s = self.ch2_scalar_offset[0] * beta + self.ch2_scalar_offset[1]
        
        # Stack the channels back together
        scaled_x = torch.stack((alpha_s, beta_s), dim=1)
        
        return scaled_x
