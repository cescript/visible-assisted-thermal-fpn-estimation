import torch
import torch.nn as nn
from .OutputScalingModule import OutputScalingModule

class SIMPLENetwork(nn.Module):
    """Create SIMPLE Multi Layer Convolutional Network"""
    
    def __init__(self, ic, oc, alpha_scalar_offset, beta_scalar_offset):
        super(SIMPLENetwork, self).__init__()
        
        # single Conv2D + Normalization + ReLU block
        ld1 = [nn.Conv2d(ic, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld2 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld3 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld4 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld5 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld6 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld7 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld8 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        ld9 = [nn.Conv2d(32, oc, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(oc), nn.Tanh()]
        scm = OutputScalingModule(alpha_scalar_offset, beta_scalar_offset)
        
        # create sequential encoder model
        self.denoiser = nn.Sequential(*ld1, *ld2, *ld3, *ld4, *ld5, *ld6, *ld7, *ld8, *ld9, scm)
    
    def forward(self, inImage):
        return self.denoiser.forward(inImage)