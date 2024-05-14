import torch
import torch.nn as nn
from ..common.OutputScalingModule import OutputScalingModule
from ..common.InitNetwork import InitNetwork

class SAFTAGenerator(nn.Module):
    """ u-NET like network for IR image generation from RGB+noisy input data """
    
    def __init__(self, ic, oc):
        super(SAFTAGenerator, self).__init__()

        self.pre_conv1 = nn.Sequential(nn.Conv2d(ic, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pre_conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.pre_conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())

        self.post_conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.post_conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.post_conv3 = nn.Sequential(nn.Conv2d(32, oc, kernel_size=3, stride=1, padding=1), nn.ReLU())
        
        # ENCODER BLOCK: enc_conv + downsample + enc_conv + downsample ...
        self.enc_conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.enc_conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU())

        # generic down/up sample layers (used many times)
        self.downsample = nn.MaxPool2d(2)
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # DECODER BLOCK: dec_conv + upsample + dec_conv + upsample ...
        self.dec_conv5 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.dec_conv4 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.dec_conv3 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1), nn.ReLU())
        self.dec_conv2 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, padding=1), nn.ReLU())
        self.dec_conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
    
    def forward(self, net_in):
        # ENCODER BLOCK: [4x256x256] -> [32x256x256] -> [32x128x128] -> [64x128x128] -> [64x64x64] -> [128x64x64] -> [128x32x32] -> [128x32x32] -> [128x16x16] -> [128x16x16]
        #                  net_in            e1             p1               e2             p2            e3             p3             e4              p4            e5
        a1 = self.pre_conv1(net_in)
        a2 = self.pre_conv2(a1)
        a3 = self.pre_conv3(a2)
        
        e1 = self.enc_conv1(a3)
        p1 = self.downsample(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.downsample(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.downsample(e3)
        e4 = self.enc_conv4(p3)
        p4 = self.downsample(e4)
        e5 = self.enc_conv5(p4)

        # DECODER BLOCK: [128x16x16] -> [128x32x32 | 128x32x32] x 2 -> [128x64x64 | 128x64x64] x 2 -> [128x128x128 | 64x128x128] x 2 -> [64x256x256 | 32x256x256] -> [32x256x256] -> [2x256x256]
        #                    e5              d5         e4                  d4         e3                   d3           e2                  d2           e1              d1             out
        d5 = self.upsample(e5)
        d4 = self.dec_conv5(torch.cat([d5, e4], dim=1))
        d4 = self.upsample(d4)
        d3 = self.dec_conv4(torch.cat([d4, e3], dim=1))
        d3 = self.upsample(d3)
        d2 = self.dec_conv3(torch.cat([d3, e2], dim=1))
        d2 = self.upsample(d2)
        d1 = self.dec_conv2(torch.cat([d2, e1], dim=1))
        d0 = self.dec_conv1(d1)

        z1 = self.post_conv1(d0)
        z2 = self.post_conv2(z1)
        ot = self.post_conv3(z2)
        
        # overall output of the network model
        return ot

class SAFTADenoiser(nn.Module):
    """ Create a FCN layers for IR NUC parameter estimation """
    
    def __init__(self, ic, oc, alpha_scalar_offset, beta_scalar_offset):
        super(SAFTADenoiser, self).__init__()
        
        # denoising model
        ld1 = [nn.Conv2d(ic, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)]
        ld2 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)]
        ld3 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)]
        ld4 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)]
        ld5 = [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.BatchNorm2d(32)]
        ld6 = [nn.Conv2d(32, oc, kernel_size=3, stride=1, padding=1), nn.Tanh()]
        scm = OutputScalingModule(alpha_scalar_offset, beta_scalar_offset)
        
        # create sequential denoiser model
        self.denoiser = nn.Sequential(*ld1, *ld2, *ld3, *ld4, *ld5, *ld6, scm)
    
    def forward(self, net_in):
        return self.denoiser.forward(net_in)
