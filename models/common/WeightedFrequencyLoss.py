import torch
import torch.nn as nn
import torch.nn.functional as funs
import matplotlib.pyplot as plt

class WeightedFrequencyLoss(nn.Module):
    """
    Compute HF and LF loss separately and merge them using the weights
    Assumes that the inputs and the targets are [BATCH x CHANNELS x HEIGHT x WIDTH] and weights is 1xCHANNELS
    """
    def __init__(self, weights):
        super(WeightedFrequencyLoss, self).__init__()
        # make weights 1 x CHANNELS x 1 x 1 tensor
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.lowf_loss = torch.nn.L1Loss()
        self.higf_loss = torch.nn.L1Loss()
    
    # return low frequency signals
    def get_low_frequency_signals(self, clean, noisy):
        lp_clean = funs.interpolate(clean, size=(11, 11), mode='bicubic', align_corners=False)
        lp_noisy = funs.interpolate(noisy, size=(11, 11), mode='bicubic', align_corners=False)
        
        # return the difference between two low freq. signal
        return lp_clean, lp_noisy
    
    def get_high_frequency_signals(self, clean, noisy):
        # return the original signals
        return clean, noisy
    
    def visualize_differences(self, output, target, selected):
        # (BATCH x AGG) x HEIGHT x WIDTH
        diff = output[selected, :, :, :] - target[selected, :, :, :]

        # normalize the tensor to [0, 1] for displaying as an image
        diff = diff - diff.min()
        diff = diff / diff.max()

        # make HEIGHT x (BATCH x AGG x WIDTH) x 1
        image = diff.permute(1, 2, 0).detach().cpu().numpy()

        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.show()
    
    # output and target (BATCH x AGG) x 1 x HEIGHT x WIDTH
    def forward(self, output, target):
        # ensure input and target are of the same shape
        if output.shape != target.shape:
            raise ValueError("output and target must have the same shape")
    
        # get the low frequency signals and compute loss
        ol, tl = self.get_low_frequency_signals(output, target)
        lf_loss = self.lowf_loss(ol, tl)

        # get the high frequency signals and compute loss
        oh, th = self.get_high_frequency_signals(output, target)
        hf_loss = self.higf_loss(oh, th)

        # sum losses and return
        return self.weights[0] * lf_loss + self.weights[1] * hf_loss
