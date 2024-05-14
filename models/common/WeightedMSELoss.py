import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    Compute channels weighted MSE loss. Assumes that the inputs and the targets are [BATCH x CHANNELS x HEIGHT x WIDTH] and weights is 1xCHANNELS
    """
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        # make weights 1 x CHANNELS x 1 x 1 tensor
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, output, target):
        # ensure input and target are of the same shape
        if output.shape != target.shape:
            raise ValueError("output and target must have the same shape")
        
        # compute the squared differences
        diff = output - target
        squared_diff = diff ** 2
        
        # apply weights
        weighted_squared_diff = squared_diff * self.weights
        
        # compute the mean over all dimensions except for the channel dimension
        loss = torch.sqrt(weighted_squared_diff.mean(dim=[0, 2, 3]))
        
        # sum over the channel dimensions to get the final loss value
        return loss.sum()
