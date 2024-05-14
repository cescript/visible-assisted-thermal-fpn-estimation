import torch
import torch.nn as nn

class OLSFPNEstimator(nn.Module):
    """
    Estimate FPN using OLS algorithm using the input as formed [BATCH x [AGG_CIR | AGG_NIR] x HEIGHT x WIDTH]
    """
    def __init__(self, agg_size, alpha_so, beta_so):
        super(OLSFPNEstimator, self).__init__()
        self.agg_size = agg_size
        self.alpha_so = alpha_so
        self.beta_so = beta_so

    def forward(self, tinput):
        # x: ir_estimation (ire), z: ir_noisy (irn) and x = a.z + b
        ire = tinput[:, 0:self.agg_size, :, :]
        irn = tinput[:, self.agg_size:, :, :]
        
        # get the means (BATCH x 1 x 288 x 288)
        mx = ire.mean(dim=1).unsqueeze(dim=1)
        mz = irn.mean(dim=1).unsqueeze(dim=1)
        
        # calculate sums of products for numerator and sums of squares for denominator
        numerator = ((irn - mz) * (ire - mx)).sum(dim=1)
        denominator = ((irn - mz) ** 2).sum(dim=1)

        # handling possible division by zero
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        # calculating a and b separately
        a = numerator / denominator
        b = mx.squeeze(dim=1) - a * mz.squeeze(dim=1)

        return torch.cat((a.unsqueeze(dim=1), b.unsqueeze(dim=1)), dim=1)
