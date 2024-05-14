import torch

# inserts the given FPN pattern onto the image
def FPNInsert(img, fpn):
    return torch.clamp((img - fpn[1, :, :]) / fpn[0, :, :], min=0.0, max=1.0)

# removes the given FPN pattern from the image
def FPNRemove(img, fpn):
    return torch.clamp(img * fpn[0, :, :] + fpn[1, :, :], min=0.0, max=1.0)

def FPNRemoveTensor(irn, fpn):
    
    # stack in the first and second dimension
    irs = irn.view(-1, irn.shape[2], irn.shape[3], irn.shape[4])
    fpn = fpn.view(-1, fpn.shape[2], fpn.shape[3], fpn.shape[4])
    
    # apply the function to each image in the batch
    irc = torch.stack([FPNRemove(irs[idx], fpn[idx]) for idx in range(irs.shape[0])])
    
    # reshape to the original size
    return irc.view_as(irn)

def FPNNormalize(fpn, ab_mins, ab_maxs):
    
    # prevent zero divisions on alpha
    if ab_mins[0] == ab_maxs[0]:
        fpn_as = 0.5 * torch.ones_like(fpn[0, :, :])
    else:
        fpn_as = (fpn[0, :, :] - ab_mins[0]) / (ab_maxs[0]-ab_mins[0])

    # prevent zero divisions on beta
    if ab_mins[1] == ab_maxs[1]:
        fpn_bs = 0.5 * torch.ones_like(fpn[1, :, :])
    else:
        fpn_bs = (fpn[1, :, :] - ab_mins[1]) / (ab_maxs[1]-ab_mins[1])
    
    # stack the channels back together
    return torch.stack((fpn_as, fpn_bs), dim=0)