import torch

def anatomy_consistency_loss(x, y, noise):
    noise[noise != 0] = 1
    masked_x = x * (1 - noise)
    masked_y = y * (1 - noise)
    l2_distance = torch.pow(masked_x - masked_y, 2)
    return l2_distance.mean()

def masked_l1_loss(pred, gt, mask):
    masked_pred = (1 - mask) * pred
    masked_gt = (1 - mask) * gt
    l1_distance = torch.abs(masked_pred - masked_gt)
    return l1_distance.mean(), l1_distance