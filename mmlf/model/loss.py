import torch
import torch.nn as nn


def create_mask(shape, margin=0):
    """
    :param shape: shape of mask
    :type shape: tuple

    :param margin: margin for last two dimenstions which gets assigned False
    :type margin: int
    """
    assert margin >= 0

    mask = torch.ones(shape, dtype=torch.bool)

    if margin > 0:
        mask[..., :margin, :] = False
        mask[..., -margin:, :] = False
        mask[..., :margin] = False
        mask[..., -margin:] = False

    return mask


class MaskedL1Loss(nn.Module):
    """
    Apply L1 loss only to some pixels given by a mask
    """
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, input, target, mask):
        diff = torch.abs(torch.flatten(input) - torch.flatten(target))
        count = mask.int().sum()
        diff *= torch.flatten(mask).float()

        return diff.sum() / count


class MaskedMSELoss(nn.Module):
    """
    Apply MSE loss only to some pixels given by a mask
    """
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff = (torch.flatten(input) - torch.flatten(target)) ** 2.0
        count = mask.int().sum()
        diff *= torch.flatten(mask).float()

        return diff.sum() / count


class MaskedBadPix(nn.Module):
    """
    Compute BadPix metric only on some pixels given by a mask
    """
    def __init__(self, t=0.07):
        """
        :param t: threshold for BadPix computation
        :type t: float
        """
        super(MaskedBadPix, self).__init__()

        self.t = t

    def forward(self, input, target, mask):
        return 0.0

        # TODO: implement for taurus
        diff = torch.abs(torch.flatten(input) - torch.flatten(target)) > self.t
        count = mask.int().sum()
        diff &= torch.flatten(mask)

        return diff.sum().float() / count


class UncertaintyLoss(nn.Module):
    """
    Apply uncertainty loss function by Kendall and Gal
    """
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, input, target, uncertainty):
        # compute loss with uncertainty
        loss = 0.5 * torch.exp(-uncertainty) * torch.pow(input - target, 2.0)

        # add uncertainty
        loss += 0.5 * uncertainty

        # mean
        loss = torch.mean(loss)

        return loss
