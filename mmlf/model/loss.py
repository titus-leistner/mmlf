import torch
import torch.nn as nn


def create_mask_margin(shape, margin=0):
    """
    Create a mask with a False margin

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


def create_mask_texture(center, wsize, threshold):
    """
    Create a mask with False values for each pixel with a mean L1 distance to
    all neighboring pixels in a rolling window lower than a given threshold

    This implicitely adds a margin of wsize // 2 to the mask

    :param center: the center view
    :type center: torch.Tensor

    :param wsize: the window size
    :type wsize: int

    :param threshold: mean L1 threshold
    :type threshold: float
    """
    b, w, h = center.shape[0], center.shape[-1], center.shape[-2]

    # unfold and reshape to image
    mask = nn.functional.unfold(center, kernel_size=wsize, padding=wsize//2)
    mask = mask.view(b, 3, -1, h, w)

    # subtract the center pixel and compute the MAE
    mask = torch.abs(mask - center.unsqueeze(2)).mean((1, 2))

    # apply the threshold
    mask = mask >= threshold

    # also mask the boundary
    mask = (mask.int() * create_mask_margin(mask.shape, wsize//2).int()).bool()

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
        diff = torch.abs(torch.flatten(input) - torch.flatten(target)) > self.t
        count = mask.int().sum()

        diff = diff.int() * torch.flatten(mask).int()

        return diff.sum().float() / count


class UncertaintyMSELoss(nn.Module):
    """
    Apply an MSE loss with uncertainty introduced by Kendall and Gal
    """
    def __init__(self):
        super(UncertaintyMSELoss, self).__init__()

    def forward(self, input, target, uncertainty):
        # compute loss with uncertainty
        loss = 0.5 * torch.exp(-uncertainty) * (input - target) ** 2.0

        # add uncertainty
        loss += 0.5 * uncertainty

        # mean
        loss = torch.mean(loss)

        return loss


class UncertaintyL1Loss(nn.Module):
    """
    Apply an L1 loss with uncertainty
    """
    def __init__(self):
        super(UncertaintyL1Loss, self).__init__()

    def forward(self, input, target, uncertainty):
        # compute loss with uncertainty
        loss = 0.5 * torch.exp(-uncertainty) * torch.abs(input - target)

        # add uncertainty
        loss += 0.5 * uncertainty

        # mean
        loss = torch.mean(loss)

        return loss
