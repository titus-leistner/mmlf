import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MaskedL1Loss(nn.Module):
    """
    Apply L1 loss only to some pixels given by a mask
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.in_lz = 0
        self.in_gz = 0
        self.in_mean = 0.0

        self.gt_lz = 0
        self.gt_gz = 0
        self.gt_mean = 0.0

        self.count = 0

    def forward(self, input, target, mask):
        # from ..utils.dl import save_img
        # save_img('out/pred.png', input['mean'][0])
        # save_img('out/mask.png', mask[0].float())

        # self.count += 1
        # print('in:  ', torch.min(input['mean']).item(), torch.max(input['mean']).item())
        # print('inm: ', torch.min(input['mean'] * mask).item(), torch.max(input['mean'] * mask).item())
        # self.in_lz += (torch.sum(input['mean'] < 0.0)).item()
        # self.in_gz += (torch.sum(input['mean'] > 0.0)).item()
        # self.in_mean += torch.mean(input['mean']).item()
        # print('mn:  ', self.in_mean / (self.count + 1e-12))
        # print('-/+: ', self.in_lz / (self.in_gz + 1e-12))
        # print()

        # print('gt:  ', torch.min(target).item(), torch.max(target).item())
        # print('gtm: ', torch.min(target * mask).item(), torch.max(target * mask).item())
        # self.gt_lz += (torch.sum(target < 0.0)).item()
        # self.gt_gz += (torch.sum(target > 0.0)).item()
        # self.gt_mean += torch.mean(target).item()
        # print('mn:  ', self.gt_mean / (self.count + 1e-12))
        # print('-/+: ', self.gt_lz / (self.gt_gz + 1e-12))
        # print()

        diff = torch.abs(torch.flatten(input['mean']) - torch.flatten(target))
        count = mask.int().sum()
        diff *= torch.flatten(mask).float()

        if count == 0:
            return diff.sum()

        return diff.sum() / count


class MultiMaskedL1Loss(nn.Module):
    """
    Apply L1 loss only to some pixels given by a mask
    """

    def __init__(self):
        super(MultiMaskedL1Loss, self).__init__()

    def forward(self, input, target, mask):
        weights = target[:, :, 3, :, :]
        targets = target[:, :, 4, :, :]
        diff = torch.abs(input['mean'].unsqueeze(1) - targets)
        diff *= weights
        diff = torch.sum(diff, dim=1)

        diff = torch.flatten(diff)

        count = mask.int().sum()
        diff *= torch.flatten(mask).float()

        if count == 0:
            return diff.sum()

        return diff.sum() / count


class MaskedMSELoss(nn.Module):
    """
    Apply MSE loss only to some pixels given by a mask
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff = (torch.flatten(input['mean']) - torch.flatten(target)) ** 2.0
        count = mask.int().sum()
        diff *= torch.flatten(mask).float()

        if count == 0:
            return diff.sum()

        return diff.sum() / count


class MultiMaskedMSELoss(nn.Module):
    """
    Apply MSE loss only to some pixels given by a mask
    """

    def __init__(self):
        super(MultiMaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        raise NotImplementedError()


class MaskedCrossEntropy(nn.Module):
    """
    Apply Cross Entropy loss only to some pixels given by a mask
    """

    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    def forward(self, input, target, mask):
        scores = F.relu(input['scores'])
        loss = torch.exp(torch.sum(scores * target, 1))
        # print('logit mean:', torch.mean(scores).item())
        loss = loss / torch.sum(torch.exp(scores), 1)
        # print('loss:', loss)
        loss = -torch.log(loss)

        # apply mask
        count = mask.float().sum()
        loss *= mask.float()

        if count == 0:
            return loss.sum()

        return loss.sum() / count


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
        diff = torch.abs(torch.flatten(
            input['mean']) - torch.flatten(target)) > self.t
        count = mask.int().sum()

        diff = diff.int() * torch.flatten(mask).int()

        if count == 0:
            return diff.sum()

        return diff.sum().float() / count


class UncertaintyMSELoss(nn.Module):
    """
    Apply an MSE loss with uncertainty introduced by Kendall and Gal
    """

    def __init__(self):
        super(UncertaintyMSELoss, self).__init__()

    def forward(self, input, target, mask):
        # compute loss with uncertainty
        loss = 0.5 * torch.exp(-input['logvar']) * \
            (input['mean'] - target) ** 2.0

        # add uncertainty
        loss += 0.5 * input['logvar']

        # multiply with mask
        count = mask.int().sum()
        loss *= mask.float()

        if count == 0:
            return loss.sum()

        return loss.sum() / count


class MultiUncertaintyMSELoss(nn.Module):
    """
    Apply an MSE loss with uncertainty introduced by Kendall and Gal
    """

    def __init__(self):
        super(MultiUncertaintyMSELoss, self).__init__()

    def forward(self, input, target, mask):
        raise NotImplementedError()


class UncertaintyL1Loss(nn.Module):
    """
    Apply an L1 loss with uncertainty
    """

    def __init__(self):
        super(UncertaintyL1Loss, self).__init__()

    def forward(self, input, target, mask):
        # compute loss with uncertainty
        loss = torch.exp(-input['logvar']) * \
            torch.abs(input['mean'] - target)

        # add uncertainty
        loss += input['logvar']

        # multiply with mask
        count = mask.int().sum()
        loss *= mask.float()

        if count == 0:
            return loss.sum()

        return loss.sum() / count


class MultiUncertaintyL1Loss(nn.Module):
    """
    Apply an L1 loss with uncertainty
    """

    def __init__(self):
        super(MultiUncertaintyL1Loss, self).__init__()

    def forward(self, input, target, mask):
        # compute loss with uncertainty
        weights = target[:, :, 3, :, :]
        targets = target[:, :, 4, :, :]

        loss = torch.exp(-input['logvar']).unsqueeze(1) * \
            torch.abs(input['mean'].unsqueeze(1) - targets)
        loss *= weights
        loss = torch.sum(weights, dim=1)

        # add uncertainty
        loss += input['logvar']

        # multiply with mask
        count = mask.int().sum()
        loss *= mask.float()

        if count == 0:
            return loss.sum()

        return loss.sum() / count


class InformationBottleneckLoss(nn.Module):
    """
    Apply an L1 loss with uncertainty
    """

    def __init__(self, beta):
        """
        :param beta: loss weights
        :type beta: float or None for beta = inf
        """
        super(InformationBottleneckLoss, self).__init__()
        self.beta_nll = 1.0 / (1.0 + beta)
        self.beta_cat_ce = 1.0 * beta / (1.0 + beta)

    def forward(self, input, target, mask):
        zixels = input['zixels']
        jac = input['jac']
        mu = input['mu']
        dists = input['dists']

        # print('mean dist:', torch.mean(dists).item())
        # print('mean absolute mu:', torch.mean(torch.abs(mu)).item())
        # print('mean absolute z:', torch.mean(torch.abs(zixels)).item())

        w = zixels.shape[-1]
        h = zixels.shape[-2]
        dims = mu.shape[-1]

        jac = jac.view(-1, 1, 1) / (dims * w * h)

        nll = ((- torch.logsumexp(- 0.5 * dists, dim=1)) - jac) / dims

        cat_ce = - \
            torch.sum((torch.log_softmax(- 0.5 * dists, dim=1)) * target,
                      dim=1)
        nll = nll.mean()
        cat_ce = cat_ce.mean()

        # print(f'nll: {nll.item()}, cat_ce: {cat_ce.item()}')

        loss = self.beta_nll * nll + self.beta_cat_ce * cat_ce

        return loss
