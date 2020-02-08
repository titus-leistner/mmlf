import sys
import os

import numpy as np
from skimage.io import imsave

import click

from . import pfm


def gaussian(x, mean, var):
    y = 1.0 / np.sqrt(2.0 * np.pi * var) * \
        np.exp(-(x - mean) ** 2.0 / (2.0 * var))

    return y


def create_mask_margin(shape, margin=15):
    """
    Create a mask with a False margin

    :param shape: shape of mask
    :type shape: tuple

    :param margin: margin for last two dimenstions which gets assigned False
    :type margin: int
    """
    assert margin >= 0

    mask = np.ones(shape, dtype=np.bool)

    if margin > 0:
        mask[..., :margin, :] = False
        mask[..., -margin:, :] = False
        mask[..., :margin] = False
        mask[..., -margin:] = False

    return mask


def masked_mse(input, target, mask):
    """
    Comput MSE loss only for pixels which are True in mask

    :param input: the prediction
    :type input: numpy.ndarray

    :param target: the ground truth
    :type target: numpy.ndarray

    :param mask: the mask
    :type mask: numpy.ndarray(dtype=numpy.bool)
    """
    diff = (input - target) ** 2.0
    count = np.sum(mask)

    return np.sum(diff * mask.astype(np.float)) / count


def masked_badpix(input, target, mask, t=0.07):
    """
    Comput L1 loss only for pixels which are True in mask

    :param input: the prediction
    :type input: numpy.ndarray

    :param target: the ground truth
    :type target: numpy.ndarray

    :param mask: the mask
    :type mask: numpy.ndarray(dtype=numpy.bool)
    """
    diff = np.abs(input - target)
    diff = (diff > t).astype(np.float)
    count = np.sum(mask)

    return np.sum(diff * mask.astype(np.float)) / count


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
@click.option('--step', default=0.005, help='step width')
def main(input, output, **kwargs):
    gmm = np.load(os.path.join(input, 'gmm.npy'))
    w = gmm.shape[-1]
    h = gmm.shape[-2]

    cnts = np.zeros((h, w))
    mode_min = np.zeros((h, w))
    mode_max = np.zeros((h, w))

    # load gt and result for mse and badpix computation
    gt = pfm.load(os.path.join(input, 'gt.pfm'))
    gt = np.flip(gt, 0)

    result = pfm.load(os.path.join(input, 'result.pfm'))
    result = np.flip(result, 0)

    for y in range(h):
        for x in range(w):
            cnts[y, x], modes = sum(gmm, x, y, **kwargs)
            if cnts[y, x] > 0:
                mode_min[y, x] = modes[0]
                mode_max[y, x] = modes[-1]

            # compute metrics
            res_diff = abs(gt[y, x] - result[y, x])
            min_diff = abs(gt[y, x] - mode_min[y, x])
            max_diff = abs(gt[y, x] - mode_max[y, x])

            if res_diff > min_diff or res_diff > max_diff:
                print("better")
                if min_diff < max_diff:
                    result[y, x] = mode_min[y, x]
                else:
                    result[y, x] = mode_max[y, x]
            else:
                print("worse")

            print((x, y), cnts[y, x], mode_min[y, x], mode_max[y, x])

        # output cnt
        cnts_out = cnts / np.max(cnts)
        imsave(os.path.join(input, "cnts.png"), cnts_out)

        # output distances
        dist_out = np.zeros((h, w, 4))
        dist_out[:, :, 0] = 1.0
        dists_norm = np.abs(mode_min - mode_max)
        dists_norm /= np.max(dists_norm)
        dist_out[:, :, 3] = dists_norm

        imsave(os.path.join(input, "dist.png"), dist_out)

        # output min and max modes
        disp_min = min(np.min(mode_min), np.min(mode_max))
        disp_max = max(np.max(mode_min), np.max(mode_max))
        mode_min_out = (mode_min - disp_min) / (disp_max - disp_min)
        mode_max_out = (mode_max - disp_min) / (disp_max - disp_min)

        imsave(os.path.join(input, 'result_min.png'), mode_min_out)
        imsave(os.path.join(input, 'result_max.png'), mode_max_out)
        imsave(os.path.join(input, 'result_best.png'), result)

        mask = create_mask_margin(result.shape)
        print(mask)

        with open(os.path.join(input, 'second_chance.txt'), 'w') as f:
            print("MSE:", masked_mse(result, gt, mask), file=f)
            print("BadPix:", masked_badpix(result, gt, mask), file=f)


def sum(gmm, x, y, start, stop, step):
    means = gmm[0, :, y, x]
    vars = gmm[1, :, y, x]
    num_gs = means.shape[0]

    ys = []
    for x in np.arange(start, stop, step):
        y = 0.0
        for i in range(num_gs):
            y += gaussian(x, means[i], vars[i]) / vars[i]

        ys.append(y)

    cnt = 0
    modes = []
    for i in range(1, len(ys) - 1):
        if ys[i - 1] < ys[i] and ys[i + 1] < ys[i]:
            cnt += 1
            modes.append(i * step + start)

    modes.sort()
    return cnt, modes


if __name__ == '__main__':
    sys.exit(main())
