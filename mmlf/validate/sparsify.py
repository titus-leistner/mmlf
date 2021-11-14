import sys
import os

import click

import numpy as np
from skimage.io import imread, imsave

from ..utils import pfm


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

    return np.sum(diff * mask.astype(float)) / count


def masked_l1(input, target, mask):
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
    count = np.sum(mask)

    return np.sum(diff * mask.astype(float)) / count


def masked_badpix(input, target, mask, threshold=0.07):
    """
    Comput L1 loss only for pixels which are True in mask

    :param input: the prediction
    :type input: numpy.ndarray

    :param target: the ground truth
    :type target: numpy.ndarray

    :param mask: the mask
    :type mask: numpy.ndarray(dtype=numpy.bool)
    """
    diff = (np.abs(input - target) > 0.07).astype(float)
    count = np.sum(mask)

    return np.sum(diff * mask.astype(float)) / count


def auc(curve, step):
    """
    Compute area under the curve

    :param curve: 1D list of y values
    :type curve: list or numpy.ndarray(dtype=numpy.float)

    :param step: step size in x-dimension
    :type step: float
    """
    auc = 0.0
    for i in range(len(curve) - 1):
        auc += (curve[i] + curve[i + 1]) / 2.0 * step

    return auc


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--step', default=0.01, help='Step size for sparsification.')
@click.option('--mse/--badpix', default=True, help='Use MSE or L1 loss?')
@click.option('--random', is_flag=True, default=False, help='Use Random Baseline?')
def main(output_dir, step, mse, random):
    # set loss function
    if mse:
        loss_fn = masked_mse
    else:
        loss_fn = masked_badpix

    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    loss = np.zeros((3, int(1.0 / step) + 1))
    for scene in scenes:
        gt = pfm.load(os.path.join(scene, 'gt.pfm')).flatten()
        result = pfm.load(os.path.join(scene, 'result.pfm')).flatten()
        uncert = pfm.load(os.path.join(scene, 'uncert.pfm')).flatten()
        img = np.flip(imread(os.path.join(scene, 'center.png')), 0)
        img_out = np.zeros_like(img)
        img_out[:, :] = np.asarray([255, 0, 0])

        img_out_oracle = np.zeros_like(img)
        img_out_oracle[:, :] = np.asarray([255, 0, 0])

        if random:
            print('Use Random')
            uncert = np.random.random(uncert.size)

        # compute error
        error = np.abs(result - gt)

        # create masks
        mask_oracle = np.zeros_like(gt, dtype=bool)
        mask_uncert = np.zeros_like(gt, dtype=bool)

        # sparsification
        for i, fract in enumerate(np.arange(0.0, 1.000000001, step)):
            loss[0, i] = 1.0 - fract

            if i == 0:
                continue
            elif fract == 1.0:
                mask_oracle[...] = True
                mask_uncert[...] = True
            else:
                k = int(fract * np.size(gt))
                idx_oracle = np.argpartition(error, k)[:k]
                idx_uncert = np.argpartition(uncert, k)[:k]

                mask_oracle[idx_oracle] = True
                mask_uncert[idx_uncert] = True

            print(i)
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if mask_uncert[y * img.shape[1] + x]:
                        img_out[y, x, :] = img[y, x, :]
                    if mask_oracle[y * img.shape[1] + x]:
                        img_out_oracle[y, x, :] = img[y, x, :]

            border = 32
            out = np.zeros((img.shape[0], 2 * img.shape[1] + border, 3))
            out[:, 0:img.shape[1], :] = img_out
            out[:, img.shape[1] + border:, :] = img_out_oracle
            out = np.pad(out, ((104, 104), (112, 112), (0, 0)))

            # imsave(os.path.join(scene, f'sparse_{i:04d}.png'), np.flip(out, 0))

            loss_oracle = loss_fn(result, gt, mask_oracle)
            loss_uncert = loss_fn(result, gt, mask_uncert)

            loss[1, i] += loss_oracle
            loss[2, i] += loss_uncert

    # reverse order
    loss = loss[:, ::-1]

    # fill in remaining and normalize
    loss[1:3] /= loss[1, 0]

    # delete last element (1.0 makes no sense)
    loss = np.delete(loss, -1, axis=1)

    sparse_error = loss[2] - loss[1]

    with open(os.path.join(output_dir, 'sparsify.csv'), 'w') as f:
        # print and save result
        header = 'frac,     oracle,     uncert, sparse_err'
        print(header)
        print(header, file=f)
        for i in range(loss.shape[1]):
            line = f'{loss[0, i]:.2f}, {loss[1, i]:.8f}, {loss[2, i]:.8f}, ' \
                   f'{sparse_error[i]:.8f}'
            print(line)
            print(line, file=f)

    print('----------------------------------------')
    # compute area under the curve
    print('AUC: ', auc(sparse_error, step))


if __name__ == '__main__':
    sys.exit(main())
