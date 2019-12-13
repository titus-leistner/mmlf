import sys
import os

import click

import numpy as np

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

    return np.sum(diff * mask.astype(np.float)) / count


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

    return np.sum(diff * mask.astype(np.float)) / count


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
        auc += (curve[i] + curve[i+1]) / 2.0 * step

    return auc


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--step', default=0.01, help='Step size for sparsification.')
@click.option('--mse/--l1', default=True, help='Use MSE or L1 loss?')
def main(output_dir, step, mse):
    # set loss function
    if mse:
        loss_fn = masked_mse
    else:
        loss_fn = masked_l1

    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    loss = np.zeros((3, int(1.0 / step) + 1))
    for scene in scenes:
        gt = pfm.load(os.path.join(scene, 'gt.pfm')).flatten()
        result = pfm.load(os.path.join(scene, 'result.pfm')).flatten()
        uncert = pfm.load(os.path.join(scene, 'uncert.pfm')).flatten()

        # compute error
        error = np.abs(result - gt)

        # create masks
        mask_oracle = np.zeros_like(gt, dtype=np.bool)
        mask_uncert = np.zeros_like(gt, dtype=np.bool)

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

            loss_oracle = loss_fn(result, gt, mask_oracle)
            loss_uncert = loss_fn(result, gt, mask_uncert)

            loss[1, i] += loss_oracle
            loss[2, i] += loss_uncert

    # fill in remaining and normalize
    loss[1:3] /= np.max(loss[1:3])

    # reverse order
    loss = loss[:, ::-1]

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
