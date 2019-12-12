import sys
import os

import click

import numpy as np

from ..utils import pfm


def masked_mse(input, target, mask):
    """
    Computed MSE loss only for pixels which are True in mask

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


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--step', default=0.01, help='Step size for sparsification.')
def main(output_dir, step):
    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    mse = np.zeros((3, int(1.0 / step) + 1))
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
            mse[0, i] = 1.0 - fract

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

            mse_oracle = masked_mse(result, gt, mask_oracle)
            mse_uncert = masked_mse(result, gt, mask_uncert)

            mse[1, i] += mse_oracle
            mse[2, i] += mse_uncert

    # fill in remaining and normalize
    mse[1:3] /= len(scenes)

    mse = mse[:, ::-1]

    with open(os.path.join(output_dir, 'sparsify.csv'), 'w') as f:
        # print and save result
        header = 'frac, mse_oracle, mse_uncert'
        print(header)
        print(header, file=f)
        for i in range(mse.shape[1]):
            line = f'{mse[0, i]:.2f}, {mse[1, i]:.8f}, {mse[2, i]:.8f}'
            print(line)
            print(line, file=f)


if __name__ == '__main__':
    sys.exit(main())
