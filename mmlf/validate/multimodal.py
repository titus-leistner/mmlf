import sys
import os

import click

import numpy as np
from skimage.io import imsave

from ..utils.pfm import load

BAD_PIX_T = 0.07
LOSS_MARGIN = 15


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
def main(output_dir, start, stop):
    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    mean_mse = 0.0
    mean_badpix = 0.0

    for i, scene in enumerate(scenes):
        # load ground truth modes, posterior and prediction
        modes = np.load(os.path.join(scene, 'gt_modes.npy'))
        posterior = np.load(os.path.join(scene, 'posterior.npy'))
        pred = np.flip(load(os.path.join(scene, 'result.pfm')), 0)
        gt = np.flip(load(os.path.join(scene, 'gt.pfm')), 0)

        h, w, k = modes.shape

        # find all modes in posterior
        post_modes = np.zeros_like(posterior)
        for y in range(h):
            for x in range(w):
                for i in range(1, posterior.shape[0] - 1):
                    if posterior[i, y, x] > posterior[i - 1, y, x] and \
                       posterior[i, y, x] > posterior[i + 1, y, x]:

                        post_modes[i, y, x] = posterior[i, y, x]

        mse = np.zeros((h, w))
        badpix = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                if modes[y, x, 0] == modes[y, x, 1]:
                    mse[y, x] = (pred[y, x] - gt[y, x]) ** 2.0
                    badpix[y, x] = np.abs(pred[y, x] - gt[y, x]) > BAD_PIX_T

                else:
                    # multimodal
                    best = np.argpartition(post_modes[:, y, x], -k)[-k:]
                    disps = best.astype(
                        np.float) / float(posterior.shape[0] - 1) * \
                        (stop - start) + start

                    disps = np.sort(disps)

                    # compute metrics
                    mse[y, x] = np.sum((disps - modes[y, x]) ** 2.0) / float(k)
                    badpix[y, x] = np.sum(
                        np.abs(disps - modes[y, x]) > BAD_PIX_T).astype(
                        np.float) / float(k)

        # remove margin
        mse = mse[LOSS_MARGIN:-LOSS_MARGIN, LOSS_MARGIN:-LOSS_MARGIN]
        badpix = badpix[LOSS_MARGIN:-LOSS_MARGIN, LOSS_MARGIN:-LOSS_MARGIN]

        imsave('mse.png', mse)
        imsave('badpix.png', badpix)

        print(np.mean(mse), np.mean(badpix))

        mean_mse += np.mean(mse)
        mean_badpix += np.mean(badpix)

    mean_mse /= float(len(scenes))
    mean_badpix /= float(len(scenes))

    print('MSE:', mean_mse)
    print('BadPix:', mean_badpix)


if __name__ == '__main__':
    sys.exit(main())
