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
@click.option('--multi/--uni', is_flag=True, default=True, help='unimodal input')
@click.option('--lb', is_flag=True, default=False, help='unimodal input')
def main(output_dir, start, stop, multi, lb):
    print(multi)

    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    sum_mse = 0.0
    sum_badpix = 0.0

    cnt = 0

    for i, scene in enumerate(scenes):
        # load ground truth modes, posterior and prediction
        modes = np.load(os.path.join(scene, 'gt_modes.npy'))
        pred = np.flip(load(os.path.join(scene, 'result.pfm')), 0)
        gt = np.flip(load(os.path.join(scene, 'gt.pfm')), 0)

        h, w, k = modes.shape

        if multi:
            # find all modes in posterior
            posterior = np.load(os.path.join(scene, 'posterior.npy'))
            post_modes = np.zeros_like(posterior)
            for y in range(h):
                for x in range(w):
                    for j in range(1, posterior.shape[0] - 1):
                        if posterior[j, y, x] > posterior[j - 1, y, x] and posterior[j, y, x] > posterior[j + 1, y, x]:

                            post_modes[j, y, x] = posterior[j, y, x]

        mse = np.zeros((h, w))
        badpix = np.zeros((h, w))

        for y in range(LOSS_MARGIN, h - LOSS_MARGIN):
            for x in range(LOSS_MARGIN, w - LOSS_MARGIN):
                if modes[y, x, 0] == modes[y, x, 1]:
                    if multi:
                        # multimodal
                        best = np.argpartition(post_modes[:, y, x], -k)[-k:]
                        disps = best.astype(
                            np.float) / float(posterior.shape[0] - 1) * \
                            (stop - start) + start

                        disps = np.sort(disps)

                        # compute metrics
                        if lb:
                            mse[y, x] = min((gt[y, x] - disps[0]) **
                                            2.0, (gt[y, x] - disps[1]) ** 2.0)
                            badpix[y, x] = min(
                                abs(gt[y, x] - disps[0]) > BAD_PIX_T, abs(gt[y, x] - disps[1]) > BAD_PIX_T)

                        else:
                            mse[y, x] = np.mean((disps - modes[y, x]) ** 2.0)
                            badpix[y, x] = np.mean(
                                np.abs(disps - modes[y, x]) > BAD_PIX_T).astype(
                                np.float)

                    else:
                        mse[y, x] = (gt[y, x] - pred[y, x]) ** 2.0
                        badpix[y, x] = abs(gt[y, x] - pred[y, x]) > BAD_PIX_T

                    cnt += 1

        imsave(f'mse_{i}.png', mse)
        imsave(f'badpix_{i}.png', badpix)

        print(np.mean(mse), np.mean(badpix))

        sum_mse += np.sum(mse)
        sum_badpix += np.sum(badpix)

    sum_mse /= float(cnt)
    sum_badpix /= float(cnt)

    print('MSE:', sum_mse)
    print('BadPix:', sum_badpix)


if __name__ == '__main__':
    sys.exit(main())
