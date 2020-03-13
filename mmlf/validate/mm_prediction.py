import sys
import os

import click

import numpy as np
from skimage.io import imread, imsave

from ..utils import pfm


def loss_fn(input, target):
    hit = np.sum(input.astype(np.float) * target.astype(np.float))

    return 1.0 - (hit / np.sum(target.astype(np.float)))


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
@click.option('--random', is_flag=True, default=False, help='Use Random Baseline?')
def main(output_dir, step, random):
    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    loss = np.zeros((3, int(1.0 / step) + 1))
    for scene in scenes:
        gt_modes = np.load(os.path.join(scene, 'gt_modes.npy'))
        mask_gt = (gt_modes[:, :, 0] != gt_modes[:, :, 1]).flatten()

        mode_prop = np.flip(pfm.load(os.path.join(
            scene, 'mode_prop.pfm')), 0).flatten()

        img = imread(os.path.join(scene, 'center.png'))
        img_out = img.copy()

        img_out_oracle = img.copy()

        # compute error

        error = ~mask_gt

        if random:
            print('Use Random')
            mode_prop = np.random.random(mode_prop.size)
        # imsave('mask_gt.png', np.reshape(mask_gt, (512, 512)).astype(np.float))

        # create masks
        mask_oracle = np.zeros_like(mask_gt, dtype=np.bool)
        mask_pred = np.zeros_like(mask_gt, dtype=np.bool)

        # sparsification
        for i, fract in enumerate(np.arange(0.0, 1.000000001, step)):
            loss[0, i] = 1.0 - fract

            if fract == 1.0:
                mask_oracle[...] = True
                mask_pred[...] = True
            else:
                k = int(fract * np.size(mask_gt))
                idx_oracle = np.argpartition(error, k)[:k]
                idx_pred = np.argpartition(-mode_prop, k)[:k]

                mask_oracle[idx_oracle] = True
                mask_pred[idx_pred] = True

                # imsave(f'oracle_{i}.png', np.reshape(mask_oracle.astype(np.float), (512, 512)))
                # imsave(f'pred_{i}.png', np.reshape(mask_pred.astype(np.float), (512, 512)))

            loss_oracle = loss_fn(mask_oracle, mask_gt)
            loss_pred = loss_fn(mask_pred, mask_gt)

            #print(loss_oracle, loss_pred)

            
            print(i)
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if mask_pred[y * img.shape[1] + x]:
                        img_out[y, x, :] = np.asarray([255, 0, 0])
                    if mask_oracle[y * img.shape[1] + x]:
                        img_out_oracle[y, x, :] = np.asarray([255, 0, 0])
            
            border = 32
            out = np.zeros((img.shape[0], 2 * img.shape[1] + border, 3))
            out[:, 0:img.shape[1], :] = img_out
            out[:, img.shape[1] + border:, :] = img_out_oracle
            out = np.pad(out, ((104, 104), (112, 112), (0, 0)))

            imsave(os.path.join(scene, f'mm_{i:04d}.png'), out)

            
            print(i, loss_oracle)
            loss[1, i] += loss_oracle
            loss[2, i] += loss_pred

    # reverse order
    # loss = loss[:, ::-1]
    loss[0] = 1.0 - loss[0]

    # fill in remaining and normalize
    print(loss[1, 0])
    loss[1:3] /= loss[1, 0]

    # delete last element (1.0 makes no sense)
    loss = np.delete(loss, -1, axis=1)

    sparse_error = loss[2] - loss[1]

    with open(os.path.join(output_dir, 'mm_pred.csv'), 'w') as f:
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
