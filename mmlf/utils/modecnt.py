import sys
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.io import imsave

import click

from ..utils import pfm


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--outlier', default=0.1, help='proporional outlier threshold')
def main(output_dir, outlier):
    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    for scene in scenes:
        print(f'Processing scene {scene}...')
        # load and convolve posterior
        posterior = np.load(os.path.join(scene, 'posterior.npy'))
        posterior = gaussian_filter1d(posterior, sigma=2, axis=0)

        n, h, w = posterior.shape

        # compute multimodality
        mode_prop = np.zeros((h, w))
        mode_cnt = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                # count modes
                mins = []
                maxs = []
                for i in range(1, n - 1):
                    left = posterior[i - 1, y, x]
                    center = posterior[i, y, x]
                    right = posterior[i + 1, y, x]

                    if left < center and right < center:
                        maxs.append((i, center))
                    elif left > center and right > center:
                        mins.append((i, center))

                # reject outliers
                min_clean = []
                max_clean = []

                maxs = sorted(maxs, key=lambda e: e[1])
                mins = sorted(mins, key=lambda e: e[1])
                for e in maxs:
                    if e[1] > maxs[-1][1] * outlier:
                        max_clean.append(e)

                for e in mins:
                    if e[1] < mins[0][1] / outlier:
                        min_clean.append(e)

                mode_cnt[y, x] = len(max_clean) > 1

                if len(max_clean) > 1:
                    top_max = sorted(max_clean, key=lambda e: e[1])[-2:]

                    # find minimum between two highest maxima
                    interval = sorted([top_max[0][0], top_max[1][0]])
                    top_min = [e[1] for e in mins if e[0] >
                               interval[0] and e[0] < interval[1]]
                    if len(top_min) > 0:
                        top_min = min(top_min)

                        # compute proportion of smallest maximum
                        # and minimum between the two modes
                        mode_prop[y, x] = top_max[0][1] / top_min

        imsave(os.path.join(scene, 'mode_cnt.png'), mode_cnt)
        imsave(os.path.join(scene, 'mode_prop.png'),
               np.clip(mode_prop, 0, 10))

        print(mode_prop.dtype)
        pfm.save(os.path.join(scene, 'mode_prop.pfm'),
                 np.flip(mode_prop.astype(np.float32), 0))


if __name__ == '__main__':
    sys.exit(main())
