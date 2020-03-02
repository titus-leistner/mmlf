import sys
import os
import math

import click

import numpy as np
from scipy.ndimage import sobel
from skimage.io import imsave
from sklearn.cluster import KMeans

from ..utils import pfm


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--radius', default=2.0, help='Radius of neighborhood')
@click.option('-k', default=2, help='Number of modes')
def main(output_dir, radius, k):
    scenes = [f.path for f in os.scandir(
        os.path.join(output_dir, 'scenes')) if f.is_dir()]

    for i, scene in enumerate(scenes):
        # load ground truth
        gt = pfm.load(os.path.join(scene, 'gt.pfm'))
        gt = np.flip(gt, 0)

        # approximate derivation with sobel filter
        der = np.sqrt(sobel(gt, 0) ** 2.0 + sobel(gt, 1) ** 2.0)
        edges = (der > 0.5)

        modes = np.zeros((gt.shape[0], gt.shape[1], 2))

        # detect modes
        for y in range(gt.shape[0]):
            print(y)
            for x in range(gt.shape[1]):
                if not edges[y, x]:
                    # no edge, only single mode
                    modes[y, x, :] = gt[y, x]

                else:
                    # edge detected
                    # find two modes from neigborhood, using K-Means
                    # get neighborhood
                    disps = []
                    radius_int = math.ceil(radius)
                    for dy in range(-radius_int, radius_int + 1):
                        for dx in range(-radius_int, radius_int + 1):
                            if (dy**2.0 + dx**2.0) ** 0.5 <= radius:
                                sy = max(0, min(gt.shape[0] - 1, y + dy))
                                sx = max(0, min(gt.shape[1] - 1, x + dx))
                                disps.append(gt[sy, sx])

                    # KMeans
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(np.asarray(disps).reshape((-1, 1)))

                    # assign
                    modes[y, x] = np.sort(kmeans.cluster_centers_.reshape(-1))

                    # print(kmeans.cluster_centers_)

        np.save(os.path.join(scene, 'gt_modes.npy'), modes)
        for i in range(k):
            imsave(f'mode_{i}.png', modes[:, :, i])


if __name__ == '__main__':
    sys.exit(main())
