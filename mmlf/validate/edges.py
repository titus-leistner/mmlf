import sys
import os

import click

import numpy as np
from scipy.ndimage import sobel
from skimage.io import imsave

from ..utils import pfm


@click.command()
@click.argument('dataset', type=click.Path(exists=True))
def main(dataset):
    scenes = [f.path for f in os.scandir(dataset) if f.is_dir()]

    for i, scene in enumerate(scenes):
        # load ground truth
        gt = pfm.load(os.path.join(scene, 'gt_disp_lowres.pfm'))
        gt = np.flip(gt, 0)

        # approximate derivation with sobel filter
        der = np.sqrt(sobel(gt, 0) ** 2.0 + sobel(gt, 1) ** 2.0)
        edges = (der > 0.5).astype(np.float)
        imsave(os.path.join(scene, 'edges.png'), edges)


if __name__ == '__main__':
    sys.exit(main())
