import sys
import os

import numpy as np
from skimage.io import imread, imsave
from scipy.ndimage import gaussian_filter1d

import click


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
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
def main(output_dir, x, y, start, stop):
    input = os.path.join(output_dir, 'posterior.npy')
    output = os.path.join(output_dir, f'posterior_{x}_{y}.csv')

    posterior = np.load(input)

    # posterior = gaussian_filter1d(posterior, sigma=2, axis=0)

    num_samples = posterior.shape[0]
    posterior = posterior[:, y, x]
    posterior /= auc(posterior, (stop - start) / float(num_samples))

    with open(output, 'w') as f:
        f.write(f'y, p\n')
        for i in range(num_samples):
            disp = float(i) / float(num_samples - 1) * (stop - start) + start
            prob = posterior[i]
            f.write(f'{disp}, {prob}\n')

    # save center view with pixel marked
    center = imread(os.path.join(output_dir, 'center.png'))
    center[y, x] = np.asarray([255.0, 0.0, 0.0])
    imsave(os.path.join(output_dir, f'center_{x}_{y}.png'), center)


if __name__ == '__main__':
    sys.exit(main())
