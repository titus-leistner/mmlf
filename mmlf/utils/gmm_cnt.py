import sys

import numpy as np
from skimage.io import imsave

import click


def gaussian(x, mean, var):
    y = 1.0 / np.sqrt(2.0 * np.pi * var) * \
        np.exp(-(x - mean) ** 2.0 / (2.0 * var))

    return y


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
@click.option('--step', default=0.005, help='step width')
def main(input, output, **kwargs):
    gmm = np.load(input)
    w = gmm.shape[-1]
    h = gmm.shape[-2]

    cnts = np.zeros((h, w))
    mode_min = np.zeros((h, w))
    mode_max = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            cnts[y, x], modes = sum(gmm, x, y, **kwargs)
            if cnts[y, x] > 0:
                mode_min[y, x] = modes[0]
                mode_max[y, x] = modes[-1]
            print((x, y), cnts[y, x], mode_min[y, x], mode_max[y, x])

        # output cnt
        cnts_out = cnts / np.max(cnts)
        imsave(output + "_cnts.png", cnts_out)

        # output distances
        dist_out = np.zeros((h, w, 4))
        dist_out[:, :, 0] = 1.0
        dists_norm = np.abs(mode_min - mode_max)
        dists_norm /= np.max(dists_norm)
        dist_out[:, :, 3] = dists_norm

        imsave(output + "_dist.png", dist_out)

        # output min and max modes
        disp_min = min(np.min(mode_min), np.min(mode_max))
        disp_max = max(np.max(mode_min), np.max(mode_max))
        mode_min_out = (mode_min - disp_min) / (disp_max - disp_min)
        mode_max_out = (mode_max - disp_min) / (disp_max - disp_min)


        imsave(output + "_min.png", mode_min_out)
        imsave(output + "_max.png", mode_max_out)


def sum(gmm, x, y, start, stop, step):
    means = gmm[0, :, y, x]
    vars = gmm[1, :, y, x]
    num_gs = means.shape[0]

    ys = []
    for x in np.arange(start, stop, step):
        y = 0.0
        for i in range(num_gs):
            y += gaussian(x, means[i], vars[i]) / vars[i]

        ys.append(y)

    cnt = 0
    modes = []
    for i in range(1, len(ys) - 1):
        if ys[i - 1] < ys[i] and ys[i + 1] < ys[i]:
            cnt += 1
            modes.append(i * step + start)

    modes.sort()
    return cnt, modes


if __name__ == '__main__':
    sys.exit(main())
