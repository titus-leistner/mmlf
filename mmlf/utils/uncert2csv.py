import sys
import click
import numpy as np
from . import pfm


def gaussian(x, mean, var):
    y = 1.0 / np.sqrt(2.0 * np.pi * var) * \
        np.exp(-(x - mean) ** 2.0 / (2.0 * var))

    return y


@click.command()
@click.argument('mean', type=click.Path(exists=True))
@click.argument('variance', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
@click.option('--step', default=0.005, help='step width')
def main(mean, variance, output, x, y, start, stop, step):
    mean = pfm.load(mean)
    mean = np.flip(mean, 0).copy()
    mean = mean[y][x]
    var = pfm.load(variance)
    var = np.flip(var, 0).copy()
    var = np.exp(var[y][x])

    with open(output, 'w') as f:
        f.write(f'x, p\n')
        norm = 0.0
        values = []
        for x in np.arange(start, stop, step):
            y = gaussian(x, mean, var)

            values.append((x, y))
            norm = max(y, norm)

        for value in values:
            x, y = value
            y /= norm
            f.write(f'{x}, {y}\n')


if __name__ == '__main__':
    sys.exit(main())
