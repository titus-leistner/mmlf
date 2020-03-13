import sys

import numpy as np

import click


def gaussian(x, mean, var):
    y = 1.0 / np.sqrt(2.0 * np.pi * var) * \
        np.exp(-(x - mean) ** 2.0 / (2.0 * var))

    return y


@click.command()
@click.argument('output', type=click.Path())
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
@click.option('--step', default=0.005, help='step width')
def main(output, start, stop, step):
    means = np.asarray([-1.0, 1.0])
    vars = np.asarray([0.3, 0.4])
    num_gs = 2

    with open(output, 'w') as f:
        f.write(f'x, p\n')
        norm = 0.0
        values = []
        for x in np.arange(start, stop, step):
            y = 0.0
            for i in range(num_gs):
                y += gaussian(x, means[i], vars[i]) / vars[i]

            values.append((x, y))
            norm = max(y, norm)

        for value in values:
            x, y = value
            y /= norm
            f.write(f'{x}, {y}\n')


if __name__ == '__main__':
    sys.exit(main())
