import sys

import numpy as np

import click


def gaussian(x, mean, var):
    y = 1.0 / np.sqrt(2.0 * np.pi * var) * \
        np.exp(-(x - mean) ** 2.0 / (2.0 * var))

    return y


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
@click.option('--step', default=0.01, help='step width')
@click.option('--sum_only', is_flag=True, help='sum over gaussians?')
def main(sum_only, **kwargs):
    if sum_only:
        sum(**kwargs)
    else:
        separate(**kwargs)


def sum(input, output, x, y, start, stop, step):
    gmm = np.load(input)
    means = gmm[0, :, y, x]
    vars = gmm[1, :, y, x]
    num_gs = means.shape[0]

    with open(output, 'w') as f:
        f.write(f'x, p\n')

        for x in np.arange(start, stop, step):
            print(x)
            y = 0.0
            for i in range(num_gs):
                y += gaussian(x, means[i], vars[i])
            y /= float(num_gs)

            f.write(f'{x}, {y}\n')


def separate(input, output, x, y, start, stop, step):
    gmm = np.load(input)
    means = gmm[0, :, y, x]
    vars = gmm[1, :, y, x]
    num_gs = means.shape[0]

    with open(output, 'w') as f:
        f.write(f'x, ')
        for i in range(num_gs):
            f.write(f'G_{i}')
            if i < num_gs - 1:
                f.write(', ')
        f.write('\n')

        for x in np.arange(start, stop, step):
            print(x)
            f.write(f'{x}, ')
            for i in range(num_gs):
                print(means[i])
                y = gaussian(x, means[i], vars[i])
                f.write(f'{y}')
                if i < num_gs - 1:
                    f.write(', ')
            f.write('\n')


if __name__ == '__main__':
    sys.exit(main())
