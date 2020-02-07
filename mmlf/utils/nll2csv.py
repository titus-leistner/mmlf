import sys

import numpy as np

import click


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('output', type=click.Path())
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.option('--start', default=-3.5, help='lower limit')
@click.option('--stop', default=3.5, help='upper limit')
def main(input, output, x, y, start, stop):
    nll = np.load(input)[:, y, x]
    lh = np.exp(-nll)
    # lh /= np.sum(lh)

    with open(output, 'w') as f:
        f.write(f'disparity, likelihood\n')
        classes = nll.shape[0]
        for i in range(classes):
            x = float(i) / float(classes) * (stop - start) + start
            f.write(f'{x}, {lh[i]}\n')


if __name__ == '__main__':
    sys.exit(main())
