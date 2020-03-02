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
    posterior = np.load(input)
    num_samples = posterior.shape[0]

    with open(output, 'w') as f:
        f.write(f'y, p\n')
        for i in range(num_samples):
            disp = float(i) / float(num_samples - 1) * (stop - start) + start
            prob = posterior[i, y, x]
            f.write(f'{disp}, {prob}\n')


if __name__ == '__main__':
    sys.exit(main())
