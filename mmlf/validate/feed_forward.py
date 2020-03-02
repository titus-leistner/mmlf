import sys
import os
import time

from ..data import hci4d
from ..model.feed_forward import FeedForward
from ..model.ensamble import Ensamble
from ..model.invertible import ZixelWrapper
from ..model import loss

import numpy as np
import torch
import click


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--model_invertible', is_flag=True,
              help='Use invertible architecture?')
@click.option('--model_discrete', is_flag=True,
              help='Discretize disparity output?')
@click.option('--val_loss_margin', default=15,
              help='Margin around each image to omit for the validation loss')
@click.option('--val_ensamble', is_flag=True,
              help='Use a network ensamble?')
@click.option('--val_disp_min', default=-3.5,
              help='Minimum disparity of dataset')
@click.option('--val_disp_max', default=3.5,
              help='Maximum disparity of dataset')
@click.option('--val_disp_step', default=0.1,
              help='Disparity increment for ensamble')
def main(output_dir, dataset, model_invertible, model_discrete,
         val_loss_margin, val_ensamble, val_disp_step, val_disp_min,
         val_disp_max):
    valset = hci4d.HCI4D(dataset)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)

    # load hyper parameters
    state = torch.load(os.path.join(output_dir, 'checkpoint.pt'))
    kwargs = state['hyper_parameters']
    kwargs.update({'model_discrete': model_discrete, 'val_disp_min': val_disp_min, 'val_disp_max': val_disp_max})

    if model_invertible:
        model = ZixelWrapper(**kwargs).cuda()
    else:
        model = FeedForward(**kwargs).cuda()

    mse_fn = loss.MaskedMSELoss()
    bad_pix_fn = loss.MaskedBadPix()

    # load model
    print('Loading model...')
    model.load_state_dict(state['model_state_dict'])

    # load mu
    if model_invertible:
        model.invertible.mu.data.copy_(state['mu'].data)

    if val_ensamble:
        # initialize ensamble
        model = Ensamble(model, val_disp_min, val_disp_max, val_disp_step)

    # validate
    with torch.no_grad():
        model.eval()

        mse_avg = 0.0
        bad_pix_avg = 0.0
        for i, data in enumerate(valloader):
            if i == len(valset.scenes):
                break

            print(f'Processing scene {i}...')
            t_start = time.time()

            h_views, v_views, i_views, d_views, center, gt, _, index = data
            h_views = h_views.cuda()
            v_views = v_views.cuda()
            i_views = i_views.cuda()
            d_views = d_views.cuda()
            gt = gt.cuda()
            mask = loss.create_mask_margin(
                gt.shape, val_loss_margin).cuda()

            output = model(h_views, v_views, i_views, d_views)

            mse = mse_fn(output, gt, mask)
            mse_avg += mse

            bad_pix = bad_pix_fn(output, gt, mask)
            bad_pix_avg += bad_pix

            # save results
            mean = output['mean'].cpu().numpy()

            logvar = output.get('logvar', None)
            if logvar is not None:
                logvar = logvar.cpu().numpy()

            # GMM parameters
            means = output.get('means', None)
            logvars = output.get('logvars', None)

            gmm = None
            if means is not None and logvars is not None:
                means = means.cpu().numpy()
                logvars = np.exp(logvars.cpu().numpy())
                gmm = np.stack([means, logvars], 0)

            nll = output.get('nll', None)
            if nll is not None:
                nll = nll.cpu().numpy()

            posterior = output.get('posterior', None)
            if posterior is not None:
                posterior = posterior.cpu().numpy()
            runtime = time.time() - t_start
            valset.save_batch(output_dir, index.numpy(), mean,
                              logvar, runtime, gmm, nll, posterior)

        mse_avg /= i
        bad_pix_avg /= i

    print(f'MSE: {mse_avg:.8f}, BadPix007: {bad_pix_avg:.8f}')


if __name__ == '__main__':
    sys.exit(main())
