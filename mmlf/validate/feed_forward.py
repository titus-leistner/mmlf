import sys
import os
import time

from ..data import hci4d
from ..model.feed_forward import FeedForward
from ..model import loss

import torch
import click


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--val_loss_margin', default=15,
              help='Margin around each image to omit for the validation loss.')
def main(output_dir, dataset, val_loss_margin):
    valset = hci4d.HCI4D(dataset)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)

    # load hyper parameters
    state = torch.load(os.path.join(output_dir, 'checkpoint.pt'))
    kwargs = state['hyper_parameters']

    # init model and loss functions
    model = FeedForward(**kwargs).cuda()
    mse_fn = loss.MaskedMSELoss()
    bad_pix_fn = loss.MaskedBadPix()

    # load model
    print('Loading model...')
    model.load_state_dict(state['model_state_dict'])

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

            disp, uncert = model(
                h_views, v_views, i_views, d_views)

            mse = mse_fn(disp, gt, mask)
            mse_avg += mse

            bad_pix = bad_pix_fn(disp, gt, mask)
            bad_pix_avg += bad_pix

            # save results
            if uncert is not None:
                uncert = uncert.cpu().numpy()
            disp = disp.cpu().numpy()

            runtime = time.time() - t_start
            valset.save_batch(output_dir, index.numpy(), disp, uncert, runtime)

        mse_avg /= i
        bad_pix_avg /= i

    print(f'MSE: {mse_avg:.8f}, BadPix007: {bad_pix_avg:.8f}')


if __name__ == '__main__':
    sys.exit(main())
