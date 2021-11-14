import sys
import os
import time

from ..data import hci4d
from ..model.feed_forward import FeedForward
from ..model.ensamble import Ensamble
# from ..model.invertible import ZixelWrapper
from ..model import loss
from ..utils.dl import mpi_to_weights

import numpy as np
import torch
import click


def prob_laplace(disp, mean, logvar):
    mean = np.expand_dims(mean, 1)
    var = np.exp(np.expand_dims(logvar, 1))

    prob = np.exp(-(np.abs(mean - disp)) / var) / var / 2.0

    return prob


def cdf_laplace(disp, mean, var):
    le = disp < mean
    ge = np.logical_not(le)

    result_le = np.exp((disp - mean) / var) / 2
    result_ge = (1 - np.exp(-(disp - mean) / var) / 2)

    result_le[ge] = 0.0
    result_ge[le] = 0.0

    result = result_le + result_ge

    return result


def laplace_to_discrete(n_bins, x_min, x_max, mean, logvar):
    step = (x_max - x_min) / n_bins

    disp_space = np.linspace(x_min - step / 2.0, x_max + step / 2.0, n_bins + 1)
    disp_space = np.expand_dims(disp_space, (0, 2, 3))
    mean = np.expand_dims(mean, 1)
    var = np.exp(np.expand_dims(logvar, 1))

    cdf = cdf_laplace(disp_space, mean, var)

    discrete = cdf[:, 1:] - cdf[:, :-1]

    return discrete


def lmm_to_discrete(n_bins, x_min, x_max, means, logvars):
    count = means.shape[0]

    result = np.zeros(means.shape[1:])
    for i in range(means.shape[0]):
        print('Discretize Laplacian ', i)
        laplace_to_discrete(n_bins, x_min, x_max, means[i], logvars[i])

    result /= count

    return result


def mean_to_discrete(n_bins, x_min, x_max, mean):
    # step = (stop - start) / n_steps
    # result = torch.linspace(start, stop, n_steps).view((1, -1, 1, 1)).unsqueeze(2)
    # breakpoint()
    # weights = arr[:, :, 3].unsqueeze(1)
    # arr = arr[:, :, 4].unsqueeze(1)
    # result = (torch.abs(result - arr) < step / 2.0).float() * weights
    # result = result.sum(2)

    step = (x_max - x_min) / n_bins
    disp_space = np.linspace(x_min, x_max, n_bins)
    disp_space = np.expand_dims(disp_space, (0, 2, 3))
    mean = np.expand_dims(mean, 1)

    result = np.abs(disp_space - mean) < step / 2.0

    return result.astype(float)


def likelihood_laplace(mpi, mean, logvar, mask):
    count = np.sum(mask)

    disp = mpi[:, :, 4]
    alpha = mpi[:, :, 3]
    mean = np.expand_dims(mean, 1)
    var = np.exp(np.expand_dims(logvar, 1))

    prob = np.exp(-(np.abs(mean - disp)) / var) / var / 2.0 + 0.00001
    prob /= np.sum(prob, 1, keepdims=True)

    lh = np.sum(alpha * prob, axis=1)

    lh *= mask

    result = np.sum(lh) / count

    print(result)
    return result


def likelihood_lmm(mpi, means, logvars):
    count = means.shape[0]
    mean = means.reshape((-1, means.shape[2], means.shape[3]))
    logvar = logvars.reshape((-1, logvars.shape[2], logvars.shape[3]))

    return likelihood_laplace(mpi, mean, logvar) / float(count)


def likelihood_discrete(weights, posterior):
    weights += 0.00001
    weights /= np.sum(weights, 1, keepdims=True)
    print(np.mean(weights * posterior))
    return np.mean(weights * posterior)


def multimodal_mask(mpi, threshhold=0.1):
    alpha = mpi[:, :, 3]

    mask = (np.sum(alpha > 0.1, 1) > 1).astype(float)

    return mask


def kl_divergence(dist, dist_gt, mask=None):
    epsilon = 0.00001
    dist += epsilon
    dist_gt += epsilon

    dist /= np.sum(dist, 1)
    dist_gt /= np.sum(dist_gt, 1)

    kld = np.sum(dist_gt * np.log(dist_gt / dist), 1)

    if mask is None:
        return np.mean(kld)

    return np.sum(kld * mask) / np.sum(mask)


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
@click.option('--train_shift', default=0.0, type=float,
              help='Static shift to apply to off-center training datasets')
def main(output_dir, dataset, model_invertible, model_discrete,
         val_loss_margin, val_ensamble, val_disp_step, val_disp_min,
         val_disp_max, train_shift):

    # load hyper parameters
    state = torch.load(os.path.join(output_dir, 'checkpoint.pt'))
    kwargs = state['hyper_parameters']
    kwargs.update({'model_discrete': model_discrete, 'val_disp_min': val_disp_min,
                  'val_disp_max': val_disp_max, 'train_shift': train_shift})

    valset = hci4d.HCI4D(dataset, transform=hci4d.Shift(kwargs['train_shift']))
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)

    # if model_invertible:
    #     model = ZixelWrapper(**kwargs).cuda()
    # else:
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

    # print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', n_params)

    # validate
    with torch.no_grad():
        model.eval()

        mse_avg = 0.0
        bad_pix_avg = 0.0
        kld_avg = 0.0
        kld_mm_avg = 0.0
        kld_um_avg = 0.0
        for i, data in enumerate(valloader):
            if i == len(valset.scenes):
                break

            print(f'Processing scene {i}...')
            t_start = time.time()

            h_views, v_views, i_views, d_views, center, gt, mpi, _, index = data
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
            dist_gt = mpi_to_weights(mpi, kwargs['val_disp_min'], kwargs['val_disp_max'], 108).cpu().numpy()
            mpi = mpi.cpu().numpy()
            mean = output['mean'].cpu().numpy()  # + kwargs['train_shift']

            logvar = output.get('logvar', None)
            if logvar is not None:
                logvar = logvar.cpu().numpy()

            # LMM parameters
            means = output.get('means', None)
            logvars = output.get('logvars', None)

            lmm = None
            if means is not None and logvars is not None:
                means = means.cpu().numpy()
                logvars = np.exp(logvars.cpu().numpy())
                lmm = np.stack([means, logvars], 0)

            nll = output.get('scores', None)
            if nll is not None:
                nll = nll.cpu().numpy()

            posterior = output.get('posterior', None)
            if posterior is not None:
                posterior = posterior.cpu().numpy()
            runtime = time.time() - t_start
            valset.save_batch(output_dir, index.numpy(), mean - kwargs['train_shift'],
                              logvar, runtime, lmm, nll, posterior)

            if kwargs['val_ensamble']:
                dist = lmm_to_discrete(108, kwargs['val_disp_min'], kwargs['val_disp_max'], means, logvars)
            elif kwargs['model_discrete']:
                dist = posterior
            elif kwargs['model_uncert']:
                dist = laplace_to_discrete(108, kwargs['val_disp_min'], kwargs['val_disp_max'], mean, logvar)
            else:
                dist = laplace_to_discrete(108, kwargs['val_disp_min'],
                                           kwargs['val_disp_max'], mean, np.zeros_like(mean))
                # dist = mean_to_discrete(108, kwargs['val_disp_min'],
                #                         kwargs['val_disp_max'], mean)

            mask = multimodal_mask(mpi)
            kld = kl_divergence(dist, dist_gt)
            kld_mm = kl_divergence(dist, dist_gt, mask)
            kld_um = kl_divergence(dist, dist_gt, 1.0 - mask)
            print(kld_um, kld_mm, kld)

            kld_avg += kld
            kld_mm_avg += kld_mm
            kld_um_avg += kld_um

        mse_avg /= (i + 1)
        bad_pix_avg /= (i + 1)
        kld_avg /= (i + 1)
        kld_mm_avg /= (i + 1)
        kld_um_avg /= (i + 1)

    print('MSE & BadPix007 & KLD_UM & KLD_MM & KLD & - & TIME \\\\')
    print(f'{mse_avg:.3f} & {bad_pix_avg:.3f} & {kld_um_avg:.3f} & {kld_mm_avg:.3f} & {kld_avg:.3f} & - & {runtime:.3f} \\\\')


if __name__ == '__main__':
    sys.exit(main())
