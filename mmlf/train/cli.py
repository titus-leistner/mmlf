import sys
import os

from ..data import hci4d
from ..model.feed_forward import FeedForward
from ..model.invertible import ZixelWrapper
from ..model.ensamble import Ensamble
from ..utils.dl import ModelSaver, reg_to_class, class_to_reg
from ..model import loss

import torch
from torchvision import transforms
import click


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--model_ksize', default=2, help='Kernel size for convolutions, e.g. 3 for 3x3 kernels')
@click.option('--model_in_blocks', default=3, help='Number of blocks for input network')
@click.option('--model_out_blocks', default=8, help='Number of blocks for output network')
@click.option('--model_chs', default=70, help='Number of channels for input network')
@click.option('--model_views', default=9, help='Number of viewpoints of the input light field, e.g. 9 for 9+8 views')
@click.option('--model_cross', is_flag=True, help='Only use cross input?')
@click.option('--model_uncert', is_flag=True, help='Use uncertainty model?')
@click.option('--model_discrete', is_flag=True, help='Discretize disparity output?')
@click.option('--model_unet', is_flag=True, help='Use a U-Net after the multistream network?')
@click.option('--model_invertible', is_flag=True, help='Use invertible architecture?')
@click.option('--model_clamp', default=0.7, help='Output clamp for coupling block?')
@click.option('--model_act_norm', default=0.7, help='Activation normalization for coupling block?')
@click.option('--model_act_norm_type', default='SOFTPLUS', help='Type of activation normalization for coupling block?')
@click.option('--model_soft_permutation', is_flag=True, help='Use soft permuation for coupling block?')
@click.option('--train_trainset', default='../lf-dataset/additional', help='Location of training dataset')
@click.option('--train_valset', default='../lf-dataset/training', help='Location of validation dataset')
@click.option('--train_num_workers', default=4, help='Number of workors for data loader')
@click.option('--train_lr', default=1e-5, help='Learning rate')
@click.option('--train_bs', default=1, help='Batch size')
@click.option('--train_ps', default=32, help='Size of training patches')
@click.option('--train_beta', default=1.0, help='Weighting between NLL and Cat CE')
@click.option('--train_mae_threshold', default=0.02, help='If the MAE of one patch is under this threshold, no loss is applied')
@click.option('--train_max_downscale', default=4, help='Maximum factor of down scaling for data augmentation')
@click.option('--train_resume', is_flag=True, help='Resume training from old checkpoint?')
@click.option('--train_loss_padding', default=None, type=float, help='Margin around ground truth to apply loss')
@click.option('--val_interval', default=100, help='Validation interval')
@click.option('--val_loss_margin', default=15, help='Margin around each image to omit for the validation loss.')
@click.option('--val_ensamble', is_flag=True, help='Use a network ensamble?')
@click.option('--val_disp_min', default=-3.5, help='Minimum disparity of dataset')
@click.option('--val_disp_max', default=3.5, help='Maximum disparity of dataset')
@click.option('--val_disp_step', default=0.1, help='Disparity increment for ensamble')
def main(output_dir, **kwargs):
    # compute radius
    kwargs['model_radius'] = (kwargs['model_in_blocks'] +
                              kwargs['model_out_blocks']) * \
        ((kwargs['model_ksize'] + 1) // 2)

    # ensamble implies uncertainty
    if kwargs['val_ensamble']:
        kwargs['model_uncert'] = True

    # initialize transforms
    transform = transforms.Compose([
        hci4d.RandomDownSampling(kwargs['train_max_downscale']),
        hci4d.RandomShift(2.0),
        hci4d.RandomCrop(kwargs['train_ps'] + 2 * 4 * 2),
        hci4d.CenterCrop(kwargs['train_ps']),
        hci4d.RandomRotate(),
        hci4d.RedistColor(),
        hci4d.Brightness(),
        hci4d.Contrast()
    ])

    # load datasets
    trainset = hci4d.HCI4D(
        kwargs['train_trainset'], transform=transform, cache=True, length=4096)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=kwargs['train_bs'],
                                              shuffle=True,
                                              num_workers=kwargs['train_num_workers'])

    valset = hci4d.HCI4D(kwargs['train_valset'], cache=True)
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)

    # init model, optimizer and train iteration
    if kwargs['model_invertible']:
        model = ZixelWrapper(**kwargs).cuda()
    else:
        model = FeedForward(**kwargs).cuda()

    if kwargs['model_invertible']:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=kwargs['train_lr'])
    else:
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=kwargs['train_lr'])

    loss_fn = loss.MaskedL1Loss()
    loss_uncert_fn = loss.UncertaintyL1Loss()
    loss_discrete_fn = loss.MaskedCrossEntropy()
    loss_invertible_fn = loss.InformationBottleneckLoss(kwargs['train_beta'])
    mse_fn = loss.MaskedMSELoss()
    bad_pix_fn = loss.MaskedBadPix()

    i = 0

    # load model if necessary
    if kwargs['train_resume']:
        print('Resume training...')
        state = torch.load(os.path.join(output_dir, 'checkpoint.pt'))

        # remove temporary vars
        for k in list(state['model_state_dict']):
            if 'tmp' in k:
                print(f'Removing "{k}" from state dict...')
                del state['model_state_dict'][k]

        model.load_state_dict(state['model_state_dict'])
        if kwargs['model_invertible']:
            model.invertible.mu.data.copy_(state['mu'].data)

        optimizer.load_state_dict(state['optimizer_state_dict'])

        # manually set new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = kwargs['train_lr']

        i = state['iteration']

    model = torch.nn.DataParallel(model)

    # create ensamble model
    if kwargs['val_ensamble']:
        val_model = Ensamble(model, **kwargs)
    else:
        val_model = model

    # logging
    mode = 'a' if kwargs['train_resume'] else 'w'
    log = open(os.path.join(output_dir, 'log.csv'), mode)

    # output header
    header = f'{"iter":>7}, loss_train,   loss_val,        mse, badpix_007'
    print(header)
    if not kwargs['train_resume']:
        print(header, file=log)

    # model saver
    # TODO: Enable only best
    model_saver = ModelSaver(only_best=False)

    while True:
        for data in trainloader:
            # train
            h_views, v_views, i_views, d_views, center, gt, mask, index = data
            dims = 4
            if kwargs['model_cross']:
                dims = 2
            dims *= kwargs['model_views'] * 3
            gt_classes = reg_to_class(
                gt, kwargs['val_disp_min'], kwargs['val_disp_max'], dims)

            h_views = h_views.cuda()
            v_views = v_views.cuda()
            i_views = i_views.cuda()
            d_views = d_views.cuda()
            gt = gt.cuda()
            gt_classes = gt_classes.cuda()

            # no loss if no texture
            mask = mask.int() * loss.create_mask_texture(
                center, kwargs['model_radius'] * 2 + 1,
                kwargs['train_mae_threshold']).int()

            mask = mask.cuda()

            if kwargs['train_loss_padding'] is not None:
                mask = mask.int() * (torch.abs(gt) <
                                     kwargs['train_loss_padding']).int()

            model.train()
            optimizer.zero_grad()

            output = model(h_views, v_views, i_views, d_views)

            if kwargs['model_uncert']:
                loss_train = loss_uncert_fn(output, gt, mask)
            elif kwargs['model_discrete']:
                loss_train = loss_discrete_fn(output, gt_classes, mask)
            elif kwargs['model_invertible']:
                loss_train = loss_invertible_fn(output, gt_classes, None)
            else:
                loss_train = loss_fn(output, gt, mask)

            loss_train.backward()
            optimizer.step()

            if i % kwargs['val_interval'] == 0:
                # validate
                with torch.no_grad():
                    model.eval()

                    loss_val_avg = 0.0
                    mse_avg = 0.0
                    bad_pix_avg = 0.0
                    for j, data in enumerate(valloader):
                        h_views, v_views, i_views, d_views, center, gt, _, index = data

                        h_views = h_views.cuda()
                        v_views = v_views.cuda()
                        i_views = i_views.cuda()
                        d_views = d_views.cuda()
                        gt = gt.cuda()
                        mask = loss.create_mask_margin(
                            gt.shape, kwargs['val_loss_margin']).cuda()

                        output = val_model(h_views, v_views, i_views, d_views)

                        if kwargs['model_uncert']:
                            loss_val = loss_uncert_fn(output, gt, mask)
                        else:
                            loss_val = loss_fn(output, gt, mask)

                        loss_val_avg += loss_val.item()

                        mse = mse_fn(output, gt, mask)
                        mse_avg += mse

                        bad_pix = bad_pix_fn(output, gt, mask)
                        bad_pix_avg += bad_pix

                        # save results
                        logvar = output.get('logvar', None)
                        if logvar is not None:
                            logvar = logvar.cpu().numpy()
                        mean = output['mean'].cpu().numpy()

                        valset.save_batch(output_dir, index.numpy(
                        ), mean, logvar)

                    j += 1
                    loss_val_avg /= j
                    mse_avg /= j
                    bad_pix_avg /= j

                    save_additional = {}
                    if kwargs['model_invertible']:
                        save_additional['mu'] = model.module.invertible.mu

                    # save model
                    model_saver(os.path.join(output_dir, 'checkpoint.pt'),
                                model, optimizer, kwargs, None, i,
                                loss_val_avg, **save_additional)

            output = f'{i:>7}, {loss_train:.8f}, {loss_val_avg:.8f}, {mse_avg:.8f}, {bad_pix_avg:.8f}'
            print(output)
            print(output, file=log, flush=True)

            i += 1


if __name__ == '__main__':
    sys.exit(main())