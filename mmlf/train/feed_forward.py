import sys
import os

from ..data import hci4d
from ..model.feed_forward import FeedForward
from ..utils.dl import ModelSaver

import torch
import torch.nn as nn
from torchvision import transforms
import click


@click.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--model_ksize', default=2, help='Kernel size for convolutions, e.g. 3 for 3x3 kernels')
@click.option('--model_in_blocks', default=3, help='Number of blocks for input network')
@click.option('--model_out_blocks', default=8, help='Number of blocks for output network')
@click.option('--model_chs', default=70, help='Number of channels for input network')
@click.option('--model_views', default=9, help='Number of viewpoints of the input light field, e.g. 9 for 9+8 views')
@click.option('--train_trainset', default='../lf-dataset/additional', help='Location of training dataset')
@click.option('--train_valset', default='../lf-dataset/training', help='Location of validation dataset')
@click.option('--train_lr', default=1e-5, help='Learning rate')
@click.option('--train_bs', default=1, help='Batch size')
@click.option('--train_ps', default=32, help='Size of training patches')
@click.option('--train_resume', is_flag=True, help='Resume training from old checkpoint?')
@click.option('--val_interval', default=1000, help='Validation interval')
def main(output_dir, **kwargs):
    # initialize transforms
    transform = transforms.Compose([
        hci4d.RandomDownSampling(2),
        hci4d.RandomCrop(kwargs['train_ps']),
        hci4d.RandomRotate(),
        hci4d.RedistColor(),
        hci4d.Brightness(),
        hci4d.Contrast()
    ])

    # load datasets
    trainset = hci4d.HCI4D(
        kwargs['train_trainset'], transform=transform, cache=True)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=kwargs['train_bs'],
                                              shuffle=True, num_workers=4)

    valset = hci4d.HCI4D(kwargs['train_valset'])
    valloader = torch.utils.data.DataLoader(valset,
                                            batch_size=1,
                                            shuffle=False, num_workers=1)

    # init model, optimizer and train iteration
    model = FeedForward(**kwargs).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['train_lr'])
    loss_fn = nn.L1Loss()
    i = 0

    # load model if necessary
    if kwargs['train_resume']:
        print('Resume training...')
        state = torch.load(os.path.join(output_dir, 'checkpoint.pt'))
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        i = state['iteration']

    # logging
    mode = 'a' if kwargs['train_resume'] else 'w'
    log = open(os.path.join(output_dir, 'log.csv'), mode)

    if not kwargs['train_resume']:
        # output header
        header = f'{"iter":>7}, loss_train,   loss_val'
        print(header)
        print(header, file=log)

    # model saver
    model_saver = ModelSaver(only_best=True)
    while True:
        for data in trainloader:
            # train
            h_views, v_views, center, gt, index = data
            h_views = h_views.cuda()
            v_views = v_views.cuda()
            gt = gt.cuda()

            model.train()
            optimizer.zero_grad()

            disp = model(h_views, v_views)

            loss_train = loss_fn(disp, gt)

            loss_train.backward()
            optimizer.step()

            if i % kwargs['val_interval'] == 0:
                # validate
                with torch.no_grad():
                    model.eval()

                    loss_val_avg = 0.0
                    for j, data in enumerate(valloader):
                        h_views, v_views, center, gt, index = data
                        h_views = h_views.cuda()
                        v_views = v_views.cuda()
                        gt = gt.cuda()

                        disp = model(h_views, v_views)
                        loss_val = loss_fn(disp, gt)
                        loss_val_avg += loss_val.item()

                        # save results
                        valset.save_batch(
                            output_dir, index.numpy(), disp.cpu().numpy())
                        if j == 4:
                            break

                    loss_val_avg /= j

                    # save model
                    model_saver(os.path.join(output_dir, 'checkpoint.pt'),
                                model, optimizer, kwargs, None, i,
                                loss_val_avg)

            output = f'{i:>7}, {loss_train:.8f}, {loss_val_avg:.8f}'
            print(output)
            print(output, file=log, flush=True)

            i += 1


if __name__ == '__main__':
    sys.exit(main())
