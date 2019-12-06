import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed Forward model based on:

    'EPINET: A Fully-Convolutional Neural Network
    Using Epipolar Geometry for Depth from Light Field Images'

    by Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon, Seon Joo Kim
    """

    def __init__(self, model_ksize, model_in_blocks, model_out_blocks,
                 model_chs, model_views, model_cross, **kwargs):
        """
        :param model_ksize: kernel size
        :type model_ksize: int

        :param model_in_blocks: number of blocks for input network
        :type model_in_blocks: int

        :param model_out_blocks: number of blocks for output network
        :type model_out_blocks: int

        :param model_chs: number of channels for input network
        :type model_chs: int

        :param model_views: number of light field views
        :type model_views: int

        :param cross: only cross setup?
        :type cross: bool
        """
        super(FeedForward, self).__init__()

        self.ksize = model_ksize
        self.chs = model_chs
        self.views = model_views
        self.cross = model_cross
        print(self.cross)

        if model_ksize % 2 == 1:
            self.padding1 = model_ksize // 2
            self.padding2 = model_ksize // 2

        else:
            self.padding1 = model_ksize // 2
            self.padding2 = model_ksize // 2 - 1

        # construct network
        self.in_net_hv = self.init_in_net(model_in_blocks)

        if not model_cross:
            self.in_net_id = self.init_in_net(model_in_blocks)

        self.out_net = self.init_out_net(model_out_blocks)

    def block(self, ch_in, ch_out=None, out_bn_relu=True):
        """
        Create convolutional block

        :param ch_in: number of input channels
        :type ch_in: int

        :param ch_out: number of output channels or None if ch_in == ch_out
        :type ch_out: int

        :param out_bn_relu: add batch norm and second activation function?
        :type out_bn_relu: bool

        :returns: layers of one convolutional block
        """
        if ch_out is None:
            ch_out = ch_in

        layers = [
            nn.Conv2d(ch_in, ch_out, self.ksize, padding=self.padding1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, self.ksize, padding=self.padding2),
        ]

        if out_bn_relu:
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def init_in_net(self, n_blocks):
        """
        Create network for one input stream

        :param n_blocks: number of blocks
        :type n_blocks: int

        :returns: layers of input net
        """
        assert n_blocks >= 1

        # input block
        blocks = [self.block(self.views * 3, self.chs)]

        # remaining blocks
        for _ in range(n_blocks - 1):
            blocks.append(self.block(self.chs))

        return nn.Sequential(*blocks)

    def init_out_net(self, n_blocks):
        """
        Create second part of network (after concatenation)

        :param n_blocks: number of blocks
        :type n_blocks: int

        :returns: layers of input net
        """
        assert n_blocks >= 1

        chs = 4 * self.chs
        if self.cross:
            chs = 2 * self.chs

        blocks = []
        # add first blocks
        for _ in range(n_blocks - 1):
            blocks.append(self.block(chs))

        blocks.append(self.block(chs, 1, False))

        return nn.Sequential(*blocks)

    def forward(self, h_views, v_views, i_views=None, d_views=None):
        """
        Forward network in an end-to-end fashion

        :param h_views: horizontal view stack
        :type h_views: torch.Tensor of shape (b, n, 3, h, w)

        :param v_views: vertical view stack
        :type v_views: torch.Tensor of shape (b, n, 3, h, w)

        :param i_views: increasing diagonal view stack
        :type i_views: torch.Tensor of shape (b, n, 3, h, w)

        :param d_views: decreasing diagonal view stack
        :type d_views: torch.Tensor of shape (b, n, 3, h, w)

        :returns: disparity
        """
        # reshape input to combine view and color dimension
        # TODO: do this in training script
        b, n, c, h, w = h_views.shape
        h_views = h_views.view(b, n * c, h, w)
        v_views = v_views.view(b, n * c, h, w)

        if not self.cross:
            i_views = i_views.view(b, n * c, h, w)
            d_views = d_views.view(b, n * c, h, w)

        # extract features
        # swap dimensions of horizontal stack
        h_views = h_views.permute(0, 1, 3, 2)

        h_features = self.in_net_hv(h_views)

        # again swap image dimensions to concatenate with vertical EPI
        h_features = h_features.permute(0, 1, 3, 2)

        v_features = self.in_net_hv(v_views)

        i_features, d_features = None, None
        if not self.cross:
            # same for diagonals
            i_views = i_views.permute(0, 1, 3, 2)

            # additionally flip x-axis
            i_views = torch.flip(i_views, (-1,))

            i_features = self.in_net_id(i_views)

            i_features = torch.flip(i_features, (-1,))
            i_features = i_features.permute(0, 1, 3, 2)

            d_features = self.in_net_id(d_views)

        # concatenate features and compute disparity
        features = None

        if self.cross:
            features = torch.cat([h_features, v_features], 1)
        else:
            features = torch.cat(
                [h_features, v_features, i_features, d_features], 1)

        disp = self.out_net(features).squeeze(1)

        return disp
