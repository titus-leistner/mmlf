import torch
import torch.nn as nn
import torch.nn.functional as F

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from ..utils.dl import class_to_reg
from .coupling_blocks import AIO_HighPerfCouplingBlock


class Invertible(nn.Module):
    """
    INN Architecture for light field depth estimation
    """

    def __init__(self, model_ksize, model_in_blocks, model_out_blocks,
                 model_views, model_cross, train_lr, train_ps, train_bs,
                 model_clamp, model_act_norm, model_act_norm_type,
                 model_soft_permutation, **kwargs):
        """
        :param model_ksize: kernel size
        :type model_ksize: int

        :param model_in_blocks: number of blocks for input network
        :type model_in_blocks: int

        :param model_out_blocks: number of blocks for output network
        :type model_out_blocks: int

        :param model_views: number of light field views
        :type model_views: int

        :param model_cross: only cross setup?
        :type model_cross: bool

        :param train_lr: the learning rate
        :type train_lr: float

        :param train_psize: patch size
        :type train_psize: int

        :param train_bsize: batch size
        :type train_bsize: int
        """
        super(Invertible, self).__init__()

        self.ksize = model_ksize
        self.chs1 = (model_views * 3 + 1) // 2
        self.chs2 = (model_views * 3) - self.chs1
        self.views = model_views
        self.cross = model_cross
        self.psize = train_ps
        self.lr = train_lr
        self.clamp = model_clamp
        self.act_norm = model_act_norm
        self.act_norm_type = model_act_norm_type
        self.soft_permutation = model_soft_permutation

        if model_ksize % 2 == 1:
            self.padding1 = model_ksize // 2
            self.padding2 = model_ksize // 2

        else:
            self.padding1 = model_ksize // 2
            self.padding2 = model_ksize // 2 - 1

        # construct network
        self.in_net_h, self.in_net_v = self.init_in_net_shared_weights(
            model_in_blocks, 'h', 'v')

        if not model_cross:
            self.in_net_i, self.in_net_d = self.init_in_net_shared_weights(
                model_in_blocks, 'i', 'd')

        def h_to_v(in_chs, out_chs):
            return TransformHtoV()

        def i_to_d(in_chs, out_chs):
            return TransformItoD()

        self.in_net_h.append(
            Ff.Node(self.in_net_h[-1], TransformHtoV, {}, name=f'h_to_v'))

        self.in_net_i.append(
            Ff.Node(self.in_net_i[-1], TransformItoD, {}, name=f'i_to_d'))

        merge = Ff.Node([self.in_net_h[-1].out0, self.in_net_v[-1].out0],
                        Fm.ConcatChannel, {}, name='merge_hv')
        self.merge_net = [merge]

        if not model_cross:
            diag = Ff.Node([self.in_net_i[-1].out0, self.in_net_d[-1].out0],
                           Fm.ConcatChannel, {}, name='merge_id')
            self.merge_net.append(diag)
            merge = Ff.Node([merge.out0, diag.out0],
                            Fm.ConcatChannel, {}, name='merge_hvid')
            self.merge_net.append(merge)

        self.out_net = self.init_out_net(model_out_blocks, merge)

        self.model = Ff.ReversibleGraphNet(
            self.in_net_h + self.in_net_v + self.in_net_i + self.in_net_d +
            self.merge_net + self.out_net)

        # cluster centers
        dims = 4
        if model_cross:
            dims = 2

        dims *= 3 * model_views
        self.mu = nn.Parameter(torch.randn(1, dims, dims))

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

        # init layers
        nn.init.kaiming_normal_(layers[0].weight)
        nn.init.kaiming_normal_(layers[2].weight)

        layers[0].weight.data *= 0.035
        layers[2].weight.data *= 0.035

        if out_bn_relu:
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU())

            # init BatchNorm
            layers[3].weight.data.fill_(1.0)
            layers[3].bias.data.zero_()

        return nn.Sequential(*layers)

    def init_in_net(self, n_blocks):
        """
        Create network blocks for one input stream

        :param n_blocks: number of blocks
        :type n_blocks: int

        :returns: layers of input net
        """
        assert n_blocks >= 1

        # remaining blocks
        blocks1 = []
        blocks2 = []
        for _ in range(n_blocks):
            blocks1.append(self.block(self.chs1))
            blocks2.append(self.block(self.chs2))

        return blocks1, blocks2

    def init_in_net_shared_weights(self, n_blocks, name1='', name2=''):
        """
        Create INN input net with weight sharing

        :param n_blocks: number of blocks
        :type n_blocks: int

        :param name1: name for INN node names
        :type name1: str

        :param name2: name for INN node names
        :type name2: str

        :returns: two lists of blocks (with weight sharing)
        """
        assert n_blocks >= 1

        # first initialize subnets
        subnets1, subnets2 = self.init_in_net(n_blocks)

        # now init the INN sequences with weight sharing
        blocks1 = [Ff.InputNode(
            self.views * 3, self.psize, self.psize, name='input x_0')]
        blocks2 = [Ff.InputNode(
            self.views * 3, self.psize, self.psize, name='input x_1')]

        # remaining blocks
        for i in range(n_blocks):
            def subnet_constructor(in_chs, out_chs):
                if in_chs == self.chs1:
                    return subnets1[i]
                elif in_chs == self.chs2:
                    return subnets2[i]
                else:
                    raise ValueError(
                        'Number of input channels in subnet constructor does'
                        'not match precomputed channels.')

            args = {
                'subnet_constructor': self.block,
                'clamp': self.clamp,
                'act_norm': self.act_norm,
                'act_norm_type': self.act_norm_type,
                'permute_soft': self.soft_permutation
            }

            blocks1.append(Ff.Node(blocks1[i-1], AIO_HighPerfCouplingBlock,
                                   args,
                                   name=f'in_coupling_{name1}_{i}'))
            blocks2.append(Ff.Node(blocks2[i-1], AIO_HighPerfCouplingBlock,
                                   args,
                                   name=f'in_coupling_{name2}_{i}'))

        return blocks1, blocks2

    def init_out_net(self, n_blocks, inp):
        """
        Create second part of network (after concatenation)

        :param n_blocks: number of blocks
        :type n_blocks: int

        :param inp: list of input nodes to this network
        :type inp: list

        :returns: list of blocks
        """
        assert n_blocks >= 1

        blocks = [Ff.Node(inp.out0, AIO_HighPerfCouplingBlock, {
                          'subnet_constructor': self.block},
                          name=f'out_coupling_0')]

        # add first blocks
        for i in range(n_blocks - 1):
            blocks.append(Ff.Node(blocks[i - 1], AIO_HighPerfCouplingBlock, {
                          'subnet_constructor': self.block},
                name=f'out_coupling_{i+1}'))

        blocks.append(Ff.OutputNode(blocks[-1], name='output'))
        return blocks

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

        :returns: per-pixel latent space (zixels)
        """
        # reshape input to combine view and color dimension
        # TODO: do this in training script
        b, n, c, h, w = h_views.shape
        h_views = h_views.view(b, n * c, h, w)
        v_views = v_views.view(b, n * c, h, w)
        h_views = h_views.permute(0, 1, 3, 2)

        views = [h_views, v_views]

        if not self.cross:
            i_views = i_views.view(b, n * c, h, w)
            d_views = d_views.view(b, n * c, h, w)
            i_views = i_views.permute(0, 1, 3, 2)
            i_views = torch.flip(i_views, (-1,))

            views.append(i_views)
            views.append(d_views)

        zixels = self.model(views)
        jac = self.model.log_jacobian(run_forward=False)

        return {'zixels': zixels, 'jac': jac, 'mu': self.mu}


class ZixelWrapper(nn.Module):
    """
    Converts the zixel space to disparity and uncertainty
    """

    def __init__(self, val_disp_min, val_disp_max, **kwargs):
        super(ZixelWrapper, self).__init__()

        self.invertible = Invertible(**kwargs)
        self.disp_min = val_disp_min
        self.disp_max = val_disp_max
        self.steps = 4
        if kwargs['model_cross']:
            self.steps = 2
        self.steps *= kwargs['model_views'] * 3

    def cluster_distances(zixels, mu):
        dims = mu.shape[-1]
        mu = mu.view(1, dims, dims, 1, 1)

        mi_mj = torch.sum(mu ** 2, dim=2)
        zi_zj = torch.sum(zixels ** 2, dim=1, keepdim=True)
        zi_mj = F.conv2d(zixels, mu[0])

        # print('mi_mj:', torch.max(mi_mj).item(), 'zi_zj:',
        #       torch.max(zi_zj).item(), 'zi_mj:',  torch.max(zi_mj).item(),
        #       'mu:', torch.max(mu).item(), 'z:', torch.max(zixels).item())

        # return torch.sqrt(-2.0 * zi_mj + zi_zj + mi_mj)
        return -2.0 * zi_mj + zi_zj + mi_mj

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

        :returns: disparity, uncertainty
        """
        output = self.invertible(h_views, v_views, i_views, d_views)

        output['dists'] = ZixelWrapper.cluster_distances(
            output['zixels'], output['mu'])

        one_hot = (torch.min(output['dists'], 1, keepdim=True)[
                   0] == output['dists']).float()
        output['one_hot'] = one_hot
        output['mean'] = class_to_reg(
            one_hot, self.disp_min, self.disp_max, self.steps)

        return output


class TransformHtoV(nn.Module):
    def __init__(self, *args):
        super(TransformHtoV, self).__init__()

    def forward(self, x, rev=False):
        x = x[0].permute(0, 1, 3, 2)

        return [x]

    def jacobian(self, x, rev=False):
        return 0.0

    def output_dims(self, dim):
        return dim


class TransformItoD(nn.Module):
    def __init__(self, *args):
        super(TransformItoD, self).__init__()

    def forward(self, x, rev=False):
        x = x[0]
        if not rev:
            x = torch.flip(x, (-1,))

        x = x.permute(0, 1, 3, 2)

        if rev:
            x = torch.flip(x, (-1,))

        return [x]

    def jacobian(self, x, rev=False):
        return 0.0

    def output_dims(self, dim):
        return dim
