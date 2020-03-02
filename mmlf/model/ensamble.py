import numpy as np
import torch
import torch.nn as nn

from ..data.hci4d import Shift
from .feed_forward import laplacian


class Ensamble(nn.Module):
    """
    Use an ensamble with weight sharing to infer a per-pixel GMM
    """

    def __init__(self, model, val_disp_min, val_disp_max, val_disp_step,
                 **kwarg):
        """
        :param model: feed forwar network model
        :type model: mmlf.model.feed_forward.FeedForward

        :param val_disp_min: minimum disparity of dataset
        :type val_disp_min: float

        :param val_disp_max: maximum disparity of dataset
        :type val_disp_max: float

        :param val_disp_step: disparity step for ensamble
        :type val_disp_step: float
        """
        super(Ensamble, self).__init__()

        self.disp_min = val_disp_min
        self.disp_max = val_disp_max
        assert self.disp_min < self.disp_max

        self.disp_step = val_disp_step
        assert self.disp_step > 0.0

        self.model = model

    def forward(self, h_views, v_views, i_views=None, d_views=None):
        """
        Forward ensamble network in an end-to-end fashion

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
        means = []
        logvars = []
        vars = []
        for shift_disp in np.arange(self.disp_min, self.disp_max,
                                    self.disp_step):
            if i_views is None or d_views is None:
                data = (h_views.clone(), v_views.clone())
            else:
                data = (h_views.clone(), v_views.clone(),
                        i_views.clone(), d_views.clone())

            shift = Shift(shift_disp)
            data = shift(data)

            output = self.model(*data)

            means.append(output['mean'] + shift_disp)
            logvars.append(output['logvar'])
            vars.append(torch.exp(output['logvar']))

        means = torch.stack(means)
        logvars = torch.stack(logvars)

        # TODO: choose disp and compute uncert
        min_index = torch.min(logvars, 0)[1]
        min_index = torch.stack([min_index] * means.shape[0])

        mean = means.gather(0, min_index)[0]
        logvar = logvars.gather(0, min_index)[0]

        # compute posterior
        b, h, w = means[0].shape
        disp = torch.zeros((b, len(means), h, w))
        disp[:, :, :, :] = torch.from_numpy(np.linspace(
            self.disp_min, self.disp_max, len(means))).view(1, -1, 1, 1)
        disp = disp.to(means[0].device)

        posterior = torch.zeros(
            (b, len(means), h, w)).to(means[0].device)

        for i in range(len(means)):
            print(posterior.shape)
            posterior += laplacian(disp, means[i], vars[i])

        posterior /= float(len(means))

        # compute variance as
        # average variance + average squared mean - square of average mean
        # var = torch.exp(logvars)
        # avg_var = torch.means(var, 0)
        # avg_sq_mean = torch.mean(means ** 2.0, 0)
        # sq_avg_mean = torch.mean(means, 0) ** 2.0
        # logvar = torch.log(avg_var + avg_sq_mean - sq_avg_mean)

        return {'mean': mean, 'logvar': logvar,
                'means': means, 'logvars': logvars, 'posterior': posterior}
