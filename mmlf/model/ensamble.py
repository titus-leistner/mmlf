import numpy as np
import torch
import torch.nn as nn

from .feed_forward import FeedForward
from ..data.hci4d import ContinuousShift


class Ensamble(nn.Module):
    """
    Use an ensamble with weight sharing to infer a per-pixel GMM
    """

    def __init__(self, model, val_disp_min, val_disp_max, val_disp_step):
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
        mean = []
        var = []
        for shift_disp in np.arange(self.disp_min, self.disp_max,
                                    self.disp_step):
            print(f'Running shift {shift_disp}...')

            if i_views is None or d_views is None:
                data = (h_views.clone(), v_views.clone())
            else:
                data = (h_views.clone(), v_views.clone(),
                        i_views.clone(), d_views.clone())

            shift = ContinuousShift(shift_disp)
            data = shift(data)

            disp, uncert = self.model(*data)

            mean.append(disp + shift_disp)
            var.append(torch.exp(uncert))

        mean = torch.stack(mean)
        var = torch.stack(var)

        # TODO: choose disp and compute uncert
        min_index = torch.min(var, 0)[1]
        min_index = torch.stack([min_index] * mean.shape[0])

        disp = mean.gather(0, min_index)[0]
        uncert = var.gather(0, min_index)[0]

        # compute variance as
        # average variance + average squared mean - square of average mean
        # avg_var = torch.mean(var, 0)
        # avg_sq_mean = torch.mean(mean ** 2.0, 0)
        # sq_avg_mean = torch.mean(mean, 0) ** 2.0
        # uncert = avg_var + avg_sq_mean - sq_avg_mean

        return disp, torch.sqrt(uncert)
