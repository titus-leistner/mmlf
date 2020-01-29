import torch
import numpy as np

import warnings


class ModelSaver:
    """
    Class to save models.
    Call an instance to save your model during training
    """

    def __init__(self, only_best=False):
        """
        :param only_best: save only the best model (lowest loss) so far?
        :type only_best: bool
        """
        self.only_best = only_best
        self.best_loss = None

    def __call__(self, fname, model, optimizer=None, hyper_parameters=None,
                 epoch=None, iteraration=None, loss=None):
        """
        Save a model

        :param fname: path to the .pt file
        :type fname: str

        :param model: the model
        :type model: torch.nn.Module

        :param optimizer: the optimizer
        :type optimizer: torch.optim.Optimizer

        :param hyper_parameters: dictionary containing hyper_parameters
        :type hyper_parameters: dict

        :param epoch: the current training epoch
        :type epoch: int

        :param iteration: the current training iteration (number of batch)
        :type iteraration: int

        :param loss: the current validation loss
        :type loss: float
        """
        if self.only_best and loss is not None:
            if self.best_loss is not None and self.best_loss < loss:
                return
            self.best_loss = loss

        # get model state dict, try model.module (for DataParallel) first
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        optimizer_state_dict = None

        if optimizer is not None:
            optimizer_state_dict = optimizer.state_dict()

        state = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'hyper_parameters': hyper_parameters,
            'epoch': epoch,
            'iteration': iteraration,
            'loss': loss
        }

        torch.save(state, fname)


def save_img(fname, arr):
    """
    Save the numpy-array as image with the given filename

    :param fname: the filename to save to
    :type fname: str

    :param arr: the image
    :type arr: numpy.ndarray with shape (3, h, w) -> rgb or (h, w) -> greyscale
    """
    import skimage.io

    # convert, if necessary
    if not isinstance(arr, np.ndarray):
        arr = arr.detach().cpu().numpy()

    # normalize, if necessary
    a_min = np.min(arr)
    a_max = np.max(arr)

    if(a_min < 0.0 or a_max > 1.0):
        arr = (arr - a_min) / (a_max - a_min)

    if len(arr.shape) == 3:
        arr = np.transpose(arr, (1, 2, 0))

    # save image
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        skimage.io.imsave(fname, skimage.img_as_ubyte(arr))


def reg_to_class(arr, start, stop, n_steps):
    """
    Convert tensor of continuous numbers to discrete one-hot-encoding

    :param arr: input tensor/array
    :type arr: torch.tensor

    :param start: lower bound of valid range
    :type start: float

    :param stop: upper bound of valid range
    :type stop: float

    :param: n_steps: number of classes
    :type: n_steps: int
    """
    step = (stop - start) / n_steps
    result = torch.linspace(start, stop, n_steps).view((1, -1, 1, 1))
    arr = arr.unsqueeze(1)

    result = (torch.abs(result - arr) < step / 2.0).float()

    return result


def class_to_reg(arr, start, stop, n_steps):
    """
    Convert tensor of discrete one-hot-encodings to continuous regressions

    :param arr: input tensor/array
    :type arr: torch.tensor

    :param start: lower bound of valid range
    :type start: float

    :param stop: upper bound of valid range
    :type stop: float

    :param: n_steps: number of classes
    :type: n_steps: int
    """
    result = torch.linspace(start, stop, n_steps).view(
        (1, -1, 1, 1)).to(arr.device)
    result = torch.sum(result * arr, 1)

    return result


class BatchIter:
    """
    Process each image in a batch iteratively
    """

    def __init__(self, net):
        """
        :param net: the pytorch network
        :type net: torch.nn.Module

        :param args: tuple with arguments
        :type args: tuple
        """
        assert(isinstance(net, torch.nn.Module))

        self.net = net

    def __call__(self, *args):
        """
        Run the network once for each image in the batch

        :param args: tuple with arguments of type torch.tensor
        :type args: tuple
        """
        for arg in args:
            assert(isinstance(arg, torch.Tensor))

        results = []

        b = args[0].shape[0]
        for i in range(b):
            net_args = []
            for arg in args:
                net_args.append(arg[i:i+1])

            results.append(self.net(*net_args))

        out = []
        for j in range(len(results[0])):
            tensor = []
            for i in range(b):
                tensor.append(results[i][j])
            tensor = torch.cat(tensor, 0)

            out.append(tensor)

        return out
