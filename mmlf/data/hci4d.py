import numpy as np
import torch
import torch.nn as nn

import os
import random
import math
import copy

from ..utils import pfm
from ..utils import dl
from ..utils import lf


def create_mask_margin(shape, margin=0):
    """
    Create a mask with a False margin

    :param shape: shape of mask
    :type shape: tuple

    :param margin: margin for last two dimenstions which gets assigned False
    :type margin: int
    """
    assert margin >= 0

    mask = torch.ones(shape, dtype=torch.bool)

    if margin > 0:
        mask[..., :margin, :] = False
        mask[..., -margin:, :] = False
        mask[..., :margin] = False
        mask[..., -margin:] = False

    return mask


def create_mask_texture(center, wsize, threshold):
    """
    Create a mask with False values for each pixel with a mean L1 distance to
    all neighboring pixels in a rolling window lower than a given threshold

    This implicitely adds a margin of wsize // 2 to the mask

    :param center: the center view
    :type center: torch.Tensor

    :param wsize: the window size
    :type wsize: int

    :param threshold: mean L1 threshold
    :type threshold: float
    """
    b, w, h = center.shape[0], center.shape[-1], center.shape[-2]

    # unfold and reshape to image
    mask = nn.functional.unfold(center, kernel_size=wsize, padding=wsize // 2)
    mask = mask.view(b, 3, -1, h, w)

    # subtract the center pixel and compute the MAE
    mask = torch.abs(mask - center.unsqueeze(2)).mean((1, 2))

    # apply the threshold
    mask = mask >= threshold

    # also mask the boundary
    mask = (mask.int() * create_mask_margin(mask.shape, wsize // 2).int())

    return mask


class HCI4D:
    """
    A class for the synthetic HCI 4D Light Field Dataset

    http://hci-lightfield.iwr.uni-heidelberg.de/
    """

    def __init__(self, root, nviews=(9, 9), transform=None, cache=False,
                 length=0, load_dict=False):
        """
        Loads the dataset

        :param root: root folder e.g. training in

              training
              |
              |---boxes
              |   |
              |   |---input_Cam00.png
              |   |---input_Cam01.png
              |   |---...
              |   |---gt_disp_lowres.pfm
              |---cotton
              |---...

        :type root: str

        :param nviews: number of views as (w_views, h_views) (default (9, 9))
        :type nviews: tuple(int, int)

        :param transform: optional transform to be applied
        :type transform: callable

        :param cache: cache all scenes to RAM first?
        :param cache: bool

        :param length: predefined length or 0 to use real length
        :type length: int
        """
        self.name = os.path.basename(root)
        self.scenes_names = [f.name for f in os.scandir(root) if f.is_dir()]
        self.scenes = [f.path for f in os.scandir(root) if f.is_dir()]
        self.nviews = nviews
        self.transform = transform
        self.length = length
        self.load_dict = load_dict

        self.cache = cache
        if cache:
            self.data = []
            self.cache_scenes()

    def load_scene(self, index):
        """
        Loads one scene

        :param index: scene index in range(0, len(dataset))
        :type index: int
        """
        import skimage.io

        scene = self.scenes[index]
        files = [f.name for f in os.scandir(scene)]
        imgs = [f for f in files if (f.endswith('.png') or f.endswith(
            '.jpg') or f.endswith('.jpeg')) and 'normals' not in f and
            'mask' not in f and 'objectids' not in f and 'unused' not in f
            and 'edges' not in f and 'specular' not in f]
        imgs.sort()

        # compute indices of cross setup
        w, h = self.nviews
        us = [int(h / 2) * w + i for i in range(h)]
        vs = [int(w / 2) + w * i for i in range(h)]

        # and for diagonals
        ids = [w - i - 1 + w * i for i in range(h)]
        ids.reverse()
        dds = [i + w * i for i in range(h)]

        # load views
        # horizontal
        h_views = []
        for i in us:
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            h_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        h_views = np.stack(h_views)
        h_views = h_views.transpose((0, 3, 1, 2))

        # vertical
        v_views = []
        for i in vs:
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            v_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        v_views = np.stack(v_views)
        v_views = v_views.transpose((0, 3, 1, 2))

        # increasing (rising) diagonal
        i_views = []
        for i in ids:
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            i_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        i_views = np.stack(i_views)
        i_views = i_views.transpose((0, 3, 1, 2))

        # decreasing (falling) diagonal
        d_views = []
        for i in dds:
            fname = imgs[i]
            fname = os.path.join(scene, fname)
            d_views.append(skimage.img_as_float(
                skimage.io.imread(fname)).astype(np.float32))
        d_views = np.stack(d_views)
        d_views = d_views.transpose((0, 3, 1, 2))

        # extract center view
        center = v_views[int(h / 2)].copy()

        # try to find the ground truth disparity pfm file
        pfms = [f for f in files if f.endswith('.pfm')]

        if len(pfms) > 1:
            # only load files with 'disp' in the name
            pfms = [f for f in pfms if 'disp' in f]
        if len(pfms) > 1:
            # only load lowres file
            pfms = [f for f in pfms if 'lowres' in f]
        if len(pfms) > 1:
            # only load center view
            pfms = [f for f in pfms if str(us[int(w / 2)]).zfill(3) in f]

        # load ground truth disparity
        gt = np.zeros_like(center[0])
        if len(pfms) > 0:
            gt = pfm.load(os.path.join(scene, pfms[0]))
            gt = np.flip(gt, 0).copy()

        # load mpis if existent
        if 'gt_mpi_lowres.npz' in files:
            mpi = np.load(os.path.join(scene, 'gt_mpi_lowres.npz'))['mpi']
            mpi = np.flip(mpi, 0).copy()
            mpi = mpi.transpose((2, 3, 0, 1))
            mpi[np.isnan(mpi)] = 0.0
            if mpi.shape[0] > 12:
                mpi = mpi[:12]
        else:
            # create one plane MPI from center view and ground truth
            mpi = np.zeros((1, 5, gt.shape[0], gt.shape[1]))
            mpi[0, :3, :, :] = center
            mpi[0, 3, :, :] = 1.0
            mpi[0, 4, :, :] = gt

            # set index
        index = np.atleast_1d(index)

        # load mask
        fname = os.path.join(scene, 'mask.png')
        if not os.path.exists(fname):
            mask = np.ones_like(gt, dtype=np.int)
        else:
            mask = (skimage.img_as_int(skimage.io.imread(fname))[:, :, 0] > 0).astype(np.int)

        # compute texture mask
        # no loss if no texture
        mask *= create_mask_texture(torch.from_numpy(center).unsqueeze(0), 23, 0.02).squeeze().int().numpy()

        if self.load_dict:
            import scipy.io as spio
            scene_dict = spio.loadmat(os.path.join(scene, 'data_k.mat'))['dic_k']
            dict_dict = scene_dict[0][0][0]
            dict_labels = scene_dict[0][0][1]
            dict_range = scene_dict[0][0][4]

            data = h_views, v_views, i_views, d_views, center, gt, mpi, mask, index, dict_dict, dict_labels, dict_range

        else:
            data = h_views, v_views, i_views, d_views, center, gt, mpi, mask, index
        return data

    def cache_scenes(self):
        """
        Loads all scenes to RAM
        """
        print('Caching dataset "{}"...'.format(self.name))
        for i, scene in enumerate(self.scenes):
            self.data.append(self.load_scene(i))

    def __len__(self):
        if self.length == 0:
            return len(self.scenes)

        return self.length

    def __getitem__(self, index):
        """
        Loads the next scene and returns it as
        (h_views, v_views, center, gt, index)
        where the views are tensors of shape (w or h, 3, h_image, w_image),
        center is the center view and gt is the ground truth
        of shape (h_img, w_img) or zeroes if the dataset does not provide it.
        Index is just a scalar list index as numpy.ndarray.

        :param index: scene index in range(0, len(dataset))
        :type index: int
        """
        index = index % len(self.scenes)

        if self.cache:
            data = self.data[index]
        else:
            data = self.load_scene(index)

        if self.transform:
            data = copy.deepcopy(data)
            data = self.transform(data)

        return data

    def save_batch(self, path, index, result=None, uncert=None, runtime=None,
                   gmm=None, nll=None, posterior=None):
        """
        Save the scene batch, ground truth and result to disk.
        Creates one one subdirectory in 'scenes/' for each scene in the batch.
        The results can be saved to 'ours/disp_maps/scene.pfm'.
        The runtime can also be saved to 'ours/runtimes/scene.txt'.

        :param path: the path to save scenes to
        :type path: str

        :param index: indices of the batch
        :type index: np.ndarray of shape (b, 1)

        :param result: batch of results
        :type result: np.ndarray of shape (b, h, w)

        :param uncert: batch of uncertainties
        :type uncert: np.ndarray of shape (b, h, w)

        :param runtime: runtime for the batch
        :type runtime: float

        :param gmm: GMM, containing means and vars for the batch
        :type gmm: np.ndarray of shape (2, K, b, h, w)

        :param nll: cluster distances
        :type nll: np.ndarray of shape (K, h, w)

        :param posterior: posterior distribution
        :type nll: np.ndarray of shape (K, h, w)
        """
        # create directories
        scenes = os.path.join(path, 'scenes')
        ours = os.path.join(path, 'ours')

        if not os.path.exists(scenes):
            os.makedirs(scenes)

        if not os.path.exists(ours):
            os.makedirs(ours)

        disp_maps = os.path.join(ours, 'disp_maps')
        if not os.path.exists(disp_maps):
            os.makedirs(disp_maps)

        runtimes = os.path.join(ours, 'runtimes')
        if not os.path.exists(runtimes):
            os.makedirs(runtimes)

        # for each scene
        for arr_i, i in enumerate(index.squeeze(1).tolist()):
            i = int(i)
            scene = self.scenes_names[i]

            # create directory
            scene_dir = os.path.join(scenes, scene)

            # get scene images
            h_views, v_views, i_views, d_views, center, gt, mpi, mask, _ = self.__getitem__(i)

            lf.save_views(scene_dir, h_views, v_views, i_views, d_views)
            dl.save_img(os.path.join(scene_dir, 'center.png'), center)
            dl.save_img(os.path.join(scene_dir, 'gt.png'), gt)
            dl.save_img(os.path.join(scene_dir, 'diff.png'), np.abs(gt - result))

            # save ground truth, results, uncertainties and/or runtimes
            gt_out = np.flip(gt.copy(), 0)
            pfm.save(os.path.join(scene_dir, 'gt.pfm'), gt_out)

            if result is not None:
                # save result as pfm
                res_out = np.flip(result[arr_i].copy(), 0)
                pfm.save(os.path.join(scene_dir, 'result.pfm'), res_out)
                pfm.save(os.path.join(disp_maps, f'{scene}.pfm'), res_out)

                # normalize and clip result
                disp_min = np.min(gt)
                disp_max = np.max(gt)

                res_img = result[arr_i].copy()
                res_img = (res_img - disp_min) / (disp_max - disp_min)
                res_img = np.clip(res_img, 0.0, 1.0)

                # save result as png
                dl.save_img(os.path.join(
                    scene_dir, 'result.png'), res_img)

            if uncert is not None:
                # save uncertainty as pfm
                uncert_out = np.flip(uncert[arr_i].copy(), 0)
                pfm.save(os.path.join(scene_dir, 'uncert.pfm'), uncert_out)
                # pfm.save(os.path.join(
                #     disp_maps, f'{scene}_uncert.pfm'), uncert_out)

                # and as png
                dl.save_img(os.path.join(
                    scene_dir, 'uncert.png'), uncert[arr_i])

            # save GMM
            if gmm is not None:
                np.save(os.path.join(scene_dir, 'gmm.npy'), gmm[:, :, arr_i])

            if nll is not None:
                np.save(os.path.join(scene_dir, 'nll.npy'),
                        nll[arr_i, ...])

            if posterior is not None:
                np.save(os.path.join(scene_dir, 'posterior.npy'),
                        posterior[arr_i, ...])
            # if mask is not None:
            #     dl.save_img(os.path.join(
            #         scene_dir, 'mask.png'), mask)

            if runtime is not None:
                # devide runtime by batchsize and output
                b = float(index.shape[0])
                with open(os.path.join(runtimes, f'{scene}.txt'), 'w') as f:
                    f.write(str(runtime / b))


class Zoom:
    """
    Rescale the input lighfield according to some factor
    """

    def __init__(self, factor):
        """
        :param factor: desired zoom factor (e.g. 0.5 for half the image size)
        :type factor: float
        """
        assert isinstance(factor, float)
        self.factor = factor

    def __call__(self, data):
        """
        Rescale the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, mpi, mask, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the scaled lightfield data
        """
        from scipy import ndimage

        data = list(data)
        for i in range(len(data)):
            shape = data[i].shape
            if len(shape) < 2 or shape[-1] <= 1 or shape[-2] <= 1:
                continue
            zoom = [1.0] * len(data[i].shape)
            zoom[-2] = zoom[-1] = self.factor
            data[i] = ndimage.zoom(data[i], zoom, order=0)

        # correct ground truth
        if len(data) > 5:
            data[5] *= float(self.factor)

        if len(data) > 6:
            data[6][:, 4, :, :] *= float(self.factor)

        return tuple(data)


class RandomZoom:
    """
    Rescale the lightfield randomly
    """

    def __init__(self, min_scale=0.5, max_scale=1.0):
        """
        :param min_scale: minimum possible scale
        :type min_scale: float

        :param max_scale: maximum possible scale
        :type max_scale: float
        """
        self.interval = (min_scale, max_scale)

    def __call__(self, data):
        factor = random.uniform(self.interval[0], self.interval[1])

        zoom = Zoom(factor)

        return zoom(data)


class DownSampling:
    """
    Downsample the light field
    """

    def __init__(self, factor):
        """
        :param factor: downsampling factor 1/factor * width is the new width
        :type factor: int
        """
        self.factor = factor

    def __call__(self, data):
        data = list(data)
        for i in range(len(data)):
            shape = data[i].shape
            if len(shape) < 2 or shape[-1] <= 1 or shape[-2] <= 1:
                continue
            data[i] = data[i][..., ::self.factor, ::self.factor]

        # correct ground truth
        if len(data) > 5:
            data[5] /= float(self.factor)

        if len(data) > 6:
            data[6][:, 4, :, :] /= float(self.factor)

        return tuple(data)


class RandomDownSampling:
    """
    Downsample the light field randomly
    """

    def __init__(self, max_factor=4):
        """
        :param max_factor: maximum downsampling factor
        :type max_factor: int
        """
        self.max_factor = max_factor

    def __call__(self, data):
        factor = random.randint(1, self.max_factor)

        down_sampling = DownSampling(factor)

        return down_sampling(data)


class Crop:
    """
    Crop the input lightfield to a given size with a given position
    """

    def __init__(self, size, pos):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int

        :param pos: crop position(s) tuple (y, x)
        :type pos: tuple(y, x)
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)
        assert isinstance(pos, tuple)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

        self.pos = pos

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        data = list(data)
        h, w = self.size
        y, x = self.pos

        for i in range(len(data)):
            shape = data[i].shape
            if len(shape) < 2 or shape[-1] <= 1 or shape[-2] <= 1:
                continue

            data[i] = data[i][..., y:y + h, x:x + w]

        return tuple(data)


class CenterCrop:
    """
    Crop by cutting off equal margins input lightfield to a given size
    """

    def __init__(self, size):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        h = data[0].shape[-2]
        w = data[0].shape[-1]

        y = int((h - self.size[0]) / 2)
        x = int((w - self.size[1]) / 2)

        assert y >= 0 and x >= 0

        crop = Crop(self.size, (y, x))

        return crop(data)


class RandomCrop:
    """
    Crop patches randomly to a given size
    """

    def __init__(self, size, pad=0):
        """
        :param size: output size. Tuple (height, width) or int for square size
        :type size: tuple(h, w) or int

        :param pad: optional padding to not choose samples from
        :tyoe pad: int
        """
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

        assert isinstance(pad, int)
        self.pad = pad

    def __call__(self, data):
        """
        Crop the lightfield data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
                     or any other sequence of images or image stacks
        :type data: tuple

        :returns: the cropped lightfield data
        """
        h = data[0].shape[-2]
        w = data[0].shape[-1]

        assert h > self.size[0]
        assert w > self.size[1]

        y = random.randint(self.pad, h - self.size[0] - self.pad)
        x = random.randint(self.pad, w - self.size[1] - self.pad)

        crop = Crop(self.size, (y, x))

        return crop(data)


class RedistColor:
    """
    Randomly redistribute color
    """

    def __call__(self, data):
        """
        Redistribute the lightfield color data.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        # create redistribution matrix
        mat = np.zeros((3, 3))
        mat[0, 0] = random.uniform(0.0, 1.0)
        mat[0, 1] = random.uniform(0.0, 1.0 - mat[0, 0])
        mat[1, 0] = random.uniform(0.0, 1.0 - mat[0, 0])
        mat[1, 1] = random.uniform(0.0, 1.0 - max(mat[0, 1], mat[1, 0]))

        mat[0, 2] = 1.0 - mat[0, 0] - mat[0, 1]
        mat[1, 2] = 1.0 - mat[1, 0] - mat[1, 1]
        mat[2, 0] = 1.0 - mat[0, 0] - mat[1, 0]
        mat[2, 1] = 1.0 - mat[0, 1] - mat[1, 1]
        mat[2, 2] = mat[0, 0] + mat[0, 1] + mat[1, 0] + mat[1, 1] - 1.0

        for i in range(min(5, len(data))):
            if data[i] is None:
                continue
            if isinstance(data[i], np.ndarray):
                stack = data[i].copy()
            else:
                stack = data[i].clone()

            assert stack.shape[-3] == 3

            data[i][..., 0, :, :] = mat[0, 0] * stack[..., 0, :, :]
            data[i][..., 0, :, :] += mat[0, 1] * stack[..., 1, :, :]
            data[i][..., 0, :, :] += mat[0, 2] * stack[..., 2, :, :]

            data[i][..., 1, :, :] = mat[1, 0] * stack[..., 0, :, :]
            data[i][..., 1, :, :] += mat[1, 1] * stack[..., 1, :, :]
            data[i][..., 1, :, :] += mat[1, 2] * stack[..., 2, :, :]

            data[i][..., 2, :, :] = mat[2, 0] * stack[..., 0, :, :]
            data[i][..., 2, :, :] += mat[2, 1] * stack[..., 1, :, :]
            data[i][..., 2, :, :] += mat[2, 2] * stack[..., 2, :, :]

        return tuple(data)


class Contrast:
    """
    Randomly change Contrast
    """

    def __init__(self, level=0.9):
        """
        :param level: level of change
        :type level: float
        """
        assert isinstance(level, float)
        self.level = level

    def __call__(self, data):
        """
        Change the lightfields contrast.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        alpha = random.uniform(-self.level, self.level) + 1.0
        mean = data[0].mean()

        data = list(data)
        for i in range(min(5, len(data))):
            if data[i] is None:
                continue

            data[i] = data[i] * alpha + mean * (1.0 - alpha)

        return tuple(data)


class Brightness:
    """
    Randomly change Brightness
    """

    def __init__(self, level=0.9):
        """
        :param level: level of change
        :type level: float
        """
        assert isinstance(level, float)
        self.level = level

    def __call__(self, data):
        """
        Change the lightfields brightness.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        alpha = random.uniform(-self.level, self.level) + 1.0

        data = list(data)
        for i in range(min(5, len(data))):
            if data[i] is None:
                continue

            data[i] = data[i] * alpha

        return tuple(data)


class Noise:
    """
    Add random Gaussian noise
    """

    def __init__(self, stdev=0.01):
        """
        :param stdev: standard deviation of noise
        :type stdev: float
        """
        assert isinstance(stdev, float)
        self.stdev = stdev

    def __call__(self, data):
        """
        Add random Gaussian noise.

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the recolored lightfield data
        """
        data = list(data)
        for i in range(min(5, len(data))):
            if data[i] is None:
                continue

            noise = np.random.normal(scale=self.stdev, size=data[i].shape)
            data[i] += noise

        return tuple(data)


class IntegerShift:
    """
    Shift the lightfield according to some integer disparity
    """

    def __init__(self, disp):
        """
        :param disp: discrete disparity that should be zero afterwards
        :type disp: int
        """
        assert isinstance(disp, int)
        self.disp = disp

    def __call__(self, data):
        """
        Shift the lightfield

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the shifted lightfield data
        """
        data = list(data)
        # test if numpy or pytorch
        cat = np.concatenate
        if not isinstance(data[0], np.ndarray):
            from torch import cat as torch_cat
            cat = torch_cat

        h_views = data[0]
        v_views = data[1]
        i_views = data[2]
        d_views = data[3]

        w = h_views.shape[-4]
        h = v_views.shape[-4]
        hw = int(w / 2)
        hh = int(h / 2)

        for i in range(w):
            shift = self.disp * (i - hw)
            h_views[..., i, :, :, :] = cat(
                [h_views[..., i, :, :, -shift:],
                 h_views[..., i, :, :, :-shift]], -1)
            i_views[..., i, :, :, :] = cat(
                [i_views[..., i, :, :, -shift:],
                 i_views[..., i, :, :, :-shift]], -1)
            d_views[..., i, :, :, :] = cat(
                [d_views[..., i, :, :, -shift:],
                 d_views[..., i, :, :, :-shift]], -1)

        for i in range(h):
            shift = self.disp * (i - hh)
            v_views[..., i, :, :, :] = cat(
                [v_views[..., i, :, -shift:, :],
                 v_views[..., i, :, :-shift, :]], -2)
            i_views[..., i, :, :, :] = cat(
                [i_views[..., i, :, shift:, :],
                 i_views[..., i, :, :shift, :]], -2)
            d_views[..., i, :, :, :] = cat(
                [d_views[..., i, :, -shift:, :],
                 d_views[..., i, :, :-shift, :]], -2)

        # correct ground truth
        if len(data) > 5:
            data[5] -= float(self.disp)

        if len(data) > 6:
            data[6][:, 4, :, :] -= float(self.disp)

        return tuple(data)


class Shift:
    """
    Shift the lightfield according to some continuous disparity
    """

    def __init__(self, disp):
        """
        :param disp: disparity that should be zero afterwards
        :type disp: float
        """
        assert isinstance(disp, float)
        self.disp = disp

    def __call__(self, data):
        """
        Shift the lightfield

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the shifted lightfield data
        """
        data = list(data)
        # test if numpy or pytorch
        cat = np.concatenate
        if not isinstance(data[0], np.ndarray):
            from torch import cat as torch_cat
            cat = torch_cat

        h_views = data[0]
        v_views = data[1]
        i_views = data[2]
        d_views = data[3]

        w = h_views.shape[-4]
        h = v_views.shape[-4]
        hw = int(w / 2)
        hh = int(h / 2)

        for i in range(w):
            alpha, shift0 = math.modf(self.disp * (i - hw))
            alpha = abs(alpha)
            shift1 = shift0 + math.copysign(1.0, shift0)
            shift0 = int(shift0)
            shift1 = int(shift1)

            h_views[..., i, :, :, :] = cat(
                [h_views[..., i, :, :, -shift0:],
                 h_views[..., i, :, :, :-shift0]], -1) * (1.0 - alpha) + cat(
                [h_views[..., i, :, :, -shift1:],
                 h_views[..., i, :, :, :-shift1]], -1) * alpha

            i_views[..., i, :, :, :] = cat(
                [i_views[..., i, :, :, -shift0:],
                 i_views[..., i, :, :, :-shift0]], -1) * (1.0 - alpha) + cat(
                [i_views[..., i, :, :, -shift1:],
                 i_views[..., i, :, :, :-shift1]], -1) * alpha

            d_views[..., i, :, :, :] = cat(
                [d_views[..., i, :, :, -shift0:],
                 d_views[..., i, :, :, :-shift0]], -1) * (1.0 - alpha) + cat(
                [d_views[..., i, :, :, -shift1:],
                 d_views[..., i, :, :, :-shift1]], -1) * alpha

        for i in range(h):
            alpha, shift0 = math.modf(self.disp * (i - hh))
            alpha = abs(alpha)
            shift1 = shift0 + math.copysign(1.0, shift0)
            shift0 = int(shift0)
            shift1 = int(shift1)

            v_views[..., i, :, :, :] = cat(
                [v_views[..., i, :, -shift0:, :],
                 v_views[..., i, :, :-shift0, :]], -2) * (1.0 - alpha) + cat(
                [v_views[..., i, :, -shift1:, :],
                 v_views[..., i, :, :-shift1, :]], -2) * alpha

            i_views[..., i, :, :, :] = cat(
                [i_views[..., i, :, shift0:, :],
                 i_views[..., i, :, :shift0, :]], -2) * (1.0 - alpha) + cat(
                [i_views[..., i, :, shift1:, :],
                 i_views[..., i, :, :shift1, :]], -2) * alpha

            d_views[..., i, :, :, :] = cat(
                [d_views[..., i, :, -shift0:, :],
                 d_views[..., i, :, :-shift0, :]], -2) * (1.0 - alpha) + cat(
                [d_views[..., i, :, -shift1:, :],
                 d_views[..., i, :, :-shift1, :]], -2) * alpha

        # correct ground truth
        if len(data) > 5:
            data[5] -= float(self.disp)

        if len(data) > 6:
            data[6][:, 4, :, :] -= float(self.disp)

        return tuple(data)


class RandomShift:
    """
    Randomly shift the lightfield an correct the ground truth accordingly
    """

    def __init__(self, disp_range):
        """
        :param disp_range: interval of disparities for shifts. tuple(min, max)
        or positive int for a range of (-disp_range, +disp_range)
        :type disp_range: tuple(float, float) or float
        """
        assert isinstance(disp_range, float) or (
            isinstance(disp_range, tuple) and len(disp_range) == 2)

        self.disp_range = disp_range

        if not isinstance(disp_range, tuple):
            assert disp_range > 0
            self.disp_range = (-disp_range, disp_range)

    def __call__(self, data):
        """
        Shift the lightfield randomly and correct the ground truth accordingly

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the shifted lightfield data
        """
        # shift randomly
        disp = random.uniform(self.disp_range[0], self.disp_range[1])

        shift = Shift(disp)
        data = shift(data)

        return data


class Rotate90:
    """
    Rotate the lightfield by 90 degrees
    """

    def __init__(self):
        pass

    def __call__(self, data):
        """
        Rotate the lightfield by 90 degrees

        :param data: Sequence containing (h_views, v_views, center, gt, index)
        :type data: tuple

        :returns: the rotated lightfield data
        """
        view = np.transpose
        flip = np.flip
        if not isinstance(data[0], np.ndarray):
            from torch import flip as torch_flip
            def view(t, s): return t.permute(s)
            def flip(t, a): return torch_flip(t, (a,))

        data = list(data)

        for i in range(min(7, len(data))):
            axis = list(range(len(data[i].shape)))
            axis[-1], axis[-2] = axis[-2], axis[-1]
            data[i] = flip(view(data[i], axis), -2).copy()

        if len(data) > 1:
            data[0], data[1] = data[1], data[0]
            data[1] = flip(data[1], -4)

        if len(data) > 3 and data[2] is not None and data[3] is not None:
            data[2], data[3] = data[3], data[2]
            data[3] = flip(data[3], -4)

        return tuple(data)


class RandomRotate:
    """
    Rotate the lightfield by 90 degrees
    """

    def __init__(self):
        self.rot = Rotate90()

    def __call__(self, data):
        r = random.randint(0, 3)

        for i in range(r):
            data = self.rot(data)

        return data
