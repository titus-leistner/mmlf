import sys
import threading
import queue

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from mmlf.data.hci4d import HCI4D


def optimize_patch(h_patch, v_patch, dict_dict, num_bins):
    clf = linear_model.Lasso(alpha=0.001, max_iter=10000, normalize=False, positive=False)
    posterior = np.zeros((num_bins,))

    for c in range(3):
        for patch in (h_patch, v_patch):
            patch = patch[:, c]
            patch = np.reshape(patch, (-1,)).copy()
            min_patch = np.min(patch)
            max_patch = np.max(patch)

            # normalize patch
            if (max_patch - min_patch) > 0.0:
                patch = (patch - min_patch) / (max_patch - min_patch)

            # center_patch = center[c, y - 4:y + 5, x - 4:x + 5]
            # plt.imshow(center_patch)
            # plt.show()

            # plt.imshow(h_patch)
            # plt.show()
            # plt.imshow(v_patch)
            # plt.show()

            clf.fit(dict_dict, patch)
            coefs = np.abs(np.reshape(clf.coef_[:-1], (num_bins, -1)))
            posterior[:] += np.sum(coefs, 1)

    return posterior


queue_patches = queue.Queue()
queue_posteriors = queue.Queue()


dataset = HCI4D(sys.argv[1], cache=False, load_dict=True)

for i in range(len(dataset)):
    data = dataset[i]
    h_views, v_views, _, _, center, gt, _, _, index, dict_dict, dict_labels, dict_range = data

    # cx = 169
    # cy = 107

    # h_views = h_views[:, :, cy - 8:cy + 8, cx - 8:cx + 8]
    # v_views = v_views[:, :, cy - 8:cy + 8, cx - 8:cx + 8]
    # center = center[:, cy - 8:cy + 8, cx - 8:cx + 8]
    # gt = gt[cy - 8:cy + 8, cx - 8:cx + 8]

    # plt.imsave('test.png', np.transpose(center, (1, 2, 0)))

    num_bins = dict_range.shape[1]

    def worker():
        while True:
            if queue_patches.empty():
                break

            item = queue_patches.get()
            h_patch, v_patch, x, y = item
            posterior = optimize_patch(h_patch, v_patch, dict_dict, num_bins)
            out = posterior, x, y
            queue_posteriors.put(out)
            print(f'Done for {x}, {y}')
            queue_patches.task_done()

    # transpose patches
    for i in range(dict_dict.shape[1]):

        patch = dict_dict[:, i].copy()
        patch = np.reshape(patch, (5, 5))
        patch = np.transpose(patch, (1, 0))
        dict_dict[:, i] = np.reshape(patch, (-1,))

        # disp = dict_labels[i, 0]
        # patch = dict_dict[:, i]
        # patch = np.reshape(patch, (5, 5))
        # plt.imsave(f'patch_test/patch_{disp}.png', patch)

    for y in range(2, h_views.shape[2] - 2):
        for x in range(2, h_views.shape[3] - 2):

            # if x == 158 and y == 101:
            h_patch = h_views[2:7, :, y, x - 2:x + 3].copy()
            v_patch = v_views[2:7, :, y - 2:y + 3, x].copy()

            # plt.imsave(f'img_test/h_{x}_{y}.png', np.transpose(h_patch, (0, 2, 1)))
            # plt.imsave(f'img_test/v_{x}_{y}.png', np.transpose(v_patch, (0, 2, 1)))
            queue_patches.put((h_patch, v_patch, x, y))

    for i in range(int(sys.argv[2])):
        threading.Thread(target=worker, daemon=True).start()

    queue_patches.join()

    # reconstruct posterior
    posterior = np.zeros((dict_range.shape[1], gt.shape[0], gt.shape[1]))

    while not queue_posteriors.empty():
        posterior_pos, x, y = queue_posteriors.get()
        posterior[:, y, x] = posterior_pos

    # normalize posterior
    posterior += 0.00001
    posterior /= np.sum(posterior, 0, keepdims=True)

    # get resulting disparities
    result = np.zeros_like(gt)
    for y in range(posterior.shape[1]):
        for x in range(posterior.shape[2]):
            argmax = np.argmax(posterior[:, y, x])
            disp = dict_range[0, argmax]
            result[y, x] = disp

            print(result[y, x], gt[y, x])

    # compute uncertainty
    logvar = (np.reshape(dict_range, (-1, 1, 1)) - np.reshape(result, (1, result.shape[0], result.shape[1]))) ** 2.0
    logvar *= posterior
    logvar = np.log(np.sum(logvar, 0))
    logvar = logvar.astype(np.float32)

    index = np.reshape(index, (1, 1))
    result = np.reshape(result, (1, result.shape[0], result.shape[1]))
    posterior = np.reshape(posterior, (1, posterior.shape[0],
                                       posterior.shape[1], posterior.shape[2])).astype(np.float32)
    logvar = np.reshape(logvar, (1, logvar.shape[0], logvar.shape[1]))

    dataset.save_batch('out_test', index, result=result, posterior=posterior, uncert=logvar)
