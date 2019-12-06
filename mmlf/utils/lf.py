import os

from . import dl


def save_views(scene_dir, h_views, v_views, i_views=None, d_views=None):
    """
    Save all views of two view stacks to a given scene directory

    :param scene_dir: the directory to save the images to
    :type scene_dir: str

    :param h_views: the horizontal view stack
    :type h_views: numpy.ndarray of shape (n, h, w)

    :param v_views: the vertical view stack
    :type v_views: numpy.ndarray of shape (n, h, w)

    :param i_views: the increasing diagonal view stack
    :type i_views: numpy.ndarray of shape (n, h, w)

    :param d_views: the decreasing diagonal view stack
    :type d_views: numpy.ndarray of shape (n, h, w)
    """
    # remove batch dimension if necessary
    if len(h_views.shape) == 5:
        h_views = h_views[0]

    if len(v_views.shape) == 5:
        v_views = v_views[0]

    # recreate directory
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    # save scene images
    for j in range(h_views.shape[0]):
        dl.save_img(os.path.join(
            scene_dir, f'view_h_{j}.png'), h_views[j])

    for j in range(v_views.shape[0]):
        dl.save_img(os.path.join(
            scene_dir, f'view_v_{j}.png'), v_views[j])

    if i_views is not None:
        for j in range(i_views.shape[0]):
            dl.save_img(os.path.join(
                scene_dir, f'view_i_{j}.png'), i_views[j])

    if d_views is not None:
        for j in range(d_views.shape[0]):
            dl.save_img(os.path.join(
                scene_dir, f'view_d_{j}.png'), d_views[j])
