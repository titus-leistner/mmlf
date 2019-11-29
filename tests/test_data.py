import pytest

from mmlf.data.hci4d import HCI4D


@pytest.fixture()
def resource():
    # setup for each test goes here

    yield

    # teardown for each test goes here


def test_hci4d():
    dataset = HCI4D('../lf-dataset/training')

    h_shape = dataset[0][0].shape
    v_shape = dataset[0][1].shape

    assert h_shape == v_shape
    assert len(h_shape) == 4
    assert h_shape[0] == 9
    assert h_shape[1] == 3

    center_shape = dataset[0][2].shape
    assert len(center_shape) == 3
    assert center_shape[0] == 3

    gt_shape = dataset[0][3].shape
    assert len(gt_shape) == 2

    index = dataset[0][4]
    assert len(index.shape) == 1
    assert index.shape[0] == 1
    assert index[0] == 0

    print(index)
