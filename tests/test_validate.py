import pytest
import numpy as np

# imports of the module you want to test and the dependencies
from mmlf.validate import sparsify


@pytest.fixture()
def resource():
    # setup for each test goes here

    yield

    # teardown for each test goes here


def test_auc():
    curve = np.array([1.0] * 101)
    assert sparsify.auc(curve, 0.01) == pytest.approx(1.0)

    curve = np.linspace(0.0, 1.0, 101)
    assert sparsify.auc(curve, 0.01) == pytest.approx(0.5)
