import pytest

# imports of the module you want to test and the dependencies
# import mmlf.model


@pytest.fixture()
def resource():
    # setup for each test goes here

    yield

    # teardown for each test goes here


def test_one():
    assert 1 == 1


def test_two():
    assert 1 == 1
