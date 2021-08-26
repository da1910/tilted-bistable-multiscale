import pytest
import numpy as np
from util import bisect_left_naive


@pytest.fixture
def positive_uniform():
    yield np.array(range(0, 10))


@pytest.fixture
def negative_uniform():
    yield np.flip(np.array(range(0, 10)))


@pytest.mark.parametrize(['target', 'expected'], [
    (0, 0),
    (1, 1),
    (8, 8),
])
def test_exact_match(positive_uniform, target, expected):
    obtained = bisect_left_naive(positive_uniform, target)
    assert expected == obtained


@pytest.mark.parametrize(['target', 'expected'], [
    (0.5, 0),
    (1.5, 1),
    (8.999, 8)
])
def test_in_interval(positive_uniform, target, expected):
    obtained = bisect_left_naive(positive_uniform, target)
    assert expected == obtained


@pytest.mark.parametrize(['target', 'expected'], [
    (0, None),
    (1, 8),
    (8, 1)
])
def test_exact_match(negative_uniform, target, expected):
    obtained = bisect_left_naive(negative_uniform, target)
    assert expected == obtained


@pytest.mark.parametrize(['target', 'expected'], [
    (0.5, 8),
    (1.5, 7),
    (8.999, 0)
])
def test_in_interval(negative_uniform, target, expected):
    obtained = bisect_left_naive(negative_uniform, target)
    assert expected == obtained


@pytest.mark.parametrize('target', [-100, -0.001, 10, 100])
def test_outside_interval(positive_uniform, target):
    obtained = bisect_left_naive(positive_uniform, target)
    assert obtained is None
