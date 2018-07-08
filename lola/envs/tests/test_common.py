import numpy as np

from lola.envs.common import OneHot


def test_onehot():
    space = OneHot(3)

    # Do checks
    correct_samples = [
        space.sample(),
        np.array([1, 0, 0], dtype='int'),
    ]
    for s in correct_samples:
        assert space.contains(s)

    incorrect_samples = [
        np.array([0, 0, 0]),
        np.array([1, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0.5, 0.5, 0]),
    ]
    for s in incorrect_samples:
        assert not space.contains(s)
