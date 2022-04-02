import numpy as np

from machine_learning.utils.utils_signals import stack_windows


def test_stack_windows():
    """
    tests stack_windows()
    """
    obs = np.array([0, 1, 2, 3, 4])

    X_1, Y_1 = stack_windows(obs)
    np.testing.assert_equal(X_1, np.array([[[0]], [[1]], [[2]], [[3]]]))
    np.testing.assert_equal(Y_1, np.array([1, 2, 3, 4]))

    X_2, Y_2 = stack_windows(obs, 2)
    np.testing.assert_equal(X_2, np.array([[[0, 1]], [[1, 2]], [[2, 3]]]))
    np.testing.assert_equal(Y_2, np.array([2, 3, 4]))

    X_2_vertical, Y_2_vertical = stack_windows(obs, 2, vertical=True)
    np.testing.assert_equal(X_2_vertical, np.array([[[0, 1]], [[1, 2]], [[2, 3]]]).reshape((3, 2, 1)))
    np.testing.assert_equal(Y_2_vertical, np.array([2, 3, 4]))
