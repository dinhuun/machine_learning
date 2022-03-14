import numpy as np
import pytest
from scipy.special import expit

from machine_learning.utils.utils_misc import hard_arg_max, soft_arg_max, raise_not_implemented_error

x_0 = 0.0
x_1 = 0.1
X = np.array([x_0, x_1])


def test_hard_arg_max():
    """
    tests hard_arg_max()
    """
    hard = hard_arg_max(X)
    np.testing.assert_equal(hard, np.array([0, 1]))


def test_soft_arg_max():
    """
    tests soft_arg_max()
    soft_arg_max(), with soft_arg_max([0, x]), generalizes (1 - sigmoid(x), sigmoid(x))
    """
    soft = soft_arg_max(X)
    y_1 = expit(np.array(x_1))
    np.testing.assert_equal(soft, np.array([1 - y_1, y_1]))


def test_raise_not_implemented_error():
    """
    tests raise_not_implemented_error()
    """
    with pytest.raises(NotImplementedError):
        raise_not_implemented_error("object", "object_name")
