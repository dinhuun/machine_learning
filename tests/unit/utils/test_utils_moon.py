import numpy as np

from machine_learning.utils.utils_moons import MoonsSet, load_moons_data

n_samples = 100
n_moons = 2
n_moons_train = 5
n_moons_val = 2
n_moons_test = 1


def test_load_moons_data():
    """
    tests load_moons_data()
    """
    X_train_loader, X_val_loader, X_test_loader = load_moons_data(
        n_samples, n_moons_train, n_moons_val, n_moons_test
    )
    assert len(X_train_loader.dataset) == n_samples * n_moons_train
    assert len(X_val_loader.dataset) == n_samples * n_moons_val
    assert len(X_test_loader.dataset) == n_samples * n_moons_test
    X_batch, Y_batch = next(iter(X_train_loader))
    assert X_batch.shape == (n_samples, 2)
    np.testing.assert_equal(X_batch.numpy(), Y_batch.numpy())


def test_MoonsSet():
    """
    tests MoonsSet
    """
    # that __init__() works
    moons_set = MoonsSet(n_samples, n_moons)
    # that __getitem__() works
    x_0_0, x_0_1 = moons_set[0]
    assert x_0_0.shape == (2,)
    np.testing.assert_equal(x_0_0.numpy(), x_0_1.numpy())
    # that __len__() works
    assert len(moons_set) == n_samples * n_moons
