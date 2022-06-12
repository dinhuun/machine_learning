import numpy as np

from machine_learning.utils.utils_ecg import load_ecg_data, load_ecg_data_for_autoencoder


filepath = "~/Dinh/machine_learning/notebooks/data/ecg.csv"

n = 4998
m = 140
normal_n = 2919
anomalous_n = n - normal_n

test_size = 290
train_size = normal_n - test_size * 3
val_size = test_size * 2
batch_size = 32


def test_load_ecg_data():
    """
    tests load_ecg_data()
    """
    X, Y = load_ecg_data(filepath)
    assert X.shape == (n, m)
    assert Y.shape == (n,)


def test_load_ecg_data_for_autoencoder():
    """
    tests load_ecg_data_for_autoencoder()
    """
    normal_X_train_loader, normal_X_val_loader, normal_X_test_loader, anomalous_X_loader = load_ecg_data_for_autoencoder(filepath, False, val_size, test_size, batch_size)

    assert len(normal_X_train_loader.dataset) == train_size
    assert len(normal_X_val_loader.dataset) == val_size
    assert len(normal_X_test_loader.dataset) == test_size

    normal_X_batch, normal_X_hat_batch = next(iter(normal_X_train_loader))
    assert normal_X_batch.shape == (batch_size, m)
    np.testing.assert_array_equal(normal_X_batch, normal_X_hat_batch)

    anomalous_X_batch, _ = next(iter(anomalous_X_loader))
    assert anomalous_X_batch.shape == (batch_size, m)
