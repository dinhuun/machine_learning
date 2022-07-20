import os

import torch
from PIL.Image import Image

from machine_learning.utils.utils_mnist import (
    MnistSet,
    load_mnist_data,
    load_mnist_data_for_autoencoder,
    load_mnist_sets,
)


current_dirpath = os.path.dirname(__file__)
dirpath = os.path.join(current_dirpath, "../../../notebooks/data/pytorch_mnist")
batch_size = 256
train_set_size = 60000
test_set_size = 10000
val_size = 0.2
X_train_size = train_set_size * (1 - val_size)
X_val_size = train_set_size * val_size
X_test_size = test_set_size
h = w = 28


def test_load_mnist_data():
    """
    tests load_mnist_data()
    """
    # that it yields datasets of tensors and labels of expected sizes
    X_train_loader, X_val_loader, X_test_loader = load_mnist_data(
        dirpath, False, val_size, batch_size
    )
    assert len(X_train_loader.dataset) == train_set_size * (1 - val_size)
    assert len(X_val_loader.dataset) == train_set_size * val_size
    assert len(X_test_loader.dataset) == test_set_size
    X_batch, Y_batch = next(iter(X_train_loader))
    assert X_batch.shape == (batch_size, 1, h, w)
    assert Y_batch.shape == (batch_size,)

    # that it flattens tensors as expected
    X_train_loader_flattened, _, _ = load_mnist_data(
        dirpath, True, val_size, batch_size
    )
    X_batch_flattened, _ = next(iter(X_train_loader_flattened))
    assert X_batch_flattened.shape == (batch_size, h * w)


def test_load_mnist_data_for_autoencoder():
    """
    tests load_mnist_data_for_autoencoder()
    """
    # that it yields default datasets of tensors of expected sizes
    X_train_loader, X_val_loader, X_test_loader = load_mnist_data_for_autoencoder(
        dirpath
    )
    assert len(X_train_loader.dataset) == X_train_size
    assert len(X_val_loader.dataset) == X_val_size
    assert len(X_test_loader.dataset) == X_test_size
    X_batch, Y_batch = next(iter(X_train_loader))
    assert X_batch.shape == (batch_size, 1, h, w)
    assert Y_batch.shape == (batch_size, 1, h, w)
    assert torch.equal(X_batch, Y_batch)

    # that it flattens tensors as expected
    X_train_loader_flattened, _, _ = load_mnist_data_for_autoencoder(
        dirpath, flatten=True
    )
    X_batch_flattened, Y_batch_flattened = next(iter(X_train_loader_flattened))
    assert X_batch_flattened.shape == (batch_size, h * w)
    assert Y_batch_flattened.shape == (batch_size, h * w)
    assert torch.equal(X_batch_flattened, Y_batch_flattened)

    # that it noises images as expected
    X_train_loader_noised, _, _ = load_mnist_data_for_autoencoder(
        dirpath, noised=True, noise_factor=0.5
    )
    X_batch_noised, Y_batch_noised = next(iter(X_train_loader_noised))
    assert not torch.equal(X_batch_noised, Y_batch_noised)
    assert torch.all(torch.ge(X_batch_noised, 0))
    assert torch.all(torch.le(X_batch_noised, 1))

    # that it flattens and noises images as expected
    X_train_loader_flattened_noised, _, _ = load_mnist_data_for_autoencoder(
        dirpath, flatten=True, noised=True, noise_factor=0.5
    )
    X_batch_flattened_noised, Y_batch_flattened_noised = next(
        iter(X_train_loader_flattened_noised)
    )
    assert X_batch_flattened_noised.shape == (batch_size, h * w)
    assert Y_batch_flattened_noised.shape == (batch_size, h * w)
    assert not torch.equal(X_batch_flattened_noised, Y_batch_flattened_noised)
    assert torch.all(torch.ge(X_batch_flattened_noised, 0))
    assert torch.all(torch.le(X_batch_flattened_noised, 1))


def test_load_mnist_sets():
    """
    tests load_mnist_sets()
    """
    # that it yields datasets of images and labels of expected sizes
    train_set, test_set = load_mnist_sets(dirpath, to_tensor=False)
    assert len(train_set) == train_set_size
    assert len(test_set) == test_set_size
    x, y = train_set[0]
    assert isinstance(x, Image)
    assert isinstance(y, int)

    # that it tensors images as expected
    train_set_tensor, _ = load_mnist_sets(dirpath, to_tensor=True)
    x_tensor, _ = train_set_tensor[0]
    assert x_tensor.shape == (1, h, w)

    # that it flattens tensors as expected
    train_set_tensor_flattened, _ = load_mnist_sets(
        dirpath, to_tensor=True, flatten=True
    )
    x_tensor_flattened, _ = train_set_tensor_flattened[0]
    assert x_tensor_flattened.shape == (784,)


def test_MnistSet():
    """
    tests MnistSet
    """
    n_samples = 10
    n_features = 5
    X = torch.rand((n_samples, n_features))
    # that __init__() works
    dataset_0 = MnistSet(X)
    # that __getitem__() works
    x_0_0, x_0_1 = dataset_0[0]
    assert torch.equal(x_0_0, x_0_1)
    # that __len__() works
    assert len(dataset_0) == n_samples

    dataset_1 = MnistSet(X, noised=True, noise_factor=0.5)
    x_1_0, x_1_1 = dataset_1[0]
    # that it noises sample
    assert not torch.equal(x_1_0, x_1_1)
    # that noised sample is in expected range
    assert torch.all(torch.ge(x_1_0, 0))
    assert torch.all(torch.le(x_1_0, 1))
