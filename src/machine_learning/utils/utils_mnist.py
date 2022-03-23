from typing import Tuple

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms


class MnistSet(Dataset):
    """
    custom MNIST dataset where each member is either
        * (tensor, equal tensor) for autoencoding
        * (noised tensor, tensor) for autoencoding and denoising
    """

    def __init__(self, X: Tensor, noised: bool = False, noise_factor: float = 0.0):
        """
        initializes MnistSet
        :param X: tensors
        :param noised: whether to noise each tensor x for (noised x, x)
        :param noise_factor: coefficient in x_noised = x + noise_factor * N(0, 1).sample(x.shape)
        """
        self.X = X
        self.noised = noised
        self.noise_factor = noise_factor
        self.p = Normal(0, 1)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        x = self.X[index]
        if self.noised:
            x_noised = x + self.noise_factor * self.p.sample(x.shape)
            return torch.clamp(x_noised, 0, 1), x
        else:
            return x, x

    def __len__(self) -> int:
        return len(self.X)


def load_mnist_data(
    dirpath: str, flatten: bool = False, val_size: float = 0.2, batch_size: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    loads PyTorch MNIST (images, labels) as (possibly flattened tensors, labels)
    :param dirpath: where images are
    :param flatten: whether to flatten each tensor (1, 28, 28) to (784,)
    :param val_size: val size
    :param batch_size: batch size
    :return: loaders (X_train_loader, X_val_loader, X_test_loader)
    """
    train_set, test_set = load_mnist_sets(dirpath, flatten=flatten)

    train_size = len(train_set)
    X_train, X_val = random_split(
        train_set, [int(train_size * (1 - val_size)), int(train_size * val_size)]
    )
    X_test = test_set

    X_train_loader = DataLoader(X_train, batch_size=batch_size)
    X_val_loader = DataLoader(X_val, batch_size=len(X_val))
    X_test_loader = DataLoader(X_test, batch_size=len(X_val))
    return X_train_loader, X_val_loader, X_test_loader


def load_mnist_data_for_autoencoder(
    dirpath,
    flatten: bool = False,
    val_size: float = 0.2,
    batch_size: int = 256,
    noised: bool = False,
    noise_factor: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    loads PyTorch MNIST (images, labels) as (possibly flattened noised tensors, tensors)
    :param dirpath: where images are
    :param flatten: whether to flatten each tensor (1, 28, 28) to (784,)
    :param val_size: val size
    :param batch_size: batch size
    :param noised: whether to noise each tensor x for (noised x, x)
    :param noise_factor: coefficient in x_noised = x + noise_factor * N(0, 1).sample(x.shape)
    :return: loaders (X_train_loader, X_val_loader, X_test_loader)
    """
    train_set, test_set = load_mnist_sets(dirpath, flatten=flatten)
    train_images = [image for image, _ in train_set]
    test_images = [image for image, _ in test_set]
    X_train = torch.stack(train_images)
    X_test = torch.stack(test_images)
    new_set = MnistSet(X_train, noised, noise_factor)
    new_size = len(new_set)
    new_train_set, new_val_set = random_split(
        new_set, [int(new_size * (1 - val_size)), int(new_size * val_size)]
    )
    new_test_set = MnistSet(X_test, noised, noise_factor)
    X_train_loader = DataLoader(new_train_set, batch_size=batch_size)
    X_val_loader = DataLoader(new_val_set, batch_size=batch_size)
    X_test_loader = DataLoader(new_test_set, batch_size=batch_size)
    return X_train_loader, X_val_loader, X_test_loader


def load_mnist_sets(
    dirpath: str, to_tensor: bool = True, flatten: bool = False
) -> Tuple[Dataset, Dataset]:
    """
    loads PyTorch MNIST (images, labels) as either (images, labels) or (possibly flattened tensors, labels)
    :param dirpath: where images are
    :param to_tensor: whether to transform each image to tensor (1, 28, 28)
    :param flatten: whether to flatten each tensor (1, 28, 28) to (784,)
    :return: datasets (train_set, test_set)
    """
    t_s = []
    if to_tensor is True:
        t_s.append(ToTensor())
    if flatten is True:
        t_s.append(transforms.Lambda(lambda x: torch.flatten(x)))
    t = transforms.Compose(t_s)
    train_set = MNIST(dirpath, train=True, download=True, transform=t)
    test_set = MNIST(dirpath, train=False, download=True, transform=t)
    return train_set, test_set
